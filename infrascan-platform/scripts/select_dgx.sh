#!/usr/bin/env bash
# Pick the best DGX host for a workload based on registry.json + live probe.
#
# Usage:
#   select.sh                  → print one alias (best host) on stdout
#   select.sh --list           → print all hosts with live status (stderr table)
#   select.sh --probe HOST     → print status for one host
#   select.sh --json           → print the chosen host's full registry record
#
# Selection rule:
#   1. If env DGX_HOST is set and reachable, use it (manual override).
#   2. Probe each registered host (ssh + nvidia-smi + loadavg, 5 s timeout).
#   3. Skip hosts that fail to respond.
#   4. Score = (gpu_util_pct * 1.0) + (loadavg_1min * 10.0).
#      Lower score = freer. Tiebreak: alias alphabetical.
#
# All probe + logging output goes to stderr. Only the chosen alias goes to stdout.

set -euo pipefail
REG="$(dirname "$(readlink -f "$0")")/registry.json"

if [ ! -f "$REG" ]; then
  echo "[select-dgx] registry not found: $REG" >&2
  exit 2
fi

probe_one() {
  # Args: alias ssh_user ssh_key
  # Prints: alias\tload\tgpu_util\tstatus  (status = ok / unreachable)
  local alias=$1 user=$2 key=$3
  local probe
  probe=$(ssh -n -i "$key" -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    "$alias" 'L=$(cut -d" " -f1 /proc/loadavg); \
              U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?"); \
              echo "$L $U"' 2>/dev/null) || { printf "%s\t-\t-\tunreachable\n" "$alias"; return; }
  local load util
  read load util <<< "$probe"
  printf "%s\t%s\t%s\tok\n" "$alias" "$load" "$util"
}

read_registry() {
  python3 - "$REG" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for h in d.get("hosts", []):
    print(f"{h['alias']}\t{h.get('ssh_user','')}\t{h.get('ssh_key','')}")
PY
}

cmd=${1:-pick}

case "$cmd" in
  --list)
    printf "%-16s %-6s %-6s %s\n" "HOST" "LOAD" "GPU%" "STATUS" >&2
    while IFS=$'\t' read -r alias user key; do
      r=$(probe_one "$alias" "$user" "$key")
      IFS=$'\t' read -r a l g s <<< "$r"
      printf "%-16s %-6s %-6s %s\n" "$a" "$l" "$g" "$s" >&2
    done < <(read_registry)
    ;;

  --probe)
    target=${2:?"--probe needs HOST"}
    while IFS=$'\t' read -r alias user key; do
      if [ "$alias" = "$target" ]; then
        probe_one "$alias" "$user" "$key"
        exit 0
      fi
    done < <(read_registry)
    echo "[select-dgx] unknown host: $target" >&2; exit 2
    ;;

  --json)
    chosen=$("$0")
    python3 - "$REG" "$chosen" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for h in d["hosts"]:
    if h["alias"] == sys.argv[2]:
        print(json.dumps(h, indent=2)); break
PY
    ;;

  pick|*)
    # 1. honor DGX_HOST override
    if [ -n "${DGX_HOST:-}" ]; then
      while IFS=$'\t' read -r alias user key; do
        if [ "$alias" = "$DGX_HOST" ]; then
          r=$(probe_one "$alias" "$user" "$key")
          IFS=$'\t' read -r a l g s <<< "$r"
          if [ "$s" = "ok" ]; then
            echo "[select-dgx] using DGX_HOST=$alias (load=$l gpu=$g%)" >&2
            echo "$alias"; exit 0
          else
            echo "[select-dgx] DGX_HOST=$alias unreachable; falling back to auto-pick" >&2
          fi
        fi
      done < <(read_registry)
    fi

    # 2. probe + score
    best=""; best_score="999999"
    while IFS=$'\t' read -r alias user key; do
      r=$(probe_one "$alias" "$user" "$key")
      IFS=$'\t' read -r a l g s <<< "$r"
      if [ "$s" != "ok" ]; then
        echo "[select-dgx] skip $a (unreachable)" >&2
        continue
      fi
      # gpu util may be "?" — treat as 0
      [ "$g" = "?" ] && g=0
      score=$(python3 -c "print(float('$g')*1.0 + float('$l')*10.0)")
      echo "[select-dgx] $a  load=$l  gpu=${g}%  score=$score" >&2
      if python3 -c "import sys; sys.exit(0 if float('$score') < float('$best_score') else 1)"; then
        best=$a; best_score=$score
      fi
    done < <(read_registry)

    if [ -z "$best" ]; then
      echo "[select-dgx] no host reachable" >&2; exit 3
    fi
    echo "[select-dgx] chose: $best" >&2
    echo "$best"
    ;;
esac
