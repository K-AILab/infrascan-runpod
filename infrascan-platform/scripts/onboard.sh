#!/usr/bin/env bash
# onboard.sh — zero to working infrascan-platform dev env in one shot.
#
# Handles:
#   1. Conda env `infrascan` (creates from environment.yml if missing)
#   2. Editable pip install
#   3. Submodule bootstrap (in case `--recurse-submodules` was forgotten)
#   4. DA3 model weights (interactive prompt — needs a source you can rsync from)
#   5. Local SQLite + admin user (via bootstrap_dev.sh)
#   6. Sanity checks (DB, ffmpeg, GPU, submodule)
#   7. Prints the URLs + creds + next-step commands
#
# Re-runnable. Skips steps that are already done.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── styling ──────────────────────────────────────────────────────────────
if [ -t 1 ]; then
  B=$'\033[1m'; C=$'\033[36m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
else
  B=""; C=""; G=""; Y=""; R=""; N=""
fi
step() { echo; echo "${C}${B}== $1 ==${N}"; }
ok()   { echo "  ${G}✓${N} $1"; }
warn() { echo "  ${Y}!${N} $1"; }
fail() { echo "  ${R}✗${N} $1"; exit 1; }

# ── 0. host prereqs ──────────────────────────────────────────────────────
step "0. Host prerequisites"
command -v git      >/dev/null 2>&1 || fail "git missing"; ok "git"
command -v conda    >/dev/null 2>&1 || fail "conda missing (install miniconda/miniforge first)"; ok "conda"
command -v ffmpeg   >/dev/null 2>&1 || warn "system ffmpeg missing (conda env brings its own — OK)"
command -v nvidia-smi >/dev/null 2>&1 && ok "nvidia-smi ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))" || warn "nvidia-smi missing — DA3 needs a GPU!"

# ── 1. conda env ─────────────────────────────────────────────────────────
step "1. Conda env 'infrascan'"
ENV_NAME=infrascan
if conda env list | awk 'NR>2 {print $1}' | grep -qx "$ENV_NAME"; then
  ok "'$ENV_NAME' already exists"
else
  if [ -f environment.yml ]; then
    warn "creating from environment.yml — this can take 5-10 min…"
    conda env create -n "$ENV_NAME" -f environment.yml
    ok "created"
  else
    fail "environment.yml missing — clone the repo again with --recurse-submodules"
  fi
fi

# Discover the env's python without needing `conda activate`
ENV_PY="$(conda run -n "$ENV_NAME" which python 2>/dev/null | tail -1)"
[ -x "$ENV_PY" ] || fail "cannot find python in env '$ENV_NAME'"
ok "python: $ENV_PY"

# ── 2. editable install ──────────────────────────────────────────────────
step "2. Editable install (pip install -e .)"
if "$ENV_PY" -c "import app.main" 2>/dev/null; then
  ok "app package already importable"
else
  "$ENV_PY" -m pip install -e . >/dev/null 2>&1 || fail "pip install failed"
  ok "installed"
fi

# ── 3. submodules ────────────────────────────────────────────────────────
step "3. Submodules (ui/legacy-viewer)"
if [ -f ui/legacy-viewer/ui/viewer/app.js ]; then
  ok "ui/legacy-viewer already checked out"
else
  git submodule update --init --recursive
  if [ -f ui/legacy-viewer/ui/viewer/app.js ]; then
    ok "submodule initialised"
  else
    fail "submodule fetch didn't produce viewer/app.js — check ssh access to K-AILab/3d-object-tagging"
  fi
fi

# ── 4. DA3 weights ───────────────────────────────────────────────────────
step "4. DA3 model weights"
W=pipeline/da3_streaming/weights
mkdir -p "$W"
if [ -f "$W/model.safetensors" ] && [ -f "$W/dino_salad.ckpt" ]; then
  ok "weights present ($(du -sh "$W" | cut -f1))"
else
  cat <<EOF

  ${Y}Weights missing.${N} They exceed GitHub's 100 MB per-file limit so we
  don't track them. You need to fetch them separately (~6.7 GB):

    $W/model.safetensors  (~6.3 GB — Depth Anything v3)
    $W/dino_salad.ckpt    (~336 MB — DINO-SALAD loop closure)
    $W/config.json        (~3 KB — DA3 config)

  Ask a teammate for the canonical location. If it's on a DGX box:

    rsync -av --info=progress2 chan@dgx-kail:/path/to/da3_weights/ $W/

  Once they're in place, rerun this script.
EOF
  exit 2
fi

# ── 5. DB + admin user ───────────────────────────────────────────────────
step "5. Local SQLite + admin user"
if [ -f data/infrascan.db ]; then
  ok "data/infrascan.db already exists"
else
  bash scripts/bootstrap_dev.sh >/dev/null 2>&1 || fail "bootstrap_dev.sh failed"
  ok "created data/infrascan.db + admin user"
fi

# Whatever the bootstrap set as ADMIN_EMAIL/ADMIN_PASSWORD is what we print.
ADMIN_EMAIL_DEFAULT="admin@infrascan.local"
ADMIN_PASSWORD_DEFAULT="infrascan-admin"

# ── 6. sanity checks ─────────────────────────────────────────────────────
step "6. Sanity checks"

"$ENV_PY" -c "
from app.db import init, get_conn
init()
users = get_conn().execute('SELECT COUNT(*) FROM users').fetchone()[0]
spaces = get_conn().execute('SELECT COUNT(*) FROM spaces').fetchone()[0]
print(f'  users:  {users}')
print(f'  spaces: {spaces}')
" || fail "DB query failed"
ok "database talks back"

"$ENV_PY" -c "
import torch
print(f'  torch:  {torch.__version__}')
print(f'  cuda:   {torch.cuda.is_available()}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"—\"})')" 2>/dev/null || warn "torch not importable (DA3 will fail — check GPU install)"

# ── 7. done ──────────────────────────────────────────────────────────────
cat <<EOF

${G}${B}Onboarding complete!${N}

${B}Next steps${N} — open two terminals:

  ${C}# Terminal 1 — web app${N}
  bash scripts/run_dev.sh
  ${Y}# → uvicorn on http://localhost:8070${N}

  ${C}# Terminal 2 — pipeline worker${N}
  $ENV_PY -m scripts.worker
  ${Y}# → polls DB every 10 s for spaces with status='processing'${N}

Then open ${B}http://localhost:8070${N} in a browser and log in:

  email:    ${B}${ADMIN_EMAIL_DEFAULT}${N}
  password: ${B}${ADMIN_PASSWORD_DEFAULT}${N}

To process your first video:

  a) Click ${B}Upload${N} on the my-spaces page, or
  b) Register + drop a file:
       $ENV_PY -m scripts.register_space --slug my-first-scan \\
         --title 'My first scan' --owner-email $ADMIN_EMAIL_DEFAULT \\
         --status processing --n-views 0 --n-scanpoints 0
       mkdir -p data/my-first-scan/uploads
       cp ~/captures/my-scan.mp4 data/my-first-scan/uploads/input_equirect.mp4

Pipeline stages will unfold in the worker log; the space flips to
${B}ready${N} when done.

See ${C}README.md${N} for the full pipeline flow.
EOF
