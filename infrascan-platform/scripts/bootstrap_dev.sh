#!/usr/bin/env bash
# bootstrap_dev.sh — get the platform to a testable state in one shot.
#
# By default:
#   - initialises the SQLite DB
#   - creates an admin user
#   - leaves the spaces list EMPTY so you can upload your own
#
# Optional (pass --with-legacy-icc):
#   - registers icc1 / icc2 / icc3 by symlinking the legacy on-disk
#     artefacts into the new repo's data/<slug>/ and out/<slug>/ layout.
#     Useful for demos; not part of a clean install.
#
# Re-runnable.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${PY:=/home/chan/miniconda3/envs/infrascan/bin/python}"
: "${ADMIN_EMAIL:=admin@infrascan.local}"
: "${ADMIN_NAME:=Chan Kim}"
: "${ADMIN_PASSWORD:=infrascan-admin}"

WITH_LEGACY=0
for arg in "$@"; do
  case "$arg" in
    --with-legacy-icc) WITH_LEGACY=1 ;;
    -h|--help)
      sed -n '1,18p' "$0"
      exit 0
      ;;
  esac
done

mkdir -p data out
"$PY" -m app.db init

# Admin (idempotent)
if "$PY" -c "
from app.db import init; init()
from app.auth import get_user_by_email
import sys
sys.exit(0 if get_user_by_email('$ADMIN_EMAIL') else 1)
"; then
  echo "[bootstrap] admin user already exists: $ADMIN_EMAIL"
else
  "$PY" -m scripts.create_user --email "$ADMIN_EMAIL" --name "$ADMIN_NAME" \
        --role admin --password "$ADMIN_PASSWORD"
fi

if [ "$WITH_LEGACY" -eq 1 ]; then
  LEGACY_DATA=/home/chan/Desktop/3d-tagging-project/data
  LEGACY_OUT=/home/chan/Desktop/3d-object-tagging/out
  LEGACY_UI=/home/chan/Desktop/3d-object-tagging/ui/_spaces

  register_space () {
    local slug=$1 title=$2 n_views=$3 n_sps=$4
    local src_data="$LEGACY_DATA/$slug"
    local src_out="$LEGACY_OUT/fastsam_$slug"
    local src_web_ply="$LEGACY_UI/$slug/Data_/downsampled_web.ply"
    local src_topdown="$LEGACY_UI/$slug/topdown/topdown.png"
    if [ ! -d "$src_data" ] || [ ! -d "$src_out" ]; then
      echo "[bootstrap] skipping '$slug' — legacy paths missing"; return
    fi
    ln -snf "$src_data" "data/$slug"
    mkdir -p "out/$slug/web"
    ln -snf "$src_out/proposals.jsonl"  "out/$slug/proposals.jsonl"
    ln -snf "$src_out/embeddings.npy"   "out/$slug/embeddings.npy"
    ln -snf "$src_out/index.faiss"      "out/$slug/index.faiss"
    ln -snf "$src_out/metadata.json"    "out/$slug/metadata.json"
    ln -snf "$src_out/object_ids.npy"   "out/$slug/object_ids.npy"
    [ -f "$src_web_ply" ] && ln -snf "$src_web_ply" "out/$slug/web/downsampled_web.ply" || true
    [ -f "$src_topdown" ] && ln -snf "$src_topdown" "out/$slug/web/topdown.png"         || true
    [ -f "$LEGACY_UI/$slug/topdown/bounds.json" ] \
        && ln -snf "$LEGACY_UI/$slug/topdown/bounds.json" "out/$slug/web/bounds.json" || true
    [ -d "$LEGACY_UI/$slug/Data_/panos" ] \
        && ln -snf "$LEGACY_UI/$slug/Data_/panos"         "out/$slug/web/panos"      || true
    ln -snf "$src_data/views"        "out/$slug/web/views"
    ln -snf "$src_data/frames"       "out/$slug/web/frames"
    ln -snf "$src_data/cameras.json" "out/$slug/web/cameras.json"
    "$PY" -m scripts.register_space --slug "$slug" --title "$title" \
          --owner-email "$ADMIN_EMAIL" --status ready \
          --n-views "$n_views" --n-scanpoints "$n_sps"
  }
  register_space icc1 "ICC Office Building — Scan 1" 25056 696
  register_space icc2 "ICC Office Building — Scan 2" 25308 703
  register_space icc3 "ICC Office Building — Scan 3" 21672 602

  # Legacy import skips the upload-validation stage that normally produces
  # preflight_frames + preflight_json, so the space-detail preview strip
  # would be empty. Synthesize one from the y000_pz000 views.
  "$PY" -m scripts.gen_preflight_for_legacy --slug icc1 --slug icc2 --slug icc3
fi

echo
echo "──────────────────────────────────────────────────────────────"
echo " Bootstrap done."
echo
echo "  Admin email: $ADMIN_EMAIL"
echo "  Password   : $ADMIN_PASSWORD"
echo
if [ "$WITH_LEGACY" -eq 1 ]; then
  echo "  Legacy icc1/2/3 imported."
else
  echo "  Empty spaces list — go to /upload to add your first scan."
  echo "  (Re-run with --with-legacy-icc to import the dev-box demo spaces.)"
fi
echo
echo "  Run:  bash scripts/run_dev.sh"
echo "──────────────────────────────────────────────────────────────"
