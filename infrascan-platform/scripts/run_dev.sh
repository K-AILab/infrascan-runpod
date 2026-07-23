#!/usr/bin/env bash
# run_dev.sh — start the user-facing app for local testing.
#
# By default binds 0.0.0.0:8070 so it's reachable through a Cloudflare tunnel.
# Override PORT / HOST in env if you want.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${PY:=/home/chan/miniconda3/envs/infrascan/bin/python}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8070}"
: "${INFRASCAN_FFPROBE:=/home/chan/miniconda3/envs/infrascan/bin/ffprobe}"
: "${INFRASCAN_FFMPEG:=/home/chan/miniconda3/envs/infrascan/bin/ffmpeg}"
: "${INFRASCAN_SECRET_KEY:=dev-key-replace-in-prod}"
: "${INFRASCAN_COOKIE_DOMAIN:=}"     # blank = localhost-friendly

export INFRASCAN_FFPROBE INFRASCAN_FFMPEG INFRASCAN_SECRET_KEY INFRASCAN_COOKIE_DOMAIN

echo "[run_dev] starting uvicorn on http://${HOST}:${PORT}"
exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
