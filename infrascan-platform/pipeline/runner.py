"""Pipeline runner — orchestrates the 7 stages, reports failures upstream.

Sketch only. The real implementation invokes 00b → 04 + topdown + downsample,
each as a subprocess, and translates exit codes / stderr into the
user-friendly hints from app.validation.FAILURE_HINTS.

Usage:
    python -m pipeline.runner --slug my-floor
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from app import config
from app.spaces import update_status
from app.validation import FAILURE_HINTS, record_stage_failure, record_stage_ok


STAGES = [
    "00b_gen_da3",
    "01_propose",
    "02_embed",
    "02b_match_views",
    "03_backproject",
    "03b_merge_groups",
    "04_index",
    "gen_topdown",
    "downsample_ply",
]


def run_stage(stage: str, slug: str) -> tuple[int, str]:
    """Invoke pipeline/<stage>.py for one space. Returns (exit_code, last_stderr_chunk)."""
    script = Path(__file__).parent / f"{stage}.py"
    if not script.exists():
        return 127, f"missing script: {script}"
    cmd = [sys.executable, str(script), "--space", slug]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, (res.stderr or "")[-2000:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    args = ap.parse_args()

    update_status(args.slug, "processing")

    for stage in STAGES:
        print(f"[runner] === {stage} ===")
        rc, err = run_stage(stage, args.slug)
        if rc != 0:
            hint = FAILURE_HINTS.get(stage, f"The {stage} step exited with code {rc}.")
            record_stage_failure(args.slug, stage, hint)
            print(f"[runner] {stage} failed (rc={rc}): {hint}")
            sys.exit(1)
        record_stage_ok(args.slug, stage)

    update_status(args.slug, "ready")
    print(f"[runner] done — {args.slug}")


if __name__ == "__main__":
    main()
