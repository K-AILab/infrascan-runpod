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

# Cosmetic/derived stages: a failure here must NOT fail the whole scan (the core
# reconstruction + views + search index are already done). We log it loudly and
# keep going so the scan still reaches "ready" and training still dispatches.
OPTIONAL_STAGES = {"gen_topdown", "downsample_ply"}


def run_stage(stage: str, slug: str) -> tuple[int, str]:
    """Invoke pipeline/<stage>.py for one space. Returns (exit_code, last_stderr_chunk)."""
    script = Path(__file__).parent / f"{stage}.py"
    if not script.exists():
        return 127, f"missing script: {script}"
    # downsample_ply.py takes the space name as a POSITIONAL arg (it also accepts
    # a raw PLY path), unlike every other stage which uses --space. Passing
    # --space to it makes argparse abort with rc=2, so special-case it.
    if stage == "downsample_ply":
        cmd = [sys.executable, str(script), slug]
    else:
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
            # surface the REAL error, not just the hint, so failures are diagnosable
            if err.strip():
                print(f"[runner] --- {stage} stderr (tail) ---\n{err}\n[runner] --- end stderr ---")
            if stage in OPTIONAL_STAGES:
                print(f"[runner] {stage} is optional — continuing without it.")
                continue
            sys.exit(1)
        record_stage_ok(args.slug, stage)

    update_status(args.slug, "ready")
    print(f"[runner] done — {args.slug}")


if __name__ == "__main__":
    main()
