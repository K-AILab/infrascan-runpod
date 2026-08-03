"""Stitch an Insta360 .insv (dual-fisheye, two video streams) into an
8K equirectangular MP4 that the rest of the pipeline can consume.

Skipped automatically when the input is already a stitched equirect .mp4.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def stitch(insv_path: Path, out_mp4: Path) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-y",
        "-i", str(insv_path),
        "-filter_complex",
        "[0:v:0][0:v:1]hstack=inputs=2,"
        "v360=dfisheye:e:ih_fov=200:iv_fov=200:w=8192:h=4096",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    print(f"[stitch] {insv_path.name} → {out_mp4.name}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .insv (or .mp4 — short-circuits)")
    ap.add_argument("--output", required=True, help="Output equirect .mp4")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    if src.suffix.lower() in (".mp4", ".mov", ".m4v"):
        # Already stitched — just copy/symlink so the rest of the pipeline
        # has a stable filename.
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src.resolve())
        print(f"[stitch] {src.name} is already a video container; symlinked to {dst.name}")
        return

    if src.suffix.lower() not in (".insv", ".insp"):
        sys.exit(f"unsupported input format: {src.suffix}")

    stitch(src, dst)


if __name__ == "__main__":
    main()
