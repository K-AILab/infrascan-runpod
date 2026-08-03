#!/usr/bin/env python3
"""Downsample a large PLY pointcloud to a web-friendly size.

Usage:
    # Registered space — picks paths up from spaces.json:
    python pipeline/downsample_ply.py <space_name>
    python pipeline/downsample_ply.py icc1 --voxel 0.05

    # Or operate on arbitrary files:
    python pipeline/downsample_ply.py /abs/in.ply /abs/out.ply --voxel 0.03

Defaults (registered-space mode):
    source : data/<space>/pointcloud.ply
    target : ui/_spaces/<space>/Data_/downsampled_web.ply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent))
from _paths import space, space_choices  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source",
                   help="Registered space name OR an input PLY file path.")
    p.add_argument("dest", nargs="?", default=None,
                   help="Output PLY path (only used when source is a file path).")
    p.add_argument("--voxel", type=float, default=0.03,
                   help="Voxel grid size (m) — bigger = smaller output PLY (default 0.03)")
    args = p.parse_args()

    if args.source in space_choices():
        sp = space(args.source)
        in_path  = sp["pointcloud"]
        out_path = REPO / "ui" / "_spaces" / args.source / "Data_" / "downsampled_web.ply"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        in_path  = Path(args.source)
        out_path = Path(args.dest) if args.dest else in_path.with_name("downsampled_web.ply")

    if not in_path.exists():
        sys.exit(f"[downsample] input PLY missing: {in_path}")

    print(f"[downsample] {in_path} ({in_path.stat().st_size / 1e9:.2f} GB)  →  {out_path}")
    print(f"[downsample] voxel = {args.voxel} m")

    pcd = o3d.io.read_point_cloud(str(in_path))
    print(f"[downsample]   input points  : {len(pcd.points):,}")
    down = pcd.voxel_down_sample(args.voxel)
    print(f"[downsample]   output points : {len(down.points):,}")

    # If the dest is a symlink, replace it with the real file
    if out_path.is_symlink():
        out_path.unlink()
    o3d.io.write_point_cloud(str(out_path), down, write_ascii=False)
    print(f"[downsample] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
