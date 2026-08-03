"""Clean + heavily downsample a point cloud for the browser viewer.

Stages:
    1. statistical outlier removal — drops scattered noise points
    2. voxel downsample at coarse resolution — slashes file size 10-20×

Usage:
    python pipeline/clean_and_downsample.py --space <slug>
        # writes  out/<slug>/web/downsampled_web.ply  (replaces existing)

    python pipeline/clean_and_downsample.py --in path/in.ply --out path/out.ply \\
        --voxel 0.1 --nb-neighbors 20 --std-ratio 2.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import space   # noqa: E402


def clean_and_downsample(src: Path, dst: Path, voxel: float,
                          nb_neighbors: int, std_ratio: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[clean] reading {src}", flush=True)
    t = time.time()
    pcd = o3d.io.read_point_cloud(str(src))
    print(f"[clean]   {len(pcd.points):,} points · {time.time()-t:.0f}s", flush=True)

    if nb_neighbors > 0:
        print(f"[clean] stat outlier removal (nb={nb_neighbors}, std={std_ratio}) …", flush=True)
        t = time.time()
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                std_ratio=std_ratio)
        print(f"[clean]   {len(pcd.points):,} kept · {time.time()-t:.0f}s", flush=True)

    print(f"[clean] voxel downsample (voxel={voxel} m) …", flush=True)
    t = time.time()
    pcd = pcd.voxel_down_sample(voxel)
    print(f"[clean]   {len(pcd.points):,} points · {time.time()-t:.0f}s", flush=True)

    print(f"[clean] writing {dst}", flush=True)
    o3d.io.write_point_cloud(str(dst), pcd, write_ascii=False, compressed=True)
    sz = dst.stat().st_size / 1024 / 1024
    print(f"[clean] done · {sz:.1f} MB on disk", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", help="slug; resolves data/<slug>/pointcloud.ply → out/<slug>/web/downsampled_web.ply")
    ap.add_argument("--in",  dest="src", help="input .ply (overrides --space)")
    ap.add_argument("--out", dest="dst", help="output .ply (overrides --space)")
    ap.add_argument("--voxel",        type=float, default=0.08, help="voxel size in meters (default 0.08)")
    ap.add_argument("--nb-neighbors", type=int,   default=20,   help="stat outlier neighbours (0 = skip)")
    ap.add_argument("--std-ratio",    type=float, default=2.0,  help="stat outlier std-dev ratio")
    args = ap.parse_args()

    if args.src and args.dst:
        clean_and_downsample(Path(args.src), Path(args.dst),
                             args.voxel, args.nb_neighbors, args.std_ratio)
        return

    if not args.space:
        ap.error("--space or both --in/--out required")
    paths = space(args.space)
    src = paths["pointcloud"]
    dst = paths["out_dir"] / "web" / "downsampled_web.ply"
    clean_and_downsample(src, dst, args.voxel, args.nb_neighbors, args.std_ratio)


if __name__ == "__main__":
    main()
