#!/usr/bin/env python
"""Convert a captured point cloud into the frame tri-viewer's 3D mode expects.

The viewer's Splat/Points toggle reads scene/<id>.points.ply and applies one
fixed mapping to it, splat_xyz = world_xyz[[0,2,1]] / 4.94, matching the
convention splat_to_pointcloud.py produces. An independently captured cloud is
in neither that frame nor that scale, so it has to be pre-transformed for the
viewer's own mapping to undo:

  1. invert the ICP registration from align_splat_to_pointcloud.py;
  2. back out that script's axis map, sign and scale to reach splat-native xyz;
  3. re-encode so the viewer's divide-by-4.94 reproduces it.

Use --extra-yaw-deg when the target scene's .ksplat was built from a derotated
ply rather than the original.

Usage:
  python pointcloud_to_triviewer_points.py \
    --pointcloud-ply data/factory_space_14/pointcloud.ply \
    --transform out/factory_space_14_splat_to_pc_transform.json \
    --extra-yaw-deg -13.639 \
    --out ../tri-viewer/modes/threed/scene/factory14_detail.points.ply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

# Must match tri-viewer/modes/threed/main.js's WORLD_TO_SPLAT_* constants.
VIEWER_AXIS_MAP = [0, 2, 1]
VIEWER_SCALE = 4.94


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pointcloud-ply", required=True)
    ap.add_argument("--transform", required=True,
                    help="align_splat_to_pointcloud.py output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-yaw-deg", type=float, default=0.0,
                    help="rotate the result by this yaw about the vertical axis. Use "
                         "-<room yaw> when the target scene's .ksplat was built from "
                         "the DEROTATED ply (every factory scene here) rather than "
                         "the original (shinhan)")
    ap.add_argument("--max-points", type=int, default=1_500_000,
                    help="downsample cap — this file is fetched and parsed by the "
                         "browser on every load of the 3D mode")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    T = json.loads(Path(args.transform).read_text())
    M = np.asarray(T["transform_4x4"], dtype=np.float64)
    R, t = M[:3, :3], M[:3, 3]
    sign = np.asarray(T["axis_sign"], dtype=np.float64)
    amap = list(T["axis_map"])
    S = float(T["true_scale_to_meters"])

    p = PlyData.read(args.pointcloud_ply)["vertex"]
    xyz = np.stack([p["x"], p["y"], p["z"]], axis=1).astype(np.float64)
    names = set(p.data.dtype.names)
    has_rgb = {"red", "green", "blue"} <= names
    rgb = (np.stack([p["red"], p["green"], p["blue"]], axis=1).astype(np.uint8)
           if has_rgb else None)

    if len(xyz) > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(xyz), args.max_points, replace=False))
        xyz = xyz[idx]
        if rgb is not None:
            rgb = rgb[idx]

    # 1. undo the ICP registration  ->  v (geo frame at the transform's own scale)
    v = (xyz - t) @ R          # R.T @ x  ==  x @ R  for row-vectors

    # 2. back out align_splat_to_pointcloud's encoding -> splat-native xyz
    u = v / (S * sign)
    splat = u[:, amap]

    # 2b. match the frame this scene's ksplat is actually in
    if abs(args.extra_yaw_deg) > 1e-9:
        th = np.radians(args.extra_yaw_deg)
        c, sn = np.cos(th), np.sin(th)
        splat = splat @ np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]]).T
        print(f"  applied extra yaw {args.extra_yaw_deg:+.3f}°")

    # 3. re-encode so the viewer's own world[axis_map]/4.94 reproduces `splat`
    world_out = (splat * VIEWER_SCALE)[:, VIEWER_AXIS_MAP]

    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if rgb is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(len(world_out), dtype=fields)
    arr["x"], arr["y"], arr["z"] = world_out.T.astype(np.float32)
    if rgb is not None:
        arr["red"], arr["green"], arr["blue"] = rgb.T

    out = Path(args.out)
    if out.is_symlink():
        out.unlink()           # never write through an existing symlink
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(out))

    lo, hi = np.percentile(splat, [1, 99], axis=0)
    print(f"wrote {len(arr):,} points -> {out}")
    print(f"  splat-native span after the viewer's own transform: {(hi - lo).round(3)}")
    print(f"  (compare against the splat ply's own 1-99 percentile span)")


if __name__ == "__main__":
    main()
