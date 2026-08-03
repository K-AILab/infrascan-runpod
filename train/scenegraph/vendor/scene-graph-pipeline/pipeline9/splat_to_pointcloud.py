#!/usr/bin/env python
"""pipeline9: extract a plain point cloud from a trained 3D Gaussian Splat.

A Gaussian Splat's per-point MEANS are fundamentally a point cloud — every
downstream detector in this project (pipeline2b's geometric clustering,
pipeline4's 3DETR, pipeline6's Group-Free-3D, pipeline8's ensemble) only
ever needs xyz + rgb per point. None of them need the splat's rendering
machinery (opacity, anisotropic scale, rotation, spherical harmonics) at
all, which means none of them need the CUDA rasterizer that crashes on
this hardware (see pipeline9/README.md for that investigation) — this
script extracts the point cloud directly from the .ply's stored
parameters, no rendering involved, and writes it in the exact format
every existing detector already reads.

Two real, necessary pieces of interpretation beyond a straight column copy:

1. Color: the PLY stores each Gaussian's color as spherical-harmonics
   coefficients, not RGB directly. `f_dc_0/1/2` is the degree-0 (view-
   independent) term; converting it to RGB uses the standard SH0->color
   formula (color = 0.5 + SH_C0 * f_dc, SH_C0 = 0.28209479177387814) —
   the same constant every 3DGS implementation uses. Higher-order terms
   (f_rest_*, view-dependent) are ignored; a static point cloud has no
   viewing angle to evaluate them against anyway.

2. Scale + axis convention: this splat's own coordinate frame is NOT
   meters and does not use this project's established Y-up convention —
   confirmed by comparing extents directly against this same physical
   room's already-reconstructed, real-meter point cloud
   (data/shinhan_space/pointcloud.ply):

       real (meters):  X=17.42  Y=3.80  Z=16.67   (Y is the ~ceiling-height axis)
       splat (raw):     X=2.60  Y=2.52  Z=0.77   (Z is the smallest axis)

   The smallest-extent axis in each is the room's height in both cases —
   that identifies the splat's Z as its "up" axis, and the ratio of the
   two height extents (3.80 / 0.77 ~= 4.94) as the scale factor to real
   meters. Whether the two horizontal axes need swapping/mirroring beyond
   this is NOT resolved here (no correspondence points were available to
   check it precisely) — it doesn't affect detection quality (detectors
   don't care which way is "north"), only whether this reconstruction
   visually lines up with the camera-based one if compared side by side.
   SPLAT_AXIS_MAP below is the one assumption worth revisiting if that
   ever matters.

Also filters two well-known Gaussian Splatting artifacts that would
otherwise pollute the point cloud fed to detectors expecting real surface
points:
  - low-opacity "filler" Gaussians (~40% of this file) that contribute
    subtly to rendering but aren't solid geometry
  - rare, extreme-scale "floater" Gaussians (a handful out of 1.3M here)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814

# splat (x,y,z) column -> world (x,y,z) column, world[i] = SPLAT_AXIS_MAP[i]
# world Y (up) = splat Z, per the height-extent match in the module docstring.
SPLAT_AXIS_MAP = (0, 2, 1)  # world_x=splat_x, world_y=splat_z, world_z=splat_y


def load_splat_as_points(ply_path: Path, scale_to_meters: float,
                         opacity_thresh: float = 0.1,
                         max_gaussian_scale_m: float = 0.5):
    p = PlyData.read(str(ply_path))["vertex"]
    n = len(p)

    xyz_splat = np.stack([p["x"], p["y"], p["z"]], axis=1).astype(np.float64)
    xyz = xyz_splat[:, SPLAT_AXIS_MAP] * scale_to_meters

    opacity = 1.0 / (1.0 + np.exp(-p["opacity"].astype(np.float32)))
    gauss_scale = np.exp(np.stack(
        [p["scale_0"], p["scale_1"], p["scale_2"]], axis=1).astype(np.float32)
    ).max(axis=1) * scale_to_meters

    f_dc = np.stack([p["f_dc_0"], p["f_dc_1"], p["f_dc_2"]], axis=1).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)

    keep = (opacity >= opacity_thresh) & (gauss_scale <= max_gaussian_scale_m)
    print(f"[splat2pc] {n:,} gaussians -> {keep.sum():,} kept "
          f"({(~keep).sum():,} dropped: low-opacity/floater)")
    return xyz[keep], rgb[keep]


def write_pointcloud_ply(path: Path, xyz: np.ndarray, rgb01: np.ndarray):
    rgb255 = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    verts = np.empty(len(xyz), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb255[:, 0], rgb255[:, 1], rgb255[:, 2]
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(path))
    print(f"[splat2pc] wrote {path} ({len(xyz):,} points)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splat-ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale-to-meters", type=float, default=4.94,
                    help="see module docstring for how this was derived")
    ap.add_argument("--opacity-thresh", type=float, default=0.1)
    ap.add_argument("--max-gaussian-scale-m", type=float, default=0.5)
    args = ap.parse_args()

    xyz, rgb = load_splat_as_points(
        Path(args.splat_ply), args.scale_to_meters,
        args.opacity_thresh, args.max_gaussian_scale_m)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_pointcloud_ply(out_path, xyz, rgb)

    print(f"[splat2pc] world extent: {(xyz.max(0) - xyz.min(0)).round(2)} "
          f"(compare against data/shinhan_space/pointcloud.ply's [17.42, 3.80, 16.67])")


if __name__ == "__main__":
    main()
