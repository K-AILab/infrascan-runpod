#!/usr/bin/env python
"""pipeline9: rotate a Gaussian Splat's Gaussians around its native Z axis
(this splat's established vertical axis) by a fixed yaw, writing a new .ply.

Why: splat_analyzer always outputs axis-aligned boxes (never fits a
rotation). This room is rotated ~28 degrees relative to the splat's native
X/Y grid — confirmed three independent ways directly on this splat
(min-area-rectangle fit on floor points, on the large connected-plane
surface, and on the whole scene's footprint all gave 27-28 degrees), and
matching this project's own already-established building_yaw_deg=28.072
for this same physical room (out/geo_shinhan_space_geo/scene_graph.json).
Since every axis-aligned box shares that same ~28 degree offset relative to
the (uniformly rotated) real objects, every box looks tilted the same way
relative to what it's supposed to bound.

Fix: rotate the WHOLE scene by -yaw before running detection, so real
walls/furniture become axis-aligned in the transformed copy — an
axis-aligned box around an axis-aligned object is a correct, tight fit.
Detected boxes are then rotated back by +yaw (see rotate_boxes_back below)
to place them correctly in the original frame for the viewer.

Simplification: only positions and per-Gaussian rotation quaternions are
rotated (both are geometry, and rotation-invariant reasoning fails for
them). Per-Gaussian scale is rotation-invariant (local ellipsoid axis
lengths) and is left unchanged. The higher-order spherical-harmonic
coefficients (f_rest_*) encode view-DEPENDENT color and are technically
supposed to be rotated by a real spherical-harmonic rotation matrix for a
pixel-perfect re-render — skipped here since we only need OWLv2 to
recognize objects in the rendered views, not photorealistic specular
consistency, and doing so would add real complexity for no detection
benefit. f_dc (view-independent base color) needs no change either way.
"""
from __future__ import annotations

import argparse

import numpy as np
from plyfile import PlyData, PlyElement


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """(N,4) x (4,) or (N,4) x (N,4) Hamilton product, (w,x,y,z) order."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in-ply", required=True)
    ap.add_argument("--out-ply", required=True)
    ap.add_argument("--yaw-deg", type=float, required=True,
                    help="rotate the scene by THIS angle around native Z "
                    "(use -28.072 to de-rotate this splat's ~28deg room yaw)")
    args = ap.parse_args()

    ply = PlyData.read(args.in_ply)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float64)

    t = np.radians(args.yaw_deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    xyz_rot = xyz @ R.T

    q_yaw = np.array([np.cos(t / 2), 0.0, 0.0, np.sin(t / 2)])
    q_rot = quat_multiply(np.broadcast_to(q_yaw, q.shape), q)
    q_rot /= np.maximum(np.linalg.norm(q_rot, axis=1, keepdims=True), 1e-9)

    data = np.array(v.data)  # structured array, copy
    data["x"], data["y"], data["z"] = xyz_rot[:, 0], xyz_rot[:, 1], xyz_rot[:, 2]
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = \
        q_rot[:, 0], q_rot[:, 1], q_rot[:, 2], q_rot[:, 3]

    out_el = PlyElement.describe(data, "vertex")
    PlyData([out_el], text=False).write(args.out_ply)
    print(f"[derotate] rotated {len(xyz):,} gaussians by {args.yaw_deg}deg around Z -> {args.out_ply}")


if __name__ == "__main__":
    main()
