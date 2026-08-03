#!/usr/bin/env python
"""pipeline9: per-Gaussian normal estimation from a trained 3D Gaussian
Splat's own optimized shape parameters — no neighborhood/PCA estimation
needed (unlike a plain point cloud), since each Gaussian's own anisotropic
scale + rotation already encodes a local surface orientation directly.

Technique: the shortest of a Gaussian's 3 scale axes, rotated into world
space by its quaternion, approximates the local surface normal — a
Gaussian sitting on a surface is optimized to be flat (thin) along the
surface's normal direction and spread out (wide) along the surface
itself. This is an established technique (used in GaussianShader and
others), with a known limitation: the sign is ambiguous (a normal and its
negation are equally valid from shape alone) and arbitrary per-Gaussian.

Sign resolution went through two versions:
  v1 (wrong): orient every normal toward the WHOLE ROOM's single centroid
  — correct for room-level structure (floor/wall/ceiling really are
  defined relative to the whole room) but wrong for individual objects
  away from the room's center: a chair in a corner has its own local
  surfaces, not ones defined relative to a distant point in the middle of
  the room, so this silently flipped some of its normals the wrong way
  (caught directly: real table/object surfaces were being misclassified
  as floor/ceiling and stripped out before clustering ever saw them).
  v2 (current): local orientation CONSISTENCY (Open3D's tangent-plane MST,
  propagating agreement between neighboring points — same technique
  already validated in this project for point-cloud normals) gives every
  object its own locally-correct orientation, with only the overall
  GLOBAL sign anchored by height (do the lowest points' normals point up
  on average?) to keep the floor-up/ceiling-down convention correct.
"""
from __future__ import annotations

import numpy as np
from plyfile import PlyData


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """q: (N,4) in (w,x,y,z) order (3DGS convention) -> (N,3,3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load_splat_with_normals(ply_path: str):
    """Returns xyz (N,3), normals (N,3) unit vectors oriented toward the
    cloud's centroid, opacity (N,), max_scale (N,) — all in the splat's own
    native frame, no axis/scale conversion applied (caller's job)."""
    p = PlyData.read(ply_path)["vertex"]
    xyz = np.stack([p["x"], p["y"], p["z"]], axis=1).astype(np.float64)
    q = np.stack([p["rot_0"], p["rot_1"], p["rot_2"], p["rot_3"]], axis=1).astype(np.float64)
    scale = np.exp(np.stack([p["scale_0"], p["scale_1"], p["scale_2"]], axis=1).astype(np.float64))
    opacity = 1.0 / (1.0 + np.exp(-p["opacity"].astype(np.float64)))

    R = quat_to_rotmat(q)                      # (N,3,3)
    shortest_axis = np.argmin(scale, axis=1)    # which LOCAL axis (0,1,2) is thinnest
    # world-space normal = the rotation matrix's column corresponding to
    # the shortest local axis (R @ e_i == R[:, i]).
    normals = R[np.arange(len(R)), :, shortest_axis]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)

    # Sign resolution: a single global "toward room centroid" reference
    # (the original version of this function) is wrong for individual
    # objects away from the room's center — a chair in a corner has its
    # own local surfaces, not ones defined relative to a distant point in
    # the middle of the room; using the room centroid can silently flip
    # some of its normals the wrong way. Use LOCAL orientation consistency
    # instead (propagate agreement between neighboring points via Open3D's
    # tangent-plane MST — same technique validated earlier in this project
    # for point-cloud normals), then anchor only the overall GLOBAL sign
    # using height alone: are the lowest points' normals pointing up on
    # average? This keeps the floor-up/ceiling-down convention correct
    # without imposing a room-center bias on every other object.
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.orient_normals_consistent_tangent_plane(k=15)
    normals = np.asarray(pcd.normals)

    z = xyz[:, 2]
    low_mask = z < np.percentile(z, 10)
    if low_mask.sum() > 0 and np.median(normals[low_mask, 2]) < 0:
        normals = -normals

    max_scale = scale.max(axis=1)
    return xyz, normals, opacity, max_scale


if __name__ == "__main__":
    import sys
    xyz, normals, opacity, max_scale = load_splat_with_normals(sys.argv[1])
    print(f"loaded {len(xyz):,} gaussians")
    print(f"normal norm check (should be ~1.0): {np.linalg.norm(normals, axis=1).mean():.4f}")
    # sanity: fraction with a strongly vertical normal (candidate floor/ceiling)
    up = np.abs(normals[:, 2])  # splat Z is the established "up-ish" axis
    print(f"frac with |n.z|>0.8 (near-vertical, candidate floor/ceiling): {(up > 0.8).mean():.3f}")
    print(f"frac with |n.z|<0.3 (near-horizontal, candidate wall): {(up < 0.3).mean():.3f}")
