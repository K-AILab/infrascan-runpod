#!/usr/bin/env python
"""Shared helpers for aligning a gaussian splat's own detection frame onto
an independently-captured data/<space>/pointcloud.ply, used by both
align_splat_to_pointcloud.py (finds the transform) and
apply_scenegraph_to_pointcloud.py (applies it to a scene graph). Kept in one
place so the two scripts can't silently drift onto different conventions.
"""
from __future__ import annotations

import cv2
import numpy as np
from plyfile import PlyData


def rot_y(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def transform_points(pts, T):
    R, t = T[:3, :3], T[:3, 3]
    return pts @ R.T + t


def measure_wall_yaw_deg(ply_path, res=0.05):
    """A room's real in-plane tilt, measured directly from its own wall
    boundary (min-area rectangle over the topdown occupancy silhouette) -
    NOT derived from the splat<->pointcloud rotation matrix. That was tried
    first (on factory13): decomposing the rigid transform's rotation into a
    Y-yaw gave ~-179.6 deg (near-180, which leaves an axis-aligned box's
    edges unchanged - rectangles are 180-deg-symmetric), and composing it
    with the axis-flip that undoes the reflection gave ~-0.4 deg instead -
    both confirmed WRONG against this direct measurement (~-8.5 deg) and
    against a visual check (box edges only actually line up with the real,
    independently-measured wall boundary at the value this function
    returns). Box ORIENTATION and box POSITION are handled by two different
    mechanisms for exactly this reason - see apply_scenegraph_to_pointcloud
    .py's module docstring.
    """
    p = PlyData.read(str(ply_path))["vertex"]
    xyz = np.column_stack([p["x"], p["y"], p["z"]]).astype(np.float64)
    xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
    zmin, zmax = xyz[:, 2].min(), xyz[:, 2].max()
    W, H = int((xmax - xmin) / res) + 1, int((zmax - zmin) / res) + 1
    gx = np.clip(((xyz[:, 0] - xmin) / res).astype(int), 0, W - 1)
    gz = np.clip(((xyz[:, 2] - zmin) / res).astype(int), 0, H - 1)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[gz, gx] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(c))
    angles = []
    for i in range(4):
        v = box[(i + 1) % 4] - box[i]
        angles.append(np.degrees(np.arctan2(v[1], v[0])))
    return float(min(angles, key=abs))
