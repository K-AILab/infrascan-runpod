#!/usr/bin/env python
"""Single, canonical, correct conversion of a boxes.json entry (native splat
frame, still-yawed) to its axis-aligned bounding footprint in GEO frame
(world_x=splat_x, world_y=splat_z(up), world_z=splat_y - see
pipeline9-coordinate-frame-formula memory).

CRITICAL bug this fixes, found live: computing only the box's two OPPOSITE
derotated corners, rotating THOSE two back to native frame, and taking their
min/max is NOT the rotated rectangle's bounding box - a rotated rectangle's
other two corners can stick out further in x or z than either of the two you
happened to pick. Confirmed directly on a real box: this undercounted one
dimension by very nearly 2x (0.73m vs the correct 1.33m). This exact bug
existed independently in at least two scripts (render_pc_topdown.py's
visualization AND discover_tables_from_pointcloud.py's existing-coverage
check) - hence one shared, tested function instead of a third copy.
"""
from __future__ import annotations

import numpy as np

from refit_box_extent_from_mesh import yaw_matrix


def box_footprint_geo(b: dict, yaw_deg: float, scale_to_meters: float):
    """Returns (x0, z0, x1, z1) - the box's axis-aligned XZ bounding
    footprint in the STILL-YAWED geo frame (matching xyz_splat[:,(0,2,1)]*
    scale with no further rotation - the same frame the raw splat/gaussian
    renders use). Uses ALL FOUR footprint corners (not two)."""
    R_inv = yaw_matrix(yaw_deg).T
    R_fwd = yaw_matrix(yaw_deg)
    cd = R_inv @ np.array(b["center"])
    sx, sy = b["size"][0], b["size"][1]
    corners_derot = np.array([
        [cd[0] - sx / 2, cd[1] - sy / 2, cd[2]],
        [cd[0] + sx / 2, cd[1] - sy / 2, cd[2]],
        [cd[0] + sx / 2, cd[1] + sy / 2, cd[2]],
        [cd[0] - sx / 2, cd[1] + sy / 2, cd[2]],
    ])
    corners_native = (R_fwd @ corners_derot.T).T
    corners_geo = corners_native[:, [0, 2, 1]] * scale_to_meters
    return (float(corners_geo[:, 0].min()), float(corners_geo[:, 2].min()),
            float(corners_geo[:, 0].max()), float(corners_geo[:, 2].max()))


def box_footprint_derotated_geo(b: dict, yaw_deg: float, scale_to_meters: float):
    """Returns (x0, z0, x1, z1) in the DEROTATED geo frame - walls axis-
    aligned to the image's own x/z axes ("along the world monitor axis"),
    matching the frame every OTHER topdown script in this pipeline already
    works in (topdown_floor_contrast.py, discover_and_refine_boxes.py, etc.)
    Simpler than box_footprint_geo: derotating in native frame (rotating
    native's x,y plane) commutes with the (0,2,1) axis permutation into geo
    frame (native's rotated x,y plane becomes geo's x,z plane) - so there is
    no need to rotate back to native afterward. The box is genuinely axis-
    aligned in this frame, so all 4 corners agree with just 2 - no rotation
    ambiguity here, unlike box_footprint_geo's still-yawed frame."""
    R_inv = yaw_matrix(yaw_deg).T
    cd = R_inv @ np.array(b["center"])
    sx, sy = b["size"][0], b["size"][1]
    x0n, y0n = cd[0] - sx / 2, cd[1] - sy / 2
    x1n, y1n = cd[0] + sx / 2, cd[1] + sy / 2
    # native (x, y_native, z_native) -> derotated-geo (x, z_native, y_native) * scale
    gx0, gz0 = x0n * scale_to_meters, y0n * scale_to_meters
    gx1, gz1 = x1n * scale_to_meters, y1n * scale_to_meters
    return (min(gx0, gx1), min(gz0, gz1), max(gx0, gx1), max(gz0, gz1))


def native_center_from_derotated_geo_point(gx: float, gy: float, gz: float,
                                            yaw_deg: float, scale_to_meters: float):
    """Inverse of box_footprint_derotated_geo's forward mapping - given a
    point (gx, gy=vertical, gz) in the derotated geo frame, return its
    native (still-yawed splat frame) [x, y, z] center, suitable for writing
    into a boxes.json "center" field (with "angle": radians(yaw_deg))."""
    x_native_derot = gx / scale_to_meters
    z_native = gy / scale_to_meters
    y_native_derot = gz / scale_to_meters
    R_fwd = yaw_matrix(yaw_deg)
    return (R_fwd @ np.array([x_native_derot, y_native_derot, z_native])).tolist()


def rotate_geo_xz(pts: np.ndarray, deg: float) -> np.ndarray:
    """Rotate an (N,3) point array's (x,z) plane by deg degrees, leaving y
    (index 1) untouched."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    out = pts.copy()
    x, z = pts[:, 0], pts[:, 2]
    out[:, 0] = c * x - s * z
    out[:, 2] = s * x + c * z
    return out


# --- validated splat<->factory_space_13-pointcloud alignment (factory13_100k_sharpen) ---
#
# Found the hard way (user caught real misalignment in a render, twice): a
# rotated rectangle's own residual tilt was checked QUANTITATIVELY via
# cv2.minAreaRect on the rasterized silhouette (never trust eyeballing a
# 1M+-point scatterplot) for several candidate angles on EACH point set
# independently:
#   splat geo frame:  -8.07deg -> 0.63-0.88deg residual (correct)
#                      +8.07deg -> ~19.5deg residual (WRONG - do not use)
#   pc native frame:  +8.07deg -> 0.22deg residual (correct)
#                      -8.07deg -> ~23.4deg residual (WRONG - do not use)
# These are OPPOSITE signs - not a bug, just two independently-reconstructed
# datasets that each happened to land at a similar-magnitude, opposite-sign
# yaw relative to true walls; there's no deeper relationship between them.
# After each is independently derotated with its OWN correct sign, only a
# translation is needed to overlay them (confirmed: an x-flip test was
# WORSE, not better - these are not mirrored, just each needed their own
# correct rotation before comparison ever made sense).
FACTORY13_YAW_SPLAT_DEG = -8.07   # rotate_geo_xz(splat_geo, this) -> axis-aligned
FACTORY13_YAW_PC_DEG = 8.07       # rotate_geo_xz(pc_native_xyz, this) -> axis-aligned
FACTORY13_OFFSET_SPLAT_DEROT_TO_PC_DEROT = np.array([-4.09580555, 0.23039551, 3.34588502])


def pc_native_to_splat_derotated_geo(pc_xyz: np.ndarray) -> np.ndarray:
    """Map factory_space_13/pointcloud.ply's native xyz into the SAME
    derotated frame box_footprint_derotated_geo's boxes live in (for
    factory13_100k_sharpen specifically - these constants are per-space,
    like yaw/scale always are in this project)."""
    pc_derot = rotate_geo_xz(pc_xyz, FACTORY13_YAW_PC_DEG)
    return pc_derot - FACTORY13_OFFSET_SPLAT_DEROT_TO_PC_DEROT
