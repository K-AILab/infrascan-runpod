#!/usr/bin/env python3
"""Render a top-down floor-plan PNG from a Gaussian-splat PLY, in the SPLAT's own
coordinate frame (vertical axis = +Z), so it aligns with the 3D free-fly viewer's
camera positions.

Adapted from abai-shinhan-viewer/scripts/generate_minimap.py. Addition for our
tri-viewer: also emit `scanpoints` = {scan_id: [x,y,z]} in the splat frame, so the
3D mode can spawn the camera at the panorama's current scanpoint on mode-switch.

Outputs <output>/floorplan.png and <output>/floorplan.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from plyfile import PlyData

SH_C0 = 0.28209479177387814  # SH degree-0 basis: colour = 0.5 + SH_C0 * f_dc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ply", type=Path, default=Path("/home/abai/splatfacto/export_pz000_hires/splat.ply"))
    p.add_argument("--output", type=Path, default=Path("modes/threed/minimap"))
    p.add_argument("--width", type=int, default=1000)
    p.add_argument("--height", type=int, default=1000)
    p.add_argument("--margin", type=float, default=0.05, help="world units added around the point extent")
    p.add_argument("--z-min", type=float, default=None, help="keep points with z >= this (default: auto 8th pct)")
    p.add_argument("--z-max", type=float, default=None, help="keep points with z <= this (default: auto 92nd pct)")
    p.add_argument("--opacity-min", type=float, default=0.15, help="drop gaussians with sigmoid(opacity) below this")
    p.add_argument("--max-points", type=int, default=800_000)
    p.add_argument("--cameras", type=Path, default=None, help="infrascan cameras.json (walked path source)")
    p.add_argument("--dataparser", type=Path, default=None, help="nerfstudio dataparser_transforms.json (frame map)")
    p.add_argument("--boundary-grid", type=int, default=500, help="occupancy grid resolution for the wall outline")
    p.add_argument("--wall-margin", type=float, default=0.07, help="world units to keep the camera inside the wall")
    return p.parse_args()


def outer_boundary(xy: np.ndarray, min_x, max_x, min_y, max_y, grid: int, wall_margin: float):
    """Trace the room's outer wall outline from the top-down point density."""
    import cv2
    from scipy import ndimage

    ix = np.clip(((xy[:, 0] - min_x) / (max_x - min_x) * (grid - 1)).astype(np.int64), 0, grid - 1)
    iy = np.clip(((max_y - xy[:, 1]) / (max_y - min_y) * (grid - 1)).astype(np.int64), 0, grid - 1)
    occ = np.zeros((grid, grid), dtype=np.uint8)
    occ[iy, ix] = 1

    cell = (max_x - min_x) / grid
    close_px = max(1, int(round(0.18 / cell)))
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    filled = ndimage.binary_fill_holes(occ).astype(np.uint8)
    lbl, n = ndimage.label(filled)
    if n > 1:
        biggest = 1 + int(np.argmax([(lbl == k).sum() for k in range(1, n + 1)]))
        filled = (lbl == biggest).astype(np.uint8)

    erode_px = max(1, int(round(wall_margin / cell)))
    filled = cv2.erode(filled, np.ones((erode_px, erode_px), np.uint8))

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    eps = 2.5
    c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)

    poly = []
    for px, py in c:
        wx = min_x + (px + 0.5) / grid * (max_x - min_x)
        wy = max_y - (py + 0.5) / grid * (max_y - min_y)
        poly.append([float(wx), float(wy)])
    return poly


def scanpoints_in_splat_frame(cameras_path: Path, dataparser_path: Path):
    """Return (ids, world_xyz) — scanpoint centres in the splat's world frame.

    ids: list[int] sorted scan ids (pz000 / normal height only).
    world_xyz: (N,3) float array, aligned with ids.
    """
    import re
    cams = json.loads(cameras_path.read_text(encoding="utf-8"))
    rx = re.compile(r"(\d+)_pz(\d+)_y(\d+)_normal")
    groups: dict[int, list] = {}
    for c in cams:
        m = rx.search(c.get("pano", "") or "")
        if not m or int(m.group(2)) != 0:   # pz000 (normal height) only
            continue
        groups.setdefault(int(m.group(1)), []).append(c["pos"])
    ids = sorted(groups)
    centres = np.array([np.median(np.asarray(groups[i], dtype=np.float64), axis=0) for i in ids])

    dp = json.loads(dataparser_path.read_text(encoding="utf-8"))
    T = np.asarray(dp["transform"], dtype=np.float64)      # (3,4)
    s = float(dp["scale"])
    hom = np.hstack([centres, np.ones((len(centres), 1))])
    world = (hom @ T.T) * s                                # -> splat frame
    return ids, world


def aspect_adjusted_bounds(minimum, maximum, width, height):
    centre = (minimum + maximum) * 0.5
    size = np.maximum(maximum - minimum, 1e-3)
    target = width / height
    current = float(size[0] / size[1])
    if current < target:
        size[0] = size[1] * target
    else:
        size[1] = size[0] / target
    return (
        float(centre[0] - size[0] * 0.5), float(centre[0] + size[0] * 0.5),
        float(centre[1] - size[1] * 0.5), float(centre[1] + size[1] * 0.5),
    )


def main() -> None:
    a = parse_args()
    if not a.ply.exists():
        raise SystemExit(f"Missing {a.ply}")

    ply = PlyData.read(str(a.ply))
    v = ply["vertex"]
    names = set(v.data.dtype.names or [])
    xyz = np.column_stack((v["x"], v["y"], v["z"])).astype(np.float64)

    keep = np.isfinite(xyz).all(axis=1)
    if "opacity" in names:
        op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
        keep &= op >= a.opacity_min

    z = xyz[:, 2]
    z_floor = float(np.percentile(z[keep], 1))
    z_ceil = float(np.percentile(z[keep], 99))
    z_min = a.z_min if a.z_min is not None else float(np.percentile(z[keep], 8))
    z_max = a.z_max if a.z_max is not None else float(np.percentile(z[keep], 92))
    keep &= (z >= z_min) & (z <= z_max)

    xyz = xyz[keep]
    if len(xyz) == 0:
        raise SystemExit("No points survived the opacity / z-slice filter")

    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        dc = np.column_stack((v["f_dc_0"], v["f_dc_1"], v["f_dc_2"])).astype(np.float64)[keep]
        rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0) * 255.0
    elif {"red", "green", "blue"}.issubset(names):
        rgb = np.column_stack((v["red"], v["green"], v["blue"])).astype(np.float64)[keep]
    else:
        rgb = np.full((len(xyz), 3), 180.0, dtype=np.float64)

    xy = xyz[:, :2]
    if len(xy) > a.max_points:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(xy), a.max_points, replace=False)
        xy, rgb = xy[sel], rgb[sel]

    minimum = xy.min(axis=0) - a.margin
    maximum = xy.max(axis=0) + a.margin
    min_x, max_x, min_y, max_y = aspect_adjusted_bounds(minimum, maximum, a.width, a.height)

    W, H = a.width, a.height
    px = np.clip(((xy[:, 0] - min_x) / (max_x - min_x) * (W - 1)).astype(np.int64), 0, W - 1)
    py = np.clip(((max_y - xy[:, 1]) / (max_y - min_y) * (H - 1)).astype(np.int64), 0, H - 1)
    flat = py * W + px

    counts = np.bincount(flat, minlength=W * H).astype(np.float64)
    sums = np.zeros((W * H, 3), dtype=np.float64)
    for c in range(3):
        sums[:, c] = np.bincount(flat, weights=rgb[:, c], minlength=W * H)

    occ = counts > 0
    avg = np.zeros_like(sums)
    avg[occ] = sums[occ] / counts[occ, None]

    density = np.log1p(counts)
    pos = density[occ]
    scale = float(np.percentile(pos, 97)) if len(pos) else 1.0
    density = np.clip(density / max(scale, 1e-6), 0.0, 1.0)

    bg = np.asarray([14.0, 20.0, 28.0])
    tint = np.asarray([70.0, 95.0, 115.0])
    toned = np.clip(avg * 0.88 + tint[None, :] * 0.12, 0, 255)
    toned = np.clip(toned * 1.25, 0, 255)
    alpha = np.where(occ, 0.35 + 0.65 * np.sqrt(density), 0.0)
    img = bg[None, :] * (1.0 - alpha[:, None]) + toned * alpha[:, None]
    img = np.clip(img.reshape(H, W, 3), 0, 255).astype(np.uint8)

    pim = Image.fromarray(img, "RGB").filter(ImageFilter.GaussianBlur(radius=0.5))
    draw = ImageDraw.Draw(pim, "RGBA")
    for x in range(0, W, 100):
        draw.line([(x, 0), (x, H)], fill=(130, 165, 190, 18), width=1)
    for y in range(0, H, 100):
        draw.line([(0, y), (W, y)], fill=(130, 165, 190, 18), width=1)

    ids, path_world = (None, None)
    if a.cameras and a.dataparser:
        ids, path_world = scanpoints_in_splat_frame(a.cameras, a.dataparser)
    path = path_world[:, :2].tolist() if path_world is not None else None

    boundary = outer_boundary(xy, min_x, max_x, min_y, max_y, a.boundary_grid, a.wall_margin)

    if path is not None and len(path):
        pa = np.asarray(path)
        spawn_xy = [float(np.median(pa[:, 0])), float(np.median(pa[:, 1]))]
        spawn_z = 0.5 * (z_floor + z_ceil)
    elif boundary:
        bp = np.asarray(boundary)
        spawn_xy = [float(bp[:, 0].mean()), float(bp[:, 1].mean())]
        spawn_z = 0.5 * (z_floor + z_ceil)
    else:
        spawn_xy = [0.5 * (min_x + max_x), 0.5 * (min_y + max_y)]
        spawn_z = 0.5 * (z_floor + z_ceil)
    cx = 0.5 * (min_x + max_x); cy = 0.5 * (min_y + max_y)
    dx, dy = cx - spawn_xy[0], cy - spawn_xy[1]
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    spawn = {"pos": [spawn_xy[0], spawn_xy[1], spawn_z],
             "look": [spawn_xy[0] + dx / n, spawn_xy[1] + dy / n, spawn_z]}

    # scanpoint -> splat-frame position table (eye-level z), for position continuity.
    scanpoints = None
    if ids is not None:
        eye_z = 0.5 * (z_floor + z_ceil)
        scanpoints = {int(i): [float(path_world[k, 0]), float(path_world[k, 1]), float(eye_z)]
                      for k, i in enumerate(ids)}

    a.output.mkdir(parents=True, exist_ok=True)
    img_path = a.output / "floorplan.png"
    meta_path = a.output / "floorplan.json"
    pim.save(img_path, optimize=True)
    meta = {
        "image": img_path.name,
        "width": W, "height": H,
        "bounds": {"minX": min_x, "maxX": max_x, "minY": min_y, "maxY": max_y},
        "slice": {"zMin": z_min, "zMax": z_max},
        "zClamp": {"min": z_floor, "max": z_ceil},
        "spawn": spawn,
        "source": str(a.ply),
    }
    if path is not None:
        meta["path"] = path
    if boundary is not None:
        meta["boundary"] = boundary
    if scanpoints is not None:
        meta["scanpoints"] = scanpoints

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"points rasterised: {len(xy):,}")
    print(f"z-slice: [{z_min:.3f}, {z_max:.3f}]  z-clamp: [{z_floor:.3f}, {z_ceil:.3f}]")
    print(f"bounds x[{min_x:.3f},{max_x:.3f}] y[{min_y:.3f},{max_y:.3f}]")
    print(f"boundary: {len(boundary) if boundary else 0} vertices   spawn: {[round(x,2) for x in spawn['pos']]}")
    if path is not None:
        pa = np.asarray(path)
        print(f"scan path: {len(path)} scanpoints, x[{pa[:,0].min():.2f},{pa[:,0].max():.2f}] y[{pa[:,1].min():.2f},{pa[:,1].max():.2f}]")
        print(f"scanpoints table: {len(scanpoints)} entries (ids {min(scanpoints)}..{max(scanpoints)})")
    print(f"written: {img_path}\nwritten: {meta_path}")


if __name__ == "__main__":
    main()
