#!/usr/bin/env python
"""General, per-space-parameterized version of the topdown-segmentation
table-box refinement technique (see README_table_geometry_refinement.md for
the full history of why this approach works: a naive top-down render picks
up the ceiling everywhere; restricting to a horizontal band well above the
floor and below the ceiling isolates real tabletop-height content into
clean, separated, correctly-colored blobs).

Everything that was hardcoded for shinhan_space in the original
refit_box_topdown_segmentation.py is now either a CLI argument or derived
from the data itself:
  - floor height / room footprint  -> pipeline9/geo_bounds.py (auto-derived,
    overridable, same convention as yaw-deg/scale-to-meters)
  - single-object pixel-size thresholds -> derived from the MEDIAN footprint
    area of the base detector's own boxes for --label, converted to pixels
    via this room's own scale/resolution, instead of fixed pixel counts
    tuned to one room's size

Verified to reproduce shinhan_space's previously hand-tuned result (same
set of matched/unmatched candidates) when run with shinhan's own already-
established yaw-deg/scale-to-meters and no bound overrides.

Usage:
  python pipeline9/refit_topdown_general.py \
    --ply data/shinhan_hires_30k_derotated.ply \
    --boxes pipeline9/out/_tmp_reverted_boxes.json \
    --yaw-deg 28.072 --scale-to-meters 4.94 \
    --label table \
    --out pipeline9/out/_tmp_topdownseg_boxes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo_bounds import auto_floor_z, auto_room_bounds  # noqa: E402
from refit_box_extent_from_mesh import yaw_matrix  # noqa: E402

SH_C0 = 0.28209479177387814  # SH DC-term -> RGB, see splat_to_pointcloud.py


def load_band_mask_and_image(ply_path, xmin, xmax, ymin, ymax, res, floor_z,
                              band_lo_m, band_hi_m, scale, opacity_thresh):
    p = PlyData.read(str(ply_path))["vertex"]
    xyz = np.column_stack([p["x"], p["y"], p["z"]]).astype(np.float64)
    opacity = 1 / (1 + np.exp(-np.array(p["opacity"], dtype=np.float64)))
    f_dc = np.column_stack([p["f_dc_0"], p["f_dc_1"], p["f_dc_2"]]).astype(np.float64)
    rgb = np.clip(0.5 + SH_C0 * f_dc, 0, 1)

    band_lo = floor_z + band_lo_m / scale
    band_hi = floor_z + band_hi_m / scale
    keep = (opacity >= opacity_thresh) & (xyz[:, 2] >= band_lo) & (xyz[:, 2] <= band_hi)
    xyz_b, rgb_b = xyz[keep], rgb[keep]

    px = np.clip(((xyz_b[:, 0] - xmin) / (xmax - xmin) * (res - 1)).astype(np.int32), 0, res - 1)
    py = np.clip(((xyz_b[:, 1] - ymin) / (ymax - ymin) * (res - 1)).astype(np.int32), 0, res - 1)

    mask = np.zeros((res, res), dtype=bool)
    img = np.zeros((res, res, 3), dtype=np.float64)
    cnt = np.zeros((res, res), dtype=np.int32)
    np.add.at(mask, (py, px), True)
    np.add.at(img, (py, px), rgb_b)
    np.add.at(cnt, (py, px), 1)
    has = cnt > 0
    img[has] /= cnt[has][:, None]
    return mask, img, has


def px_to_world(px, py, xmin, xmax, ymin, ymax, res):
    wx = px / (res - 1) * (xmax - xmin) + xmin
    wy = py / (res - 1) * (ymax - ymin) + ymin
    return wx, wy


def valley_split(ys, xs, n_parts):
    y0, y1 = ys.min(), ys.max()
    hist = np.bincount(ys - y0, minlength=y1 - y0 + 1).astype(np.float64)
    hist_smooth = ndimage.uniform_filter1d(hist, size=5)
    target_len = len(hist) / n_parts
    splits = [0]
    for k in range(1, n_parts):
        center = int(k * target_len)
        window = 15
        lo, hi = max(0, center - window), min(len(hist_smooth), center + window)
        local_min = lo + np.argmin(hist_smooth[lo:hi])
        splits.append(local_min)
    splits.append(len(hist))
    parts = []
    for i in range(n_parts):
        s0, s1 = splits[i] + y0, splits[i + 1] + y0
        m = (ys >= s0) & (ys < s1)
        if m.sum() >= 30:
            parts.append((ys[m], xs[m]))
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ply", required=True, help="already-derotated splat ply")
    ap.add_argument("--boxes", required=True, help="base detections, {label,center,size,angle,id} schema")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="table", help="which detection label to refine "
                     "(assumes a flat horizontal surface visible from directly above)")
    ap.add_argument("--yaw-deg", type=float, required=True)
    ap.add_argument("--scale-to-meters", type=float, required=True)
    ap.add_argument("--res", type=int, default=600)
    ap.add_argument("--floor-z", type=float, default=None, help="default: auto-derived")
    ap.add_argument("--xmin", type=float, default=None)
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--band-lo-m", type=float, default=0.35, help="height band above floor")
    ap.add_argument("--band-hi-m", type=float, default=0.95)
    ap.add_argument("--opacity-thresh", type=float, default=0.3)
    ap.add_argument("--max-aspect", type=float, default=4.0,
                     help="reject blobs more elongated than this (wall strips/pillars)")
    ap.add_argument("--min-desk-frac", type=float, default=0.65,
                     help="lower pixel-area cutoff as a fraction of the derived single-object size")
    ap.add_argument("--max-desk-frac", type=float, default=1.45,
                     help="upper pixel-area cutoff (above this, a blob is treated as several merged objects)")
    ap.add_argument("--single-px", type=float, default=None,
                     help="override the derived single-object pixel-area directly "
                     "(default: median base-detection footprint area, converted to pixels)")
    ap.add_argument("--match-dist-max-m", type=float, default=None,
                     help="default: no cap here — outlier rejection happens in the caller "
                     "(refine_and_discover.py), which has visibility across ALL matches")
    args = ap.parse_args()

    p = PlyData.read(str(args.ply))["vertex"]
    xyz_all = np.column_stack([p["x"], p["y"], p["z"]]).astype(np.float64)
    opacity_all = 1 / (1 + np.exp(-np.array(p["opacity"], dtype=np.float64)))

    floor_z = args.floor_z if args.floor_z is not None else auto_floor_z(
        xyz_all, opacity_all, args.opacity_thresh)
    if None in (args.xmin, args.xmax, args.ymin, args.ymax):
        auto_x0, auto_x1, auto_y0, auto_y1 = auto_room_bounds(xyz_all, opacity_all, args.opacity_thresh)
    xmin = args.xmin if args.xmin is not None else auto_x0
    xmax = args.xmax if args.xmax is not None else auto_x1
    ymin = args.ymin if args.ymin is not None else auto_y0
    ymax = args.ymax if args.ymax is not None else auto_y1
    print(f"[topdown-general] floor_z={floor_z:.4f}  bounds=({xmin:.3f},{xmax:.3f},{ymin:.3f},{ymax:.3f})")

    data = json.loads(Path(args.boxes).read_text())
    objs = [b for b in data["boxes"] if b["label"] == args.label]
    if not objs:
        raise SystemExit(f"no boxes with label={args.label!r} in {args.boxes}")

    # single-object pixel-size expectation, derived from THIS run's own base
    # detections instead of a fixed pixel count tuned to one room
    median_area_native = float(np.median([b["size"][0] * b["size"][1] for b in objs]))
    px_per_x = (args.res - 1) / (xmax - xmin)
    px_per_y = (args.res - 1) / (ymax - ymin)
    single_px = args.single_px if args.single_px is not None else median_area_native * px_per_x * px_per_y
    min_px, max_px = args.min_desk_frac * single_px, args.max_desk_frac * single_px
    print(f"[topdown-general] median base {args.label} area={median_area_native:.5f} native units^2 "
          f"-> single_px~{single_px:.0f} (accept range {min_px:.0f}-{max_px:.0f})")

    mask, img, has_color = load_band_mask_and_image(
        args.ply, xmin, xmax, ymin, ymax, args.res, floor_z,
        args.band_lo_m, args.band_hi_m, args.scale_to_meters, args.opacity_thresh)

    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    print(f"[topdown-general] {n} raw components")

    candidates = []
    for lbl in range(1, n + 1):
        sz = sizes[lbl - 1]
        if sz < min_px:
            continue
        ys, xs = np.where(labeled == lbl)
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        if sz > max_px:
            n_parts = int(round(sz / single_px))
            if n_parts < 2:
                continue
            if h >= w:
                for part_ys, part_xs in valley_split(ys, xs, n_parts):
                    candidates.append((part_ys, part_xs))
            # only Y-elongated merges are handled - matches the shinhan
            # investigation's confirmed shape of merges; an X-elongated
            # oversized blob is left unsplit and will fail the aspect filter
            continue
        candidates.append((ys, xs))
    print(f"[topdown-general] {len(candidates)} candidates after size/split pass")

    kept = []
    for ys, xs in candidates:
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        aspect = max(h, w) / max(min(h, w), 1)
        area = len(ys)
        if area < min_px * 0.5 or aspect > args.max_aspect:
            continue
        wx0, wy0 = px_to_world(xs.min(), ys.min(), xmin, xmax, ymin, ymax, args.res)
        wx1, wy1 = px_to_world(xs.max(), ys.max(), xmin, xmax, ymin, ymax, args.res)
        mean_color = img[ys, xs][has_color[ys, xs]].mean(axis=0) if has_color[ys, xs].any() else None
        kept.append({
            "center": ((wx0 + wx1) / 2, (wy0 + wy1) / 2, None),
            "size_xy": (wx1 - wx0, wy1 - wy0),
            "n_px": area,
            "mean_color": None if mean_color is None else mean_color.tolist(),
        })
    print(f"[topdown-general] {len(kept)} plausible single-object blobs after aspect filter")

    R_inv = yaw_matrix(args.yaw_deg).T
    R_fwd = yaw_matrix(args.yaw_deg)
    obj_centers_derot = np.array([R_inv @ np.array(b["center"]) for b in objs])

    n_matched = 0
    used_idx = set()
    matches = []  # (base_id, match_dist_m, new_size_native) - reported for the caller's outlier pass
    for cand in kept:
        cx, cy, _ = cand["center"]
        dists = np.linalg.norm(obj_centers_derot[:, :2] - np.array([cx, cy]), axis=1)
        j = int(np.argmin(dists))
        if dists[j] > 0.25 or j in used_idx:
            continue
        used_idx.add(j)
        b = objs[j]
        old_z_half = b["size"][2] / 2
        new_center_derot = np.array([cx, cy, obj_centers_derot[j, 2]])
        new_center_orig = R_fwd @ new_center_derot
        new_size = [cand["size_xy"][0], cand["size_xy"][1], old_z_half * 2]
        dist_m = float(dists[j] * args.scale_to_meters)
        matches.append({"id": b["id"], "match_dist_m": dist_m,
                         "old_size": list(b["size"]), "new_size": new_size})
        print(f"  {args.label} id={b['id']:>3}: matched (dist={dists[j]:.3f}, {dist_m:.3f}m) "
              f"size {[round(v,3) for v in b['size']]} -> {[round(v,3) for v in new_size]}")
        b["center"] = [float(v) for v in new_center_orig]
        b["size"] = [float(v) for v in new_size]
        n_matched += 1

    unmatched_candidates = []
    for cand in kept:
        cx, cy, _ = cand["center"]
        dists = np.linalg.norm(obj_centers_derot[:, :2] - np.array([cx, cy]), axis=1)
        if dists.min() > 0.25:
            unmatched_candidates.append(cand)

    print(f"[topdown-general] matched/updated {n_matched} of {len(objs)} existing {args.label}s, "
          f"{len(unmatched_candidates)} candidate blobs have no existing detection within the "
          f"(loose, center-distance) match radius")

    # NOTE for callers doing new-object discovery: use `kept` (ALL plausible
    # single-object blobs), not `unmatched_candidates` above - the match
    # radius here is deliberately loose (0.25 native units, matches this
    # object to ITS OWN prior detection for refinement purposes) and can
    # therefore mask a genuinely separate real object that merely happens to
    # sit within that radius of some other object's original position (this
    # happened for shinhan_space's table 2 - see README). True coverage
    # needs a footprint-OVERLAP test against the FINAL, post-outlier-revert
    # boxes, not a center-distance test against original pre-refinement ones.
    out = {
        "data": data,
        "matches": matches,
        "kept_candidates": kept,
        "unmatched_candidates": unmatched_candidates,
        "floor_z": floor_z, "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
        "median_confirmed_area_native": median_area_native,
        "px_per_x": px_per_x, "px_per_y": px_per_y, "res": args.res,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
