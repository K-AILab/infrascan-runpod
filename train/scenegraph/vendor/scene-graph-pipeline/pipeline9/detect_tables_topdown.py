#!/usr/bin/env python
"""Detect tables, desks and workbenches directly from scene geometry.

A work surface is a large horizontal plane at a characteristic height, which
makes it findable without a detector — and more reliably than with one, since
a 2D detector has to recognise a surface it usually sees edge-on.

Method:
  1. Find the dominant horizontal-surface height from a z-histogram over the
     plausible range. This self-calibrates per room: an office desk and a
     factory bench sit at different heights, and the peak locates whichever
     this room has.
  2. Project the Gaussians in a thin band around that height onto the floor
     plane and take connected components. Restricting to the band is essential
     — a plain top-down projection sees only the ceiling.
  3. Filter blobs by area, aspect and rectangularity, splitting any that are
     longer than a single surface along their own long axis.
  4. Emit each blob as an oriented box: footprint from a robust angle sweep,
     vertical extent from the floor to the measured surface height.

A blob that coincides with an existing detection updates that box's geometry.
Whether it also renames it depends on the existing label: a class that is
already a work surface keeps its name, anything else is renamed to --label.

Usage:
  python detect_tables_topdown.py --ply data/scene.ply --boxes boxes.json \
    --scale-to-meters 6.8 --room-yaw-deg 28.07 --label workbench \
    --out boxes_with_surfaces.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo_bounds import auto_floor_z, auto_room_bounds  # noqa: E402
from refit_topdown_general import load_band_mask_and_image  # noqa: E402

# Classes a desk-height horizontal surface may legitimately be hiding under.
# A blob overlapping one of these gets to rename it; anything else is left alone.
RELABELLABLE = {"shelf", "cabinet", "table", "desk", "unclassified_object",
                "cardboard_box", "workbench", "bench", "counter", "workstation"}

# Classes that are already a horizontal-surface class. A blob matching one of
# these confirms its geometry but must not rename it — a workbench is already a
# work surface under a more specific name. A shelf or cabinet at desk height
# with a desk footprint is a desk, and does get renamed.
SURFACE_KEEP = {"workbench", "desk", "table", "bench", "counter", "workstation"}


def yaw_rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def dominant_surface_z(xyz, floor_z, scale, lo_m, hi_m, bin_m=0.02):
    """Height of the strongest horizontal surface in the plausible desk range.

    Work surfaces in one room are near-uniform in height, so their tops pile
    into a single sharp z-histogram bin — the same reasoning geo_bounds uses to
    locate the floor, applied to the desk band. Locating the peak beats a fixed
    window: too wide a band also collects chair seats and backs, which break
    the surface blobs apart.
    """
    z = xyz[:, 2]
    lo, hi = floor_z + lo_m / scale, floor_z + hi_m / scale
    band = z[(z >= lo) & (z <= hi)]
    if band.size < 1000:
        return None, 0
    bins = max(int((hi - lo) / (bin_m / scale)), 5)
    counts, edges = np.histogram(band, bins=bins)
    i = int(counts.argmax())
    return float((edges[i] + edges[i + 1]) / 2), int(counts[i])


def robust_yaw(w, room_yaw_rad, sweep_deg=45.0, step_deg=0.5, lo=2.0, hi=98.0):
    """Footprint yaw, by minimising a robust extent product over an angle sweep.

    Preferred over cv2.minAreaRect for two reasons. minAreaRect fits the exact
    convex hull, so a few stray blob pixels rotate the answer; scoring on
    percentile extents instead ignores them. And a rectangle's orientation is
    only defined modulo 90 degrees (with width and height swapped), so the
    sweep is centred on the room yaw to resolve that ambiguity toward the
    representative that matches the room grid.
    """
    best = (None, np.inf)
    for d in np.arange(-sweep_deg, sweep_deg, step_deg):
        th = room_yaw_rad + np.radians(d)
        c, sn = np.cos(-th), np.sin(-th)
        loc = w @ np.array([[c, -sn], [sn, c]]).T
        ext = np.percentile(loc, hi, axis=0) - np.percentile(loc, lo, axis=0)
        area = float(ext[0] * ext[1])
        if area < best[1]:
            best = (th, area)
    th = best[0]
    c, sn = np.cos(-th), np.sin(-th)
    loc = w @ np.array([[c, -sn], [sn, c]]).T
    ext = np.percentile(loc, hi, axis=0) - np.percentile(loc, lo, axis=0)
    ctr_loc = (np.percentile(loc, hi, axis=0) + np.percentile(loc, lo, axis=0)) / 2.0
    cb, sb = np.cos(th), np.sin(th)
    ctr = np.array([[cb, -sb], [sb, cb]]) @ ctr_loc
    return float(ctr[0]), float(ctr[1]), float(ext[0]), float(ext[1]), float(th)


def split_long_blob(w, k):
    """Split a blob's world points into k parts along its own long axis.

    A row of desks runs in whatever direction the room does, so the cut has to
    follow the blob's own long axis rather than an image axis. Cuts are placed
    at the deepest minima of the along-axis point density so they land in the
    gaps between objects rather than at arbitrary equal fractions.
    """
    (cx, cy), (rw, rh), deg = cv2.minAreaRect(w.astype(np.float32))
    th = np.radians(deg)
    axis = np.array([np.cos(th), np.sin(th)]) if rw >= rh else np.array([-np.sin(th), np.cos(th)])
    t = (w - np.array([cx, cy])) @ axis
    order = np.argsort(t)
    ts = t[order]
    hist, edges = np.histogram(ts, bins=max(int(len(ts) ** 0.5), 3 * k))
    hist = ndimage.uniform_filter1d(hist.astype(float), size=3)
    seg = len(hist) / k
    cuts = []
    for i in range(1, k):
        c = int(i * seg)
        lo, hi = max(0, c - int(seg * 0.35)), min(len(hist), c + int(seg * 0.35))
        if hi > lo:
            cuts.append(edges[lo + int(np.argmin(hist[lo:hi]))])
    bounds = [-np.inf] + sorted(cuts) + [np.inf]
    out = []
    for i in range(len(bounds) - 1):
        m = (t > bounds[i]) & (t <= bounds[i + 1])
        if m.sum() >= 30:
            out.append(w[m])
    return out or [w]


def blob_world_points(ys, xs, xmin, xmax, ymin, ymax, res):
    wx = xs / (res - 1) * (xmax - xmin) + xmin
    wy = ys / (res - 1) * (ymax - ymin) + ymin
    return np.column_stack([wx, wy])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale-to-meters", type=float, required=True)
    ap.add_argument("--label", default="table",
                    help="class given to blobs that have NO existing detection; a "
                         "matched box keeps its own label unless that label is not "
                         "already a horizontal-surface class")
    ap.add_argument("--room-yaw-deg", type=float, default=0.0,
                    help="this room's yaw IN THE FRAME OF --boxes. The footprint yaw "
                         "sweep is centred here, which is what resolves the 90-degree "
                         "min-area-rectangle ambiguity toward the room's own grid")
    ap.add_argument("--res", type=int, default=900)
    ap.add_argument("--search-lo-m", type=float, default=0.55,
                    help="low end of the plausible desk-height search range")
    ap.add_argument("--search-hi-m", type=float, default=1.15)
    ap.add_argument("--band-half-m", type=float, default=0.10,
                    help="half-thickness of the band kept around the detected "
                         "tabletop height")
    ap.add_argument("--min-area-m2", type=float, default=0.30)
    ap.add_argument("--max-area-m2", type=float, default=6.00,
                    help="above this a blob is assumed to be several desks pushed "
                         "together and is split")
    ap.add_argument("--split-unit-m2", type=float, default=1.10,
                    help="nominal single-desk footprint used when splitting a merged blob")
    ap.add_argument("--max-long-m", type=float, default=2.10,
                    help="longest single desk; a blob longer than this is split along "
                         "its own long axis, which catches a merged ROW of desks that "
                         "is still small in total area")
    ap.add_argument("--max-aspect", type=float, default=5.0)
    ap.add_argument("--min-fill", type=float, default=0.45,
                    help="minimum blob-area / min-area-rect-area; a real tabletop is "
                         "close to rectangular, a wall smear or floor streak is not")
    ap.add_argument("--opacity-thresh", type=float, default=0.3)
    ap.add_argument("--dedup-dist-m", type=float, default=0.80)
    args = ap.parse_args()

    S = args.scale_to_meters
    p = PlyData.read(args.ply)["vertex"]
    xyz = np.column_stack([p["x"], p["y"], p["z"]]).astype(np.float64)
    opac = 1 / (1 + np.exp(-np.array(p["opacity"], dtype=np.float64)))
    vis = xyz[opac >= args.opacity_thresh]

    floor_z = auto_floor_z(vis, np.ones(len(vis)), 0.0)
    xmin, xmax, ymin, ymax = auto_room_bounds(vis, np.ones(len(vis)), 0.0)

    top_z, npk = dominant_surface_z(vis, floor_z, S, args.search_lo_m, args.search_hi_m)
    if top_z is None:
        print("[tables] no horizontal surface found in the desk-height range")
        Path(args.out).write_text(Path(args.boxes).read_text())
        return
    print(f"[tables] floor_z={floor_z:.4f}  tabletop height="
          f"{(top_z - floor_z) * S:.2f} m above floor ({npk:,} gaussians in peak bin)")

    band_lo_m = (top_z - floor_z) * S - args.band_half_m
    band_hi_m = (top_z - floor_z) * S + args.band_half_m
    mask, _img, _has = load_band_mask_and_image(
        args.ply, xmin, xmax, ymin, ymax, args.res, floor_z,
        band_lo_m, band_hi_m, S, args.opacity_thresh)

    # Close 1-pixel gaps so a tabletop reconstructed with holes stays one blob.
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=2)
    lab, n = ndimage.label(mask)
    m2_per_px = ((xmax - xmin) * S / args.res) * ((ymax - ymin) * S / args.res)
    print(f"[tables] {n} raw components, {m2_per_px * 1e4:.1f} cm^2 per pixel")

    cands = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        area = len(ys) * m2_per_px
        if area < args.min_area_m2:
            continue
        w_all = blob_world_points(ys, xs, xmin, xmax, ymin, ymax, args.res)
        # Split on LENGTH as well as area. A long narrow row of desks can be
        # well under any sane area cap while plainly being more than one desk.
        (_c, (rw0, rh0), _d) = cv2.minAreaRect(w_all.astype(np.float32))
        long_m = max(rw0, rh0) * S
        k_area = int(round(area / args.split_unit_m2)) if area > args.max_area_m2 else 1
        k_len = int(np.ceil(long_m / args.max_long_m)) if long_m > args.max_long_m else 1
        k = max(k_area, k_len)
        parts_w = split_long_blob(w_all, k) if k >= 2 else [w_all]
        frac = [len(q) / max(len(w_all), 1) for q in parts_w]
        for w, f in zip(parts_w, frac):
            a = area * f
            cx, cy, rw, rh, theta = robust_yaw(w, np.radians(args.room_yaw_deg))
            rw_m, rh_m = rw * S, rh * S
            if min(rw_m, rh_m) < 1e-6:
                continue
            aspect = max(rw_m, rh_m) / min(rw_m, rh_m)
            fill = a / max(rw_m * rh_m, 1e-9)
            if aspect > args.max_aspect or fill < args.min_fill:
                continue
            cands.append({"cx": cx, "cy": cy, "rw": rw, "rh": rh,
                          "theta": theta, "area_m2": a,
                          "aspect": aspect, "fill": fill})

    print(f"[tables] {len(cands)} tabletop candidates after area/aspect/fill filters")
    for c in sorted(cands, key=lambda c: -c["area_m2"])[:40]:
        print(f"    {c['rw'] * S:5.2f} x {c['rh'] * S:5.2f} m  area={c['area_m2']:4.2f} m^2  "
              f"aspect={c['aspect']:.2f}  fill={c['fill']:.2f}")

    boxes = json.loads(Path(args.boxes).read_text())["boxes"]
    dedup = args.dedup_dist_m / S
    added = relabelled = kept_label = 0

    for c in cands:
        # Desk box: measured footprint, floor to tabletop.
        height = top_z - floor_z
        center = [float(c["cx"]), float(c["cy"]), float(floor_z + height / 2.0)]
        size = [float(c["rw"]), float(c["rh"]), float(height)]

        # Does this coincide with an existing floor-standing detection?
        hit = None
        for b in boxes:
            d = np.hypot(b["center"][0] - c["cx"], b["center"][1] - c["cy"])
            base_m = (b["center"][2] - b["size"][2] / 2.0 - floor_z) * S
            if d <= dedup and base_m < 0.6 and b["label"] in RELABELLABLE:
                if hit is None or d < hit[0]:
                    hit = (d, b)
        if hit is not None:
            b = hit[1]
            # geometry is adopted either way; the LABEL only changes when the
            # existing one is not already a horizontal-surface class
            if b["label"] in SURFACE_KEEP:
                b["surface_confirmed"] = True
                kept_label += 1
            else:
                b["relabelled_from"] = b["label"]
                b["label"] = args.label
                relabelled += 1
            b["center"], b["size"] = center, size
            b["angle"] = float(c["theta"])
            b["source"] = "topdown_surface"
        else:
            boxes.append({"label": args.label, "center": center, "size": size,
                          "angle": float(c["theta"]), "source": "topdown_surface"})
            added += 1

    Path(args.out).write_text(json.dumps({"boxes": boxes}, indent=2))
    print(f"\n[tables] {relabelled} existing boxes relabelled -> {args.label}, "
          f"{kept_label} kept their already-surface label, {added} new added")
    print(f"  final labels: {dict(Counter(b['label'] for b in boxes))}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
