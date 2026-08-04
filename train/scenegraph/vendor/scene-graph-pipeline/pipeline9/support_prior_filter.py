#!/usr/bin/env python
"""Drop or reclassify detections that nothing physically supports.

A real object rests on something. This pass checks each box against that
constraint and removes the ones that fail, which is the most reliable way to
eliminate false positives that look plausible in 2D but float in mid-air.

Support is satisfied by any one of:
  1. the box's base sits within --floor-tol-m of the floor;
  2. another detection's top surface is within --stack-tol-m below the base and
     their ground projections overlap (a monitor on a desk);
  3. the point cloud itself has a continuous column of geometry under the
     footprint, with no vertical gap larger than --stack-tol-m. This covers
     support by furniture that was never detected.

Unsupported boxes are resolved by where they actually are: near the ceiling
they become --ceiling-label, otherwise they are dropped.

Ceiling- and wall-mounted classes are checked against their own constraint
instead: a light must be near the ceiling, a window near a wall.

Usage:
  python support_prior_filter.py --ply data/scene.ply --boxes boxes.json \
    --scale-to-meters 6.8 --out boxes_supported.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo_bounds import auto_ceiling_z, auto_floor_z  # noqa: E402

CEILING_CLASSES = {"light", "lamp", "air_duct", "air_conditioner", "smoke_detector",
                   "projector", "ceiling_fixture"}
WALL_CLASSES = {"window", "door", "curtain", "whiteboard", "wall"}


def yaw_rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def footprint_mask(xy, center, size, angle, pad=0.0):
    """Points whose ground projection falls inside the box's oriented footprint."""
    R = yaw_rot(-angle)[:2, :2]
    loc = (xy - np.asarray(center)[:2]) @ R.T
    half = np.asarray(size)[:2] / 2.0 + pad
    return np.all(np.abs(loc) <= half, axis=1)


def footprints_overlap(a, b):
    """Cheap axis-aligned overlap of two oriented footprints, via their
    circumscribed radii — deliberately permissive, since this only has to
    decide 'could a rest on b', not measure an area."""
    ca, sa = np.asarray(a["center"])[:2], np.asarray(a["size"])[:2]
    cb, sb = np.asarray(b["center"])[:2], np.asarray(b["size"])[:2]
    return np.linalg.norm(ca - cb) <= (np.linalg.norm(sa) + np.linalg.norm(sb)) / 2.0


def column_supported(xy_all, z_all, box, floor_z, stack_tol):
    """True if the splat has a continuous column of Gaussians from the floor up
    to the box's base under its own footprint (no gap > stack_tol)."""
    base = box["center"][2] - box["size"][2] / 2.0
    m = footprint_mask(xy_all, box["center"], box["size"], box.get("angle", 0.0))
    if not m.any():
        return False
    z = z_all[m]
    z = z[(z >= floor_z - stack_tol) & (z <= base + stack_tol)]
    if z.size < 5:
        return False
    zs = np.sort(np.concatenate([[floor_z], z, [base]]))
    return float(np.max(np.diff(zs))) <= stack_tol


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale-to-meters", type=float, required=True)
    ap.add_argument("--floor-tol-m", type=float, default=0.25,
                    help="a floor-standing object's base must be within this of the floor")
    ap.add_argument("--stack-tol-m", type=float, default=0.20,
                    help="largest vertical gap still counted as resting on something")
    ap.add_argument("--ceiling-tol-m", type=float, default=0.60,
                    help="top within this of the ceiling counts as ceiling-mounted")
    ap.add_argument("--ceiling-label", default="light",
                    help="class an unsupported ceiling-height box is reassigned to")
    ap.add_argument("--opacity-thresh", type=float, default=0.3)
    ap.add_argument("--drop-unsupported", action="store_true", default=True)
    args = ap.parse_args()

    S = args.scale_to_meters
    floor_tol = args.floor_tol_m / S
    stack_tol = args.stack_tol_m / S
    ceil_tol = args.ceiling_tol_m / S

    p = PlyData.read(args.ply)["vertex"]
    xyz = np.stack([p["x"], p["y"], p["z"]], axis=1).astype(np.float64)
    opac = 1.0 / (1.0 + np.exp(-np.asarray(p["opacity"], dtype=np.float64)))
    xyz = xyz[opac >= args.opacity_thresh]
    floor_z = auto_floor_z(xyz, np.ones(len(xyz)), 0.0)
    ceil_z = auto_ceiling_z(xyz, np.ones(len(xyz)), 0.0)
    xy_all, z_all = xyz[:, :2], xyz[:, 2]
    print(f"[support] floor_z={floor_z:.4f} ceiling_z={ceil_z:.4f} "
          f"(room height {(ceil_z - floor_z) * S:.2f} m)")

    boxes = json.loads(Path(args.boxes).read_text())["boxes"]

    kept, actions = [], Counter()
    detail = []
    for b in boxes:
        label = b["label"]
        base = b["center"][2] - b["size"][2] / 2.0
        top = b["center"][2] + b["size"][2] / 2.0
        h_m = (b["center"][2] - floor_z) * S

        if label in CEILING_CLASSES:
            ok = (ceil_z - top) <= ceil_tol
            if ok:
                kept.append(b); actions["ceiling_ok"] += 1
            else:
                actions["ceiling_but_low_dropped"] += 1
                detail.append((label, h_m, "ceiling class but not near ceiling", None))
            continue
        if label in WALL_CLASSES:
            kept.append(b); actions["wall_class_kept"] += 1
            continue

        # floor-standing class: does anything hold it up?
        on_floor = (base - floor_z) <= floor_tol
        on_object = False
        if not on_floor:
            for o in boxes:
                if o is b:
                    continue
                o_top = o["center"][2] + o["size"][2] / 2.0
                if -stack_tol <= (base - o_top) <= stack_tol and footprints_overlap(b, o):
                    on_object = True
                    break
        on_geom = False
        if not (on_floor or on_object):
            on_geom = column_supported(xy_all, z_all, b, floor_z, stack_tol)

        if on_floor or on_object or on_geom:
            kept.append(b)
            actions["supported_floor" if on_floor else
                     ("supported_object" if on_object else "supported_geometry")] += 1
            continue

        # unsupported — resolve by where it actually is
        if (ceil_z - top) <= ceil_tol:
            detail.append((label, h_m, f"unsupported at ceiling -> {args.ceiling_label}", None))
            b = dict(b); b["label"] = args.ceiling_label
            b["relabelled_from"] = label
            kept.append(b); actions["ceiling_relabelled"] += 1
        else:
            actions["floating_dropped"] += 1
            detail.append((label, h_m, "floating, unsupported", None))

    Path(args.out).write_text(json.dumps({"boxes": kept}, indent=2))

    print(f"[support] {len(boxes)} -> {len(kept)} boxes")
    for k, v in sorted(actions.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<26} {v}")
    if detail:
        print("\n  changed/dropped:")
        agg = Counter((lab, why) for lab, _h, why, _ in detail)
        hs = {}
        for lab, h, why, _ in detail:
            hs.setdefault((lab, why), []).append(h)
        for (lab, why), n in agg.most_common():
            print(f"    {lab:<18} x{n:<3} median height {np.median(hs[(lab, why)]):.2f} m — {why}")
    print(f"\n  final labels: {dict(Counter(b['label'] for b in kept))}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
