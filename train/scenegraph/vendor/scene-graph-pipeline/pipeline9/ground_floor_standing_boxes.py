#!/usr/bin/env python
"""Extend a floor-standing box down to the floor.

Thin, poorly reconstructed parts of furniture — chair castors, table legs, a
cabinet kickplate — carry very few Gaussians, so a fitted extent typically
starts part-way up the object while its top is already correct. This pass
closes that gap.

It only applies where it is physically safe:
  * the class must be floor-standing (lights and wall fixtures are skipped);
  * the gap must be small enough to be a reconstruction artefact rather than a
    real elevation (--max-gap-m);
  * the object must not be resting on another detection, which distinguishes a
    chair with missing legs from a monitor standing on a desk.

Usage:
  python ground_floor_standing_boxes.py --ply data/scene.ply \
    --boxes boxes.json --scale-to-meters 6.8 --out boxes_grounded.json
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
from geo_bounds import auto_floor_z  # noqa: E402

NON_FLOOR = {"light", "lamp", "air_duct", "air_conditioner", "smoke_detector",
             "projector", "ceiling_fixture", "window", "door", "curtain",
             "whiteboard", "wall"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale-to-meters", type=float, required=True)
    ap.add_argument("--max-gap-m", type=float, default=0.55,
                    help="largest base-to-floor gap still treated as a missing-legs "
                         "artefact rather than a genuinely elevated object")
    ap.add_argument("--stack-tol-m", type=float, default=0.20)
    ap.add_argument("--opacity-thresh", type=float, default=0.3)
    args = ap.parse_args()

    S = args.scale_to_meters
    p = PlyData.read(args.ply)["vertex"]
    xyz = np.column_stack([p["x"], p["y"], p["z"]]).astype(np.float64)
    op = 1 / (1 + np.exp(-np.array(p["opacity"], dtype=np.float64)))
    floor_z = auto_floor_z(xyz[op >= args.opacity_thresh], np.ones(int((op >= args.opacity_thresh).sum())), 0.0)

    boxes = json.loads(Path(args.boxes).read_text())["boxes"]
    max_gap = args.max_gap_m / S
    stack_tol = args.stack_tol_m / S

    changed = Counter()
    moved = []
    for b in boxes:
        if b["label"] in NON_FLOOR:
            continue
        base = b["center"][2] - b["size"][2] / 2.0
        gap = base - floor_z
        if gap <= 1e-9 or gap > max_gap:
            continue

        # resting on another detection? then the gap is real, not missing legs
        on_other = False
        for o in boxes:
            if o is b:
                continue
            o_top = o["center"][2] + o["size"][2] / 2.0
            d = np.hypot(b["center"][0] - o["center"][0], b["center"][1] - o["center"][1])
            reach = (np.linalg.norm(np.asarray(b["size"])[:2]) +
                     np.linalg.norm(np.asarray(o["size"])[:2])) / 2.0
            if abs(base - o_top) <= stack_tol and d <= reach:
                on_other = True
                break
        if on_other:
            changed["left_on_support"] += 1
            continue

        top = b["center"][2] + b["size"][2] / 2.0
        b["size"][2] = float(top - floor_z)
        b["center"][2] = float((top + floor_z) / 2.0)
        b["grounded"] = True
        changed[b["label"]] += 1
        moved.append((b["label"], gap * S, b["size"][2] * S))

    Path(args.out).write_text(json.dumps({"boxes": boxes}, indent=2))
    print(f"[ground] floor_z={floor_z:.4f}")
    for k, v in changed.most_common():
        print(f"    {k:<20} {v}")
    if moved:
        by = {}
        for lab, gap, h in moved:
            by.setdefault(lab, []).append((gap, h))
        print(f"\n  {'label':<18}{'n':>3}{'median lift':>13}{'new height':>12}")
        for lab in sorted(by):
            g = np.median([x[0] for x in by[lab]]); h = np.median([x[1] for x in by[lab]])
            print(f"  {lab:<18}{len(by[lab]):>3}{g:>12.2f}m{h:>11.2f}m")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
