#!/usr/bin/env python
"""Convert a splat_analyzer interactions.json (computed on the DE-ROTATED
splat) into this project's tri-viewer box format, in the ORIGINAL
(un-rotated) frame.

Every round of the shinhan_space investigation ran detection against
data/shinhan_hires_30k_derotated.ply (de-rotated by -28.072 degrees so an
axis-aligned box-fitter would work correctly — see RUN_NOTES.md's
"de-rotation" section and [[project-shinhan-space-yaw]]), which means
every raw interactions.json needs the SAME +28.072 degree rotation applied
back before it means anything in the viewer's (original-frame) splat.
This was previously a hand-copied Python snippet repeated ~15 times across
RUN_NOTES.md — this script replaces all of those.

Usage:
  python rotate_and_export.py --interactions out_X/interactions.json --out boxes.json
  cp boxes.json ../../tri-viewer/modes/threed/scene/shinhan_space.boxes.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

YAW_DEG = 28.072  # this project's shinhan_space room yaw — see project-shinhan-space-yaw memory


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--interactions", required=True, help="path to a splat_analyzer interactions.json")
    ap.add_argument("--out", required=True, help="output path for the viewer's boxes.json")
    ap.add_argument("--yaw-deg", type=float, default=YAW_DEG)
    args = ap.parse_args()

    t = np.radians(args.yaw_deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    d = json.loads(Path(args.interactions).read_text())
    boxes = []
    for o in d["objects"]:
        p, sc = o["position"], o["scale"]
        pos_orig = R @ np.array([p["x"], p["y"], p["z"]])
        boxes.append({"label": o["label"], "center": [float(v) for v in pos_orig],
                      "size": [sc["x"], sc["y"], sc["z"]], "angle": float(t)})

    Path(args.out).write_text(json.dumps({"boxes": boxes}, indent=2))
    from collections import Counter
    print(f"wrote {len(boxes)} boxes -> {args.out}")
    print(Counter(b["label"] for b in boxes))


if __name__ == "__main__":
    main()
