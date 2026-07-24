"""Generate real training masks that exclude the camera operator from a splatfacto
dataset's perspective crops — replacing the current all-white (fully-unmasked)
placeholders. Confirmed by direct inspection: the operator's head/hair is visible at
the bottom edge of several pz000 training crops (distinct from the panorama nadir-blur
fix, which only touches the viewer's full-sphere frames, not these re-projected crops).

Uses a YOLO segmentation model (COCO "person" class) to find the operator per-frame,
so only the FEW crops that actually contain them lose any supervision — unlike a blind
fixed geometric band applied to every crop, which was rejected as needlessly wasteful
(the operator is visible in roughly 2-3 of every 12 yaw crops per scan-point, not all).

Verified false-positive mode: at low confidence the detector sometimes fires on the
room's black dome pendant lamps (round + dark, superficially head-like). Verified true
positives (real hair/head crops, checked at full resolution) ALL had their box's top
edge in the bottom ~13% of the 1024px frame -- the operator, standing at ground level
near the tripod-mounted camera, can only ever appear near the bottom of an eye-level
(pz000) crop; the lamp false-positive was near the vertical middle. --min-y-frac
enforces this as a hard positional filter on top of the detector, using the capture
rig's actual geometry rather than trusting class confidence alone.

Must be run with the isolated .mask_venv interpreter (pulls in torch 2.13 + numpy 2.x,
which would silently break the abai env's pinned torch 2.9.0+cu130 / gsplat 1.4.0 stack
if installed there — this venv is fully separate and cannot touch that install).

Usage:
  .mask_venv/bin/python generate_person_masks.py --data <dataset dir> [--margin-px 24]
"""
import argparse, json
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

PERSON_CLASS = 0  # COCO class id for "person"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="dataset dir with transforms.json + images/")
    ap.add_argument("--margin-px", type=int, default=24, help="dilate the excluded region by this many pixels")
    ap.add_argument("--conf", type=float, default=0.15, help="YOLO detection confidence threshold")
    ap.add_argument("--model", default="yolo11x-seg.pt", help="ultralytics segmentation model "
                    "(the nano variant misses this operator almost entirely -- verified empirically; "
                    "the large variant catches clear instances but this is still a best-effort fix, "
                    "not 100% coverage -- some heavily-cropped partial views score zero even at conf=0.01)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write masks here instead of <data>/masks/ (use for verification before overwriting)")
    ap.add_argument("--min-y-frac", type=float, default=0.7,
                    help="reject detections whose box top is above this fraction of image height "
                         "(rejects the verified lamp false-positive mode; the operator is always "
                         "near the bottom of an eye-level pz000 crop)")
    a = ap.parse_args()

    tj = json.loads((a.data / "transforms.json").read_text())
    frames = tj["frames"]
    out_dir = a.out if a.out is not None else (a.data / "masks")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(a.model)
    kernel = np.ones((max(1, a.margin_px), max(1, a.margin_px)), np.uint8)

    n_touched = 0
    for i, fr in enumerate(frames):
        img_path = a.data / fr["file_path"]
        stem = Path(fr.get("mask_path", "masks/" + Path(fr["file_path"]).stem + ".png")).stem
        out_path = out_dir / f"{stem}.png"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  MISSING image {img_path}"); continue
        H, W = img.shape[:2]
        mask = np.full((H, W), 255, dtype=np.uint8)

        res = model.predict(img, classes=[PERSON_CLASS], conf=a.conf, verbose=False)[0]
        hit = False
        if res.masks is not None and len(res.masks.data) > 0:
            boxes_y1 = res.boxes.xyxy[:, 1].cpu().numpy()
            for person_mask, y1 in zip(res.masks.data.cpu().numpy(), boxes_y1):
                if y1 < a.min_y_frac * H:
                    continue                              # rejected: not near the bottom -> likely a lamp
                pm = cv2.resize(person_mask, (W, H), interpolation=cv2.INTER_LINEAR)
                pm = (pm > 0.5).astype(np.uint8)
                pm = cv2.dilate(pm, kernel)              # safety margin around segmentation edges
                mask[pm > 0] = 0
                hit = True
            if hit: n_touched += 1

        cv2.imwrite(str(out_path), mask)
        if (i + 1) % 400 == 0 or i == 0:
            print(f"  {i+1}/{len(frames)}  (masked so far: {n_touched})", flush=True)

    print(f"done: {len(frames)} frames, {n_touched} had a detected person masked out -> {out_dir}")


if __name__ == "__main__":
    main()
