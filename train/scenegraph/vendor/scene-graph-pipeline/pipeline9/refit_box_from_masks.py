#!/usr/bin/env python
"""Re-derive each object's 3D box from SAM MASKS of its own detections,
instead of from their bounding boxes.

GaussianGraph (arXiv 2503.04034) §III-A pairs the open-vocabulary detector
with SAM and matches boxes to masks by IoU, using the mask — not the box —
as the object's 2D support. That difference is what this script is for.

A 2D box is an axis-aligned rectangle around a non-rectangular object, so
its interior contains whatever sits behind and beside the object. Lifting
that interior to 3D drags the neighbours in. Measured on shinhan_space,
office chairs came out with 0.79 x 0.99 m footprints against a real office
chair's ~0.65 x 0.65 — the extra area is the desk the chair is pulled up to.
A mask has the object's actual silhouette, so the lifted points belong to the
object and nothing else. It is also the information a purely spatial
clustering lacks: with no way to tell an object's Gaussians from its
neighbour's it has to guess, which is how chairs fragment into dozens of
pieces.

Method, per object:
  1. Take the frames the detector already recorded for it (interactions.json
     stores per-object `frames` with their 2D boxes and scores) and keep the
     best --n-frames by score.
  2. Prompt SAM with each of those boxes to get the object's silhouette.
  3. Back-project the MASK's pixels through that frame's depth map into world
     space, keeping only pixels with real accumulated alpha.
  4. Pool the points from every frame and fit an oriented box: yaw from a
     min-area rectangle over the footprint, extents from robust percentiles.

BACK-VIEW SELECTION (--back-view-labels). For a chair at a desk, most views
see the chair THROUGH or AGAINST the desk, so even a perfect mask covers both
objects. Measured on shinhan_space: after deleting every desktop from the
Gaussians, the material left around each chair detection still spanned
0.85-1.95 m at seat height (desk legs and neighbouring chairs) while almost
nothing survived above 0.65 m, i.e. the backrests sit inside the desk
footprints. Extent-fitting cannot separate that.

The views that CAN are the ones looking at the chair from the side away from
its desk, where the chair's back is the nearest surface and the desk is behind
it. Those are selected geometrically: keep a frame only if the direction from
camera to chair and the direction from chair to its nearest surface point the
same way, i.e. the desk lies BEYOND the chair along the line of sight. The mask
from such a frame is the chair's back, which is what identifies it.

Multi-view pooling is what makes this better than any single mask: one view
sees one side of a chair, and the union across views closes the shape. Points
are additionally trimmed to their own spatial mode before fitting, since a
mask that leaks onto the floor in one frame would otherwise stretch the box.

SAM comes from `ultralytics` (already a dependency), which downloads
mobile_sam.pt on first use — no extra install.

Usage:
  python refit_box_from_masks.py \
    --job-dir ../external/splat_analyzer/out_shinhan_space \
    --boxes out/shinhan_space_boxes_final.json --yaw-deg 28.072 \
    --scale-to-meters 6.8 --out out/shinhan_boxes_masked.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def yaw_rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def unproject(mask, depth, alpha, K_inv, c2w, alpha_thresh, max_px):
    """Mask pixels -> world points, using the frame's own depth map."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((0, 3))
    d = depth[ys, xs]
    ok = d > 0.01
    if alpha is not None:
        ok &= alpha[ys, xs] > alpha_thresh
    ys, xs, d = ys[ok], xs[ok], d[ok]
    if len(ys) == 0:
        return np.zeros((0, 3))
    if len(ys) > max_px:
        sel = np.random.default_rng(0).choice(len(ys), max_px, replace=False)
        ys, xs, d = ys[sel], xs[sel], d[sel]
    pix = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
    rays = K_inv @ pix                       # (3,N) camera-space directions
    pts_cam = rays * (d / rays[2])[None, :]
    return (c2w[:3, :3] @ pts_cam).T + c2w[:3, 3]


def spatial_mode(pts, keep=0.85):
    """Trim to the densest cluster of points before fitting.

    A mask that leaks onto the floor or a wall in one frame contributes a
    thin tail of points far from the object; a robust percentile still gets
    pulled by it because the tail is systematic, not symmetric noise.
    """
    if len(pts) < 20:
        return pts
    c = np.median(pts, axis=0)
    d = np.linalg.norm(pts - c, axis=1)
    return pts[d <= np.quantile(d, keep)]


def oriented_box(pts, lo_pct, hi_pct, room_yaw_rad=0.0, sweep_deg=45.0, step_deg=1.0):
    """Oriented box from pooled mask points.

    The yaw is found by sweeping a 90-degree window CENTRED ON THE ROOM YAW and
    scoring robust (percentile) extents, not by cv2.minAreaRect. minAreaRect
    returns an arbitrary angle in [0, 90) fitted to the exact convex hull, and
    since this pass rewrites `angle` on every box it touched, it was the main
    source of the scattered orientations seen in the viewer — yaw mod 90 had a
    standard deviation of 31.7 degrees on factory_space_13. Sweeping around the
    room grid also resolves the 90-degree ambiguity toward the representative
    that matches the room.
    """
    best, th = np.inf, room_yaw_rad
    for d in np.arange(-sweep_deg, sweep_deg, step_deg):
        t = room_yaw_rad + np.radians(d)
        c, sn = np.cos(-t), np.sin(-t)
        loc2 = pts[:, :2] @ np.array([[c, -sn], [sn, c]]).T
        ext = np.percentile(loc2, hi_pct, axis=0) - np.percentile(loc2, lo_pct, axis=0)
        if float(ext[0] * ext[1]) < best:
            best, th = float(ext[0] * ext[1]), t
    loc = pts @ yaw_rot(-th).T
    lo = np.percentile(loc, lo_pct, axis=0)
    hi = np.percentile(loc, hi_pct, axis=0)
    size = hi - lo
    center = yaw_rot(th) @ ((hi + lo) / 2.0)
    return center, size, th


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--boxes", required=True,
                    help="boxes in the ORIGINAL splat frame (post rotate_and_export)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--yaw-deg", type=float, required=True,
                    help="detection ran on the DEROTATED splat, so mask points come "
                         "back in that frame and need +yaw to match --boxes")
    ap.add_argument("--scale-to-meters", type=float, required=True)
    ap.add_argument("--sam-model", default="mobile_sam.pt")
    ap.add_argument("--n-frames", type=int, default=6,
                    help="highest-scoring frames per object to run SAM on")
    ap.add_argument("--alpha-thresh", type=float, default=0.5)
    ap.add_argument("--max-px-per-frame", type=int, default=4000)
    ap.add_argument("--min-points", type=int, default=150)
    ap.add_argument("--lo-pct", type=float, default=2.0)
    ap.add_argument("--hi-pct", type=float, default=98.0)
    ap.add_argument("--only-labels", default=None,
                    help="restrict refitting to these labels. Masks are not uniformly "
                         "better: they tighten compact objects well, but an obliquely "
                         "viewed flat surface gives only a partial silhouette, so they "
                         "under-measure tables and benches. Apply to compact classes "
                         "and leave work surfaces to detect_tables_topdown.py")
    ap.add_argument("--back-view-labels", default="chair,office_chair,seat,stool,armchair",
                    help="labels for which frames are restricted to those seeing the "
                         "object from the side away from its nearest surface")
    ap.add_argument("--surface-labels", default="table,desk,workbench,bench,counter",
                    help="what counts as the surface a back-view label sits at")
    ap.add_argument("--back-view-dot", type=float, default=0.25,
                    help="minimum dot(camera->object, object->surface); higher is "
                         "stricter about looking at the object's free side")
    ap.add_argument("--max-grow", type=float, default=2.0,
                    help="reject a refit whose diagonal grew by more than this factor "
                         "(a mask that escaped onto the wall)")
    args = ap.parse_args()

    job = Path(args.job_dir)
    inter = json.loads((job / "interactions.json").read_text())
    transforms = json.loads((job / "transforms.json").read_text())
    boxes = json.loads(Path(args.boxes).read_text())["boxes"]
    objs = inter["objects"]
    if len(objs) < len(boxes):
        print(f"[masks] NOTE: {len(boxes)} boxes but {len(objs)} detector objects — "
              f"boxes were filtered downstream; matching by index where possible")

    K = np.array([[transforms["fl_x"], 0, transforms["cx"]],
                  [0, transforms["fl_y"], transforms["cy"]], [0, 0, 1.0]])
    K_inv = np.linalg.inv(K)

    from ultralytics import SAM
    sam = SAM(args.sam_model)

    # Map each --boxes entry to the detector object it came from BY POSITION.
    # interactions.json's ordering does not survive the pipeline (the flux
    # filter and CLIP drop boxes, the topdown detector adds them), so indexing
    # objs[i] by box position compares each box against a different object's
    # frames — which fails silently rather than erroring.
    R_inv = yaw_rot(-np.radians(args.yaw_deg))
    obj_pos = np.array([[o["position"]["x"], o["position"]["y"], o["position"]["z"]]
                        for o in objs]) if objs else np.zeros((0, 3))
    box_of_obj: dict = {}
    for bi, b in enumerate(boxes):
        if not len(obj_pos):
            break
        c_d = R_inv @ np.asarray(b["center"], dtype=np.float64)
        # XY only: ground_floor_standing_boxes.py deliberately moves a box's z
        # centre (up to ~0.22 m on chairs), so a 3D distance test rejects the very
        # boxes that were corrected — 11 of 24 chairs went unmatched that way.
        d = np.linalg.norm(obj_pos[:, :2] - c_d[:2], axis=1)
        j = int(np.argmin(d))
        if d[j] <= max(float(np.linalg.norm(np.asarray(b["size"])[:2])) * 0.6, 1e-9):
            box_of_obj[j] = bi
    print(f"[masks] matched {len(box_of_obj)}/{len(boxes)} boxes to a detector object")
    n_back: dict = {}

    back_labels = {x.strip() for x in args.back_view_labels.split(",") if x.strip()}
    surf_labels = {x.strip() for x in args.surface_labels.split(",") if x.strip()}
    surfaces = [b for b in boxes if b["label"] in surf_labels]
    cam_pos = np.array([np.array(f["transform_matrix"])[:3, 3]
                        for f in transforms["frames"]])

    def keeps_frame(bi, frame_idx):
        """For a back-view label, is this camera on the object's free side?"""
        b = boxes[bi]
        if b["label"] not in back_labels or not surfaces:
            return True
        c = R_inv @ np.asarray(b["center"], dtype=np.float64)      # derotated frame
        sc = np.array([R_inv @ np.asarray(t["center"], dtype=np.float64) for t in surfaces])
        near = sc[int(np.argmin(np.linalg.norm(sc[:, :2] - c[:2], axis=1)))]
        v_surf = near[:2] - c[:2]
        n = np.linalg.norm(v_surf)
        if n < 1e-9:
            return True
        v_surf = v_surf / n
        v_cam = c[:2] - cam_pos[frame_idx][:2]
        n2 = np.linalg.norm(v_cam)
        if n2 < 1e-9:
            return True
        return float(np.dot(v_cam / n2, v_surf)) >= args.back_view_dot

    # frame index -> [(box index, 2D box)], so each frame is opened once and
    # SAM is prompted with all of that frame's boxes together.
    per_frame: dict = {}
    for oi, o in enumerate(objs):
        if oi not in box_of_obj:
            continue
        bi = box_of_obj[oi]
        ok_frames = [fr for fr in sorted(o.get("frames", []), key=lambda x: -x["score"])
                     if keeps_frame(bi, fr["frame_idx"])]
        if boxes[bi]["label"] in back_labels:
            n_back[bi] = (len(ok_frames), len(o.get("frames", [])))
        for fr in ok_frames[:args.n_frames]:
            per_frame.setdefault(fr["frame_idx"], []).append((bi, fr["box"]))

    if n_back:
        kept = sum(v[0] for v in n_back.values()); tot = sum(v[1] for v in n_back.values())
        print(f"[masks] back-view filter: {kept}/{tot} frames kept across "
              f"{len(n_back)} back-view objects "
              f"({sum(1 for v in n_back.values() if v[0] == 0)} left with none)")

    pts_by_obj: dict = {}
    frames_meta = transforms["frames"]
    for n, (fi, items) in enumerate(sorted(per_frame.items())):
        fm = frames_meta[fi]
        img = job / fm["file_path"]
        dnpy = job / fm.get("depth_path", "").replace(".png", ".npy")
        if not img.exists() or not dnpy.exists():
            continue
        depth = np.load(dnpy).astype(np.float64)
        alpha = None
        ap_rel = fm.get("alpha_path")
        if ap_rel and (job / ap_rel).exists():
            alpha = np.load(job / ap_rel).astype(np.float64)
        c2w = np.array(fm["transform_matrix"], dtype=np.float64)

        res = sam(str(img), bboxes=[b for _oi, b in items], verbose=False)
        if not res or res[0].masks is None:
            continue
        masks = res[0].masks.data.cpu().numpy().astype(bool)
        for (oi, _b), m in zip(items, masks):
            if m.shape != depth.shape:
                m = cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            w = unproject(m, depth, alpha, K_inv, c2w, args.alpha_thresh,
                          args.max_px_per_frame)
            if len(w):
                pts_by_obj.setdefault(oi, []).append(w)
        if (n + 1) % 25 == 0:
            print(f"  [masks] {n + 1}/{len(per_frame)} frames", flush=True)

    S = args.scale_to_meters
    only = ({x.strip() for x in args.only_labels.split(",") if x.strip()}
            if args.only_labels else None)
    stats = Counter()
    rows = []
    for i, b in enumerate(boxes):
        if only is not None and b["label"] not in only:
            stats["skipped_label"] += 1
            continue
        chunks = pts_by_obj.get(i)
        if not chunks:
            stats["no_masks"] += 1
            continue
        pts_derot = np.vstack(chunks)
        if len(pts_derot) < args.min_points:
            stats["too_few_points"] += 1
            continue
        # detection ran on the derotated splat; --boxes are in the original frame
        pts = pts_derot @ yaw_rot(np.radians(args.yaw_deg)).T
        pts = spatial_mode(pts)
        c, s, th = oriented_box(pts, args.lo_pct, args.hi_pct,
                                room_yaw_rad=np.radians(args.yaw_deg))
        old_d = float(np.linalg.norm(b["size"]))
        if float(np.linalg.norm(s)) > args.max_grow * old_d or float(np.min(s)) <= 0:
            stats["implausible_growth"] += 1
            continue
        rows.append((b["label"], np.array(b["size"]) * S, s * S,
                     abs(np.degrees(th) - np.degrees(b.get("angle", 0.0))), len(pts)))
        b["center"] = [float(v) for v in c]
        b["size"] = [float(v) for v in s]
        b["angle"] = float(th)
        b["n_mask_points"] = int(len(pts))
        stats["refit"] += 1

    Path(args.out).write_text(json.dumps({"boxes": boxes}, indent=2))
    print(f"\n[masks] refit {stats['refit']}/{len(boxes)}; " +
          ", ".join(f"{k}={v}" for k, v in stats.items() if k != "refit"))
    if rows:
        by = {}
        for lab, o, nn, dth, n in rows:
            by.setdefault(lab, []).append((o, nn, dth, n))
        print(f"\n{'label':<17}{'n':>4}  {'old size (m)':<22}{'new size (m)':<22}{'pts':>7}")
        for lab in sorted(by):
            r = by[lab]
            print(f"{lab:<17}{len(r):>4}  "
                  f"{str(np.median([x[0] for x in r], axis=0).round(2)):<22}"
                  f"{str(np.median([x[1] for x in r], axis=0).round(2)):<22}"
                  f"{int(np.median([x[3] for x in r])):>7}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
