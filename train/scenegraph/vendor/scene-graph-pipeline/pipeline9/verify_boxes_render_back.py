#!/usr/bin/env python
"""Score each box by re-rendering its own Gaussians into the views it was
detected in.

A box that bounds a real object will, when only its own Gaussians are
rasterised, produce a silhouette that lands where the detector said the object
was. A box enclosing a slab of wall or floor will not, even though it contains
plenty of geometry and passes an occupancy test.

Per box: select its Gaussians, rasterise them into its best detection frames,
and compare the rendered silhouette's bounding box against the recorded 2D
detection. Records the median IoU across frames plus a coverage figure, which
separates "the object is larger than the detection" from "the object is
somewhere else".

Results are written onto each box as `verify_iou` / `verify_coverage`. Pass
--drop to remove failures; by default this annotates only, since a useful
threshold is class-dependent.

Usage:
  python verify_boxes_render_back.py --ply data/scene_derotated.ply \
    --job-dir out_scene --boxes boxes.json --yaw-deg 28.07 --out boxes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))


def yaw_rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def enclosed(xyz, center, size, angle, dilate=1.0):
    local = (xyz - np.asarray(center)) @ yaw_rot(-angle).T
    half = np.asarray(size) / 2.0 * dilate
    return np.where(np.all(np.abs(local) <= half, axis=1))[0]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True, help="DEROTATED splat (the frame poses' frame)")
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--boxes", required=True, help="boxes in the ORIGINAL splat frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--yaw-deg", type=float, required=True,
                    help="boxes are in the original frame; the splat and poses here "
                         "are derotated, so boxes are rotated by -yaw to match")
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--min-iou", type=float, default=0.25)
    ap.add_argument("--min-gaussians", type=int, default=30)
    ap.add_argument("--alpha-thresh", type=float, default=0.15)
    ap.add_argument("--opacity-thresh", type=float, default=0.3)
    ap.add_argument("--drop", action="store_true",
                    help="actually remove failures; default only annotates and reports")
    args = ap.parse_args()

    from gsplat import rasterization

    job = Path(args.job_dir)
    transforms = json.loads((job / "transforms.json").read_text())
    inter = json.loads((job / "interactions.json").read_text())
    objs = inter["objects"]
    boxes = json.loads(Path(args.boxes).read_text())["boxes"]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = PlyData.read(args.ply)["vertex"]
    xyz = np.stack([p["x"], p["y"], p["z"]], axis=1).astype(np.float64)
    quats = np.stack([p["rot_0"], p["rot_1"], p["rot_2"], p["rot_3"]], axis=1).astype(np.float32)
    scales = np.exp(np.stack([p["scale_0"], p["scale_1"], p["scale_2"]], axis=1)).astype(np.float32)
    opac = 1.0 / (1.0 + np.exp(-np.asarray(p["opacity"], dtype=np.float32)))
    keep = opac >= args.opacity_thresh
    xyz, quats, scales, opac = xyz[keep], quats[keep], scales[keep], opac[keep]
    print(f"[verify] {len(xyz):,} gaussians on {dev}")

    W, H = transforms["w"], transforms["h"]
    K = torch.tensor([[transforms["fl_x"], 0, transforms["cx"]],
                      [0, transforms["fl_y"], transforms["cy"]], [0, 0, 1.0]],
                     dtype=torch.float32, device=dev)[None]
    R_inv = yaw_rot(-np.radians(args.yaw_deg))
    frames_meta = transforms["frames"]

    # interactions.json's object order does NOT survive the pipeline: boxes are
    # dropped by the flux filter and CLIP, relabelled by the support prior, and
    # ADDED by the topdown surface detector. Indexing objs[i] by box position
    # silently compares each box against a different object's frames — which
    # reads as IoU 0.000 for every box, not as an error. Match on geometry.
    obj_pos = np.array([[o["position"]["x"], o["position"]["y"], o["position"]["z"]]
                        for o in objs]) if objs else np.zeros((0, 3))

    def frames_for(center_derot, size):
        if len(obj_pos) == 0:
            return []
        # XY only — see the note in refit_box_from_masks.py: grounding moves z.
        d = np.linalg.norm(obj_pos[:, :2] - center_derot[:2], axis=1)
        j = int(np.argmin(d))
        # must be within the box itself, else this box has no detector origin
        if d[j] > max(float(np.linalg.norm(np.asarray(size)[:2])) * 0.6, 1e-9):
            return []
        return sorted(objs[j].get("frames", []), key=lambda x: -x["score"])[:args.n_frames]

    results = []
    for i, b in enumerate(boxes):
        # boxes live in the original frame; splat + poses are derotated
        c_d = R_inv @ np.asarray(b["center"], dtype=np.float64)
        a_d = float(b.get("angle", 0.0)) - np.radians(args.yaw_deg)
        idx = enclosed(xyz, c_d, b["size"], a_d)
        if len(idx) < args.min_gaussians:
            results.append((i, None, None, 0))
            continue

        g_means = torch.tensor(xyz[idx], dtype=torch.float32, device=dev)
        g_quats = torch.tensor(quats[idx], device=dev)
        g_scales = torch.tensor(scales[idx], device=dev)
        g_op = torch.tensor(opac[idx], device=dev)
        g_col = torch.ones((len(idx), 1, 3), device=dev)

        frs = frames_for(c_d, b["size"])
        ious, covs = [], []
        for fr in frs:
            fm = frames_meta[fr["frame_idx"]]
            w2c = torch.tensor(np.linalg.inv(np.array(fm["transform_matrix"], dtype=np.float64)),
                               dtype=torch.float32, device=dev)[None]
            with torch.no_grad():
                _c, alphas, _m = rasterization(
                    means=g_means, quats=g_quats, scales=g_scales, opacities=g_op,
                    colors=g_col, viewmats=w2c, Ks=K, width=W, height=H,
                    sh_degree=0, near_plane=0.01, far_plane=1000.0)
            a = alphas[0, :, :, 0].cpu().numpy()
            ys, xs = np.where(a > args.alpha_thresh)
            if len(ys) < 10:
                continue
            rend = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            det = tuple(float(v) for v in fr["box"])
            ious.append(iou_xyxy(rend, det))
            inside = ((xs >= det[0]) & (xs <= det[2]) & (ys >= det[1]) & (ys <= det[3]))
            covs.append(float(inside.mean()))
        results.append((i, float(np.median(ious)) if ious else None,
                        float(np.median(covs)) if covs else None, len(idx)))
        if (i + 1) % 20 == 0:
            print(f"  [verify] {i + 1}/{len(boxes)}", flush=True)

    kept, stats = [], Counter()
    rows = []
    for (i, iou, cov, ng) in results:
        b = boxes[i]
        b["verify_iou"] = iou
        b["verify_coverage"] = cov
        b["verify_n_gaussians"] = ng
        if iou is None:
            stats["no_evidence"] += 1
            kept.append(b)                       # cannot judge -> do not punish
        elif iou >= args.min_iou:
            stats["pass"] += 1
            kept.append(b)
        else:
            stats["fail"] += 1
            rows.append((b["label"], iou, cov, ng))
            if not args.drop:
                kept.append(b)

    Path(args.out).write_text(json.dumps({"boxes": kept}, indent=2))
    print(f"\n[verify] pass={stats['pass']} fail={stats['fail']} "
          f"no_evidence={stats['no_evidence']}  "
          f"({'dropped' if args.drop else 'annotated only'})")
    if rows:
        print(f"\n  failing (median IoU < {args.min_iou}):")
        for lab, iou, cov, ng in sorted(rows, key=lambda r: r[1])[:25]:
            print(f"    {lab:<18} iou={iou:.3f}  coverage={cov if cov is None else round(cov, 2)}"
                  f"  n_gauss={ng}")
    ok = [r[1] for r in results if r[1] is not None]
    if ok:
        print(f"\n  IoU p10/p50/p90 = {np.percentile(ok, 10):.2f}/"
              f"{np.percentile(ok, 50):.2f}/{np.percentile(ok, 90):.2f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
