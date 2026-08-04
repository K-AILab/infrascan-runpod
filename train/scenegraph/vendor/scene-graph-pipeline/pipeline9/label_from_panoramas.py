#!/usr/bin/env python
"""Label 3D objects by fusing CLIP embeddings over many views of each object.

Follows OpenMask3D (arXiv 2306.13631): the unit of labelling is the INSTANCE,
not the pixel or the point. For each object, select the top-k views where it
projects largest and closest, cut several crops at increasing scales in each,
encode them all with CLIP and average into ONE embedding before classifying.

Deciding a label per crop and voting afterwards is what makes a row of
identical benches come out split across two names; fusing first removes that
failure mode entirely.

Crops can be read from the perspective views (--source views) or from the
full-resolution panoramas (--source pano). Views are the default and score
better in practice: the panorama route needs a world-to-equirect projection
accurate to a few pixels, and at the accuracy achievable here its crops drift
onto neighbouring objects, which costs more than the extra resolution gains.

Each object also gets a cross-view agreement figure — the share of its own
crops whose individual best guess matches the fused answer. This is reported
instead of CLIP's softmax score because raw confidence from a vision-language
model is poorly calibrated; agreement across views tracks correctness better.

Usage:
  python label_from_panoramas.py --space factory_space_14 --source views \
    --scene-graph out/factory_space_14_sg_pointcloud.json \
    --out out/factory_space_14_viewlabels.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline2b"))
import clip_utils  # noqa: E402
from geo_label_clip import PROMPTS, STRUCTURE_LABELS, VOCAB  # noqa: E402

VIEW_RE = re.compile(r"(\d+)_pz(\d+)_y(\d+)_normal")

# Equirectangular calibration for this capture rig: see the module docstring.
PANO_YAW_SIGN, PANO_PITCH_SIGN, PANO_LON_OFFSET_DEG, PANO_LAT_SIGN = 1, -1, 0.0, -1


def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def cam_dirs_to_pano_uv(d_cam, pitch_deg, yaw_deg, pw, ph):
    """Camera-frame rays -> panorama pixels, in the calibrated convention."""
    pitch = np.radians(PANO_PITCH_SIGN * pitch_deg)
    yaw = np.radians(PANO_YAW_SIGN * yaw_deg + PANO_LON_OFFSET_DEG)
    d = d_cam @ (Ry(yaw) @ Rx(pitch)).T
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    d = d / np.maximum(n, 1e-9)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    u = np.mod((np.arctan2(x, z) / (2 * np.pi) + 0.5) * pw, pw)
    v = (0.5 - np.arcsin(np.clip(PANO_LAT_SIGN * y, -1, 1)) / np.pi) * ph
    return u, np.clip(v, 0, ph - 1)


def box_corners(center, size, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])       # geo frame is Y-up
    h = np.asarray(size, dtype=float) / 2.0
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                out.append(np.asarray(center, dtype=float) + R @ (h * [sx, sy, sz]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True)
    ap.add_argument("--scene-graph", required=True,
                    help="scene graph in the POINT CLOUD frame (cameras.json's frame)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=8,
                    help="views per object, chosen by projected size / distance "
                         "(OpenMask3D's top-k by visibility)")
    ap.add_argument("--levels", type=int, default=3,
                    help="multi-level crops per view")
    ap.add_argument("--level-expansion", type=float, default=0.35,
                    help="each level grows the crop by this fraction; context helps "
                         "CLIP but too much of it labels the room, not the object")
    ap.add_argument("--min-crop-px", type=int, default=48,
                    help="in PANORAMA pixels — the whole point of this pass is that "
                         "objects are large here")
    ap.add_argument("--min-dist-m", type=float, default=0.6)
    ap.add_argument("--max-dist-m", type=float, default=12.0)
    ap.add_argument("--max-margin-px", type=float, default=90.0,
                    help="cap on the context margin added per level, in panorama "
                         "pixels; keeps small objects from being swamped by their "
                         "background")
    ap.add_argument("--exclude-structural", action="store_true", default=True,
                    help="drop wall/floor/ceiling/pillar from the candidate labels")
    ap.add_argument("--source", choices=("pano", "views"), default="pano",
                    help="where the crops come from. 'views' reads the 504x504 "
                         "perspective renders using the SAME world->view projection "
                         "geo_label_clip.py relies on, so it isolates OpenMask3D's "
                         "recipe (top-k views, multi-level crops, averaged embeddings) "
                         "from the panorama projection. If 'views' beats the baseline "
                         "and 'pano' does not, the recipe is sound and the projection "
                         "is what needs fixing")
    ap.add_argument("--point-crops", action="store_true", default=False,
                    help="define each crop from the object's OWN point-cloud points "
                         "rather than its 3D box corners. geo_label_clip.select_views "
                         "records why: a box's corner hull covers up to ~2x the "
                         "object's real screen area from an oblique angle, so the crop "
                         "is mostly its neighbours. That is what made a chair read as "
                         "'workbench' and a light next to an air conditioner read as "
                         "'air conditioner'")
    ap.add_argument("--mask-neighbours", action="store_true", default=False,
                    help="grey out pixels belonging to OTHER detected objects inside "
                         "the crop, so a neighbour that shares the rectangle cannot "
                         "vote on this object's label")
    ap.add_argument("--min-agreement", type=float, default=0.0,
                    help="report only; labels below this are still written but flagged")
    args = ap.parse_args()

    d = REPO / "data" / args.space
    cams = json.loads((d / "cameras.json").read_text())
    cams = cams if isinstance(cams, list) else cams["cameras"]
    K = json.loads((d / "intrinsics.json").read_text())
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
    Wv, Hv = K["width"], K["height"]

    meta = []
    for i, c in enumerate(cams):
        m = VIEW_RE.search(Path(c["pano"]).name)
        if not m:
            continue
        pz = int(m.group(2))
        meta.append({"i": i, "sp": m.group(1),
                     "pitch": pz if pz < 180 else pz - 360,
                     "yaw": int(m.group(3)),
                     "pos": np.asarray(c["pos"], dtype=float),
                     "R": np.asarray(c["R"], dtype=float)})
    POS = np.stack([m["pos"] for m in meta])
    ROT = np.stack([m["R"] for m in meta])
    print(f"[pano-label] {len(meta)} posed views over "
          f"{len({m['sp'] for m in meta})} panoramas")

    sg = json.loads(Path(args.scene_graph).read_text())
    nodes = sg["nodes"]
    yaw_g = -np.radians(float(sg.get("building_yaw_deg", 0.0)))

    # Real scan points, used to define crops that hug each object instead of its
    # box's corner hull.
    node_pts = [None] * len(nodes)
    if args.point_crops:
        from plyfile import PlyData
        pv = PlyData.read(str(d / "pointcloud.ply"))["vertex"]
        P = np.stack([pv["x"], pv["y"], pv["z"]], axis=1).astype(np.float64)
        if len(P) > 900_000:
            P = P[np.random.default_rng(0).choice(len(P), 900_000, replace=False)]
        cg, sg_ = np.cos(-yaw_g), np.sin(-yaw_g)
        Rg = np.array([[cg, 0, sg_], [0, 1, 0], [-sg_, 0, cg]])
        for i, nd in enumerate(nodes):
            ctr = np.asarray(nd.get("box_center", nd["centroid"]), dtype=float)
            hf = np.asarray(nd["bbox_size"], dtype=float) / 2.0
            loc = (P - ctr) @ Rg.T
            m = np.all(np.abs(loc) <= hf, axis=1)
            if m.sum() >= 25:
                q = P[m]
                node_pts[i] = q if len(q) <= 500 else q[
                    np.random.default_rng(0).choice(len(q), 500, replace=False)]
        got = sum(x is not None for x in node_pts)
        print(f"[pano-label] point crops available for {got}/{len(nodes)} objects")

    # Structure is not a candidate here. Every box reaching this stage has
    # already survived relabel_with_clip's structural drop and the support
    # prior, so it is an object by construction — but leaving "ceiling"/"wall"/
    # "floor" in the candidate set let CLIP answer with the BACKGROUND of the
    # crop instead of its subject: 21 of factory_space_14's 25 ceiling lights
    # came back as "ceiling", which is a true statement about the pixels and a
    # useless one about the object.
    vocab = ([w for w in VOCAB if w not in STRUCTURE_LABELS]
             if args.exclude_structural else list(VOCAB))
    model, preprocess, tokenizer, device = clip_utils.load_clip_model()
    text_emb = clip_utils.build_label_text_embeddings(
        model, tokenizer, device, vocab, PROMPTS)
    print(f"[pano-label] CLIP ready on {device}, {len(vocab)} labels"
          + (f" ({len(VOCAB) - len(vocab)} structural excluded)"
             if args.exclude_structural else ""))

    pano_cache: dict[str, Image.Image] = {}

    def pano_for(sp):
        if sp not in pano_cache:
            if len(pano_cache) > 6:
                pano_cache.pop(next(iter(pano_cache)))
            pano_cache[sp] = Image.open(d / "frames" / f"{sp}.jpg").convert("RGB")
        return pano_cache[sp]

    stats = Counter()
    for ni, node in enumerate(nodes):  # noqa: B007 - ni used inside the crop loop
        centre = np.asarray(node.get("box_center", node["centroid"]), dtype=float)
        size = np.asarray(node["bbox_size"], dtype=float)
        corners = box_corners(centre, size, yaw_g)

        # --- top-k views by projected size / distance -----------------------
        pc = np.einsum("nji,nj->ni", ROT, centre - POS)     # R^T (c - pos)
        z, dist = pc[:, 2], np.linalg.norm(pc, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * pc[:, 0] / z + cx
            v = fy * pc[:, 1] / z + cy
        ok = ((z > 0.3) & (dist > args.min_dist_m) & (dist < args.max_dist_m)
              & (u > 0.1 * Wv) & (u < 0.9 * Wv) & (v > 0.1 * Hv) & (v < 0.9 * Hv))
        cand = np.where(ok)[0]
        if len(cand) == 0:
            stats["no_view"] += 1
            node["pano_label"] = None
            continue

        scored, seen_sp = [], set()
        for ci in cand:
            cc = (corners - POS[ci]) @ ROT[ci]
            if (cc[:, 2] > 0.1).mean() < 0.6:
                continue
            area = ((fx * cc[:, 0] / np.maximum(cc[:, 2], .05)).ptp() *
                    (fy * cc[:, 1] / np.maximum(cc[:, 2], .05)).ptp()) / (Wv * Hv)
            scored.append((min(area, 0.8) / float(dist[ci]), int(ci)))
        scored.sort(reverse=True)
        picks = []
        for s, ci in scored:                    # at most one view per panorama
            if meta[ci]["sp"] in seen_sp:
                continue
            seen_sp.add(meta[ci]["sp"])
            picks.append(ci)
            if len(picks) >= args.top_k:
                break
        if not picks:
            stats["no_view"] += 1
            node["pano_label"] = None
            continue

        # --- multi-level crops -----------------------------------------------
        crops = []
        for ci in picks:
            m = meta[ci]
            cc = (corners - POS[ci]) @ ROT[ci]
            front = cc[cc[:, 2] > 0.05]
            if len(front) < 4:
                continue

            if args.source == "views":
                pts_i = node_pts[ni] if args.point_crops else None
                if pts_i is not None:
                    pcc = (pts_i - POS[ci]) @ ROT[ci]
                    fr = pcc[:, 2] > 0.1
                    if fr.sum() < 20:
                        continue
                    uu = fx * pcc[fr, 0] / pcc[fr, 2] + cx
                    vv = fy * pcc[fr, 1] / pcc[fr, 2] + cy
                    inb = (uu >= 0) & (uu < Wv) & (vv >= 0) & (vv < Hv)
                    if inb.sum() < 15:
                        continue
                    # p2/p98 rather than min/max: a few stray points at the box
                    # edge would put the crop back on the neighbour
                    x1, x2 = np.percentile(uu[inb], [2, 98])
                    y1, y2 = np.percentile(vv[inb], [2, 98])
                else:
                    zz = np.maximum(cc[:, 2], 0.05)
                    uu = fx * cc[:, 0] / zz + cx
                    vv = fy * cc[:, 1] / zz + cy
                    x1, x2 = float(uu.min()), float(uu.max())
                    y1, y2 = float(vv.min()), float(vv.max())
                if (x2 - x1) < 12 or (y2 - y1) < 12:
                    continue
                vp = d / "views" / Path(cams[m["i"]]["pano"]).name
                if not vp.exists():
                    continue
                vim = Image.open(vp).convert("RGB")
                if args.mask_neighbours and args.point_crops:
                    # Grey out other objects' points so a neighbour sharing the
                    # rectangle cannot vote. Grey rather than black: a black
                    # blob is itself a strong visual feature.
                    arr = np.asarray(vim).copy()
                    for nj, other in enumerate(node_pts):
                        if nj == ni or other is None:
                            continue
                        occ = (other - POS[ci]) @ ROT[ci]
                        f2 = occ[:, 2] > 0.1
                        if f2.sum() < 10:
                            continue
                        ou = (fx * occ[f2, 0] / occ[f2, 2] + cx).astype(int)
                        ov = (fy * occ[f2, 1] / occ[f2, 2] + cy).astype(int)
                        k = (ou >= 0) & (ou < Wv) & (ov >= 0) & (ov < Hv)
                        for uu_, vv_ in zip(ou[k], ov[k]):
                            arr[max(0, vv_ - 4):vv_ + 5, max(0, uu_ - 4):uu_ + 5] = 128
                    vim = Image.fromarray(arr)
                cxm, cym = (x1 + x2) / 2, (y1 + y2) / 2
                bw, bh = (x2 - x1) / 2, (y2 - y1) / 2
                for lv in range(args.levels):
                    mg = min(lv * args.level_expansion * max(bw, bh), args.max_margin_px / 8)
                    a_, b_ = int(cxm - bw - mg), int(cxm + bw + mg)
                    c_, e_ = int(cym - bh - mg), int(cym + bh + mg)
                    a_, c_ = max(0, a_), max(0, c_)
                    b_, e_ = min(Wv, b_), min(Hv, e_)
                    if b_ - a_ > 8 and e_ - c_ > 8:
                        crops.append(vim.crop((a_, c_, b_, e_)))
                continue
            dirs = front / np.linalg.norm(front, axis=1, keepdims=True)
            pano = pano_for(m["sp"])
            pw, ph = pano.size
            pu, pv = cam_dirs_to_pano_uv(dirs, m["pitch"], m["yaw"], pw, ph)
            # unwrap the seam before taking extents
            if pu.max() - pu.min() > pw / 2:
                pu = np.where(pu > pw / 2, pu - pw, pu)
            x1, x2 = float(pu.min()), float(pu.max())
            y1, y2 = float(pv.min()), float(pv.max())
            if (x2 - x1) < args.min_crop_px or (y2 - y1) < args.min_crop_px:
                continue
            cxm, cym = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = (x2 - x1) / 2, (y2 - y1) / 2
            # Context is added as an ABSOLUTE margin, capped. A fixed fractional
            # expansion is scale-free and therefore wrong: on a large bench 35%
            # is a useful sliver of surroundings, on a small light fixture it is
            # mostly ceiling, and the fused answer follows the majority pixel.
            for lv in range(args.levels):
                margin = min(lv * args.level_expansion * max(bw, bh),
                             args.max_margin_px)
                gx = 1.0 + (margin / max(bw, 1e-6))
                gy = 1.0 + (margin / max(bh, 1e-6))
                g = 1.0
                a, b = int(cxm - bw * gx), int(cxm + bw * gx)
                c_, e = int(max(0, cym - bh * gy)), int(min(ph, cym + bh * gy))
                if b - a < 8 or e - c_ < 8:
                    continue
                im = pano.crop((a % pw, c_, (a % pw) + (b - a), e)) if a >= 0 else \
                    pano.rotate(0).crop((a + pw, c_, b + pw, e))
                if im.size[0] > 4 and im.size[1] > 4:
                    crops.append(im)
        if not crops:
            stats["no_crop"] += 1
            node["pano_label"] = None
            continue

        # --- fuse: average the embeddings, then classify --------------------
        feats = clip_utils.encode_images(model, preprocess, device, crops)
        per_crop = feats @ text_emb.T
        fused = clip_utils.average_normalize(feats) @ text_emb.T
        k = int(np.argmax(fused))
        # cross-view consistency, not CLIP's own score (see module docstring)
        agree = float((per_crop.argmax(axis=1) == k).mean())
        node["pano_label"] = vocab[k]
        node["pano_agreement"] = round(agree, 3)
        node["pano_n_crops"] = len(crops)
        node["pano_topk"] = [vocab[j] for j in np.argsort(-fused)[:3]]
        node["pano_is_structure"] = vocab[k] in STRUCTURE_LABELS
        stats["labelled"] += 1
        if (ni + 1) % 25 == 0:
            print(f"  [pano-label] {ni + 1}/{len(nodes)}", flush=True)

    Path(args.out).write_text(json.dumps(sg, indent=2))
    lab = [n for n in nodes if n.get("pano_label")]
    print(f"\n[pano-label] {stats['labelled']} labelled, "
          f"{stats['no_view']} without a usable view, {stats['no_crop']} too small")
    if lab:
        ag = np.array([n["pano_agreement"] for n in lab])
        nc = np.array([n["pano_n_crops"] for n in lab])
        print(f"[pano-label] crops/object median {int(np.median(nc))}, "
              f"cross-view agreement p25/p50/p75 = "
              f"{np.percentile(ag,25):.2f}/{np.percentile(ag,50):.2f}/{np.percentile(ag,75):.2f}")
        print(f"[pano-label] labels: {dict(Counter(n['pano_label'] for n in lab))}")
        changed = [n for n in lab if n["pano_label"] != n.get("label")]
        print(f"[pano-label] differs from the current label on {len(changed)}/{len(lab)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
