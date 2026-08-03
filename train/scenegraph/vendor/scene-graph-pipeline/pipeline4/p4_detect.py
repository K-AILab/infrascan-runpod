#!/usr/bin/env python
"""pipeline4 stage 1: 3DETR object detection over a full space.

Runs the vendored 3DETR (PointNet++ set-abstraction pre-encoder + transformer
encoder/decoder, pretrained on ScanNet) in a sliding window over the scene,
merges detections across windows with 3D NMS, and writes the result in the
pipeline2/2b "stage A" geometry-JSON format so the existing labeling and
scene-graph stages plug in unchanged:

    python pipeline4/p4_detect.py           --space factory_space_14
    python pipeline2b/geo_label_clip.py     --space factory_space_14 \
        --geo-json pipeline4/out/factory_space_14_p4_geo.json
    python pipeline2b/geo_to_scenegraph.py  --space factory_space_14 \
        --geo-json pipeline4/out/factory_space_14_p4_geo.json \
        --out-space factory_space_14_p4 --no-structure-filters \
        --merge-fragments --no-audit-prune

See pipeline4/README.md for what --merge-fragments/--no-audit-prune do and
why they matter for this pipeline's box population specifically.

World frame: meters, +Y up (same as the rest of the repo).
Model frame: meters, +Z up (ScanNet): model = (x, -z, y), floor at z=0.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline4.detr3d import Detr3DDetector, SCANNET_CLASSES  # noqa: E402

FLOOR_PCTL = 1.0   # same conventions as pipeline2b/geo_cluster.py
CEIL_PCTL = 99.0


# ---------------------------------------------------------------- I/O helpers
def space_registry():
    return json.loads((REPO / "spaces.json").read_text())["spaces"]


def resolve_ply(space: str, override: str | None) -> Path:
    if override:
        return Path(override)
    sp = space_registry()[space]
    p = REPO / sp["data_root"] / "pointcloud.ply"
    if p.exists():
        return p
    web = REPO / "ui" / "_spaces" / space / "Data_" / "downsampled_web.ply"
    if web.exists():
        return web
    raise FileNotFoundError(f"no point cloud found for space {space}")


def load_ply(path: Path):
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"]
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    rgb = None
    names = v.data.dtype.names
    if "red" in names and "green" in names and "blue" in names:
        rgb = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float32) / 255.0
    return xyz, rgb


# ------------------------------------------------------------- geometry utils
def world_to_model(xyz_w: np.ndarray, y_sign: float = 1.0) -> np.ndarray:
    """(x, y_raw, z) -> (x, -z, y_raw*y_sign): right-handed Y-up -> Z-up
    rotation. y_sign corrects a space whose raw Y axis increases DOWNWARD
    instead of upward (see main()'s y_invert handling) — the model frame
    fed to 3DETR must be properly Z-up regardless of the raw ply's own
    convention. y_sign is a detection-input-only correction: it must NOT
    be baked into xyz_w itself, since xyz_w's raw Y is also what
    cameras.json's camera positions are defined against, and geo_label_
    clip.py's view-selection projects object points against those SAME
    cameras — keeping xyz_w untouched keeps that projection consistent."""
    return np.column_stack([xyz_w[:, 0], -xyz_w[:, 2], xyz_w[:, 1] * y_sign])


def model_box_to_world(bmin_m, bmax_m, floor_z_eff, y_sign: float = 1.0):
    """AABB in shifted model frame -> AABB in world (raw xyz_w Y
    convention — NOT necessarily "increases upward" if y_sign=-1 for this
    space; see world_to_model). floor_z_eff is the INTERNAL, y_sign-
    corrected floor reference used to build the model frame, not the
    output-facing floor_y."""
    bmin_m = bmin_m.copy(); bmax_m = bmax_m.copy()
    bmin_m[2] += floor_z_eff; bmax_m[2] += floor_z_eff
    y_a = bmin_m[2] * y_sign
    y_b = bmax_m[2] * y_sign
    y_lo, y_hi = (y_a, y_b) if y_a <= y_b else (y_b, y_a)
    wmin = np.array([bmin_m[0], y_lo, -bmax_m[1]])
    wmax = np.array([bmax_m[0], y_hi, -bmin_m[1]])
    return wmin, wmax


def iou_aabb(box, boxes):
    """box: (6,) [min,max]; boxes: (N,6) -> (N,) IoU."""
    lo = np.maximum(box[:3], boxes[:, :3])
    hi = np.minimum(box[3:], boxes[:, 3:])
    inter = np.prod(np.clip(hi - lo, 0, None), axis=1)
    va = np.prod(box[3:] - box[:3])
    vb = np.prod(boxes[:, 3:] - boxes[:, :3], axis=1)
    return inter / np.maximum(va + vb - inter, 1e-9)


def axis_starts(lo, hi, window, stride):
    """Tile start positions covering [lo, hi] with `window`-sized tiles, `stride`
    apart, GUARANTEED to reach the far end (plain np.arange(lo, hi-window, stride)
    silently drops the tail when (hi-lo-window) isn't a multiple of stride)."""
    if hi - lo <= window:
        return [lo]
    starts = list(np.arange(lo, hi - window + 1e-9, stride))
    last = hi - window
    if not starts or last - starts[-1] > 1e-6:
        starts.append(last)
    return starts


def nms_3d(boxes, scores, iou_thr):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = iou_aabb(boxes[i], boxes[rest])
        order = rest[ious < iou_thr]
    return np.array(keep, dtype=np.int64)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", required=True)
    ap.add_argument("--ply", default=None, help="override input point cloud")
    ap.add_argument("--checkpoint",
                    default=str(REPO / "weights" / "3detr_scannet_masked_ep1080.pth"))
    ap.add_argument("--out", default=None,
                    help="default: pipeline4/out/<space>_p4_geo.json")
    ap.add_argument("--window", type=float, default=9.0,
                    help="tile size (m). 3DETR is trained on whole ScanNet rooms "
                    "(RandomCuboid only ever crops 50-100%% of a room's own extent, "
                    "never a small fixed window) so its position-embedding/size "
                    "priors assume roughly room-scale input. Empirically (see "
                    "pipeline4/README.md) raw objectness recall on this dataset "
                    "peaks around 8-10m and collapses below ~5m — do not shrink "
                    "this for 'denser tiling', it backfires")
    ap.add_argument("--stride", type=float, default=4.5, help="tile stride (m)")
    ap.add_argument("--passes", type=int, default=3,
                    help="random 40k-point subsamples per tile (ensembled by NMS)")
    ap.add_argument("--num-points", type=int, default=40000)
    ap.add_argument("--min-prob", type=float, default=0.10,
                    help="objectness threshold; domain gap to ScanNet training "
                    "classes means novel industrial objects score lower even when "
                    "real, so this is intentionally permissive — downstream CLIP "
                    "labeling + structure filters + NMS prune the rest")
    ap.add_argument("--nms-iou", type=float, default=0.30)
    ap.add_argument("--min-tile-points", type=int, default=8000)
    ap.add_argument("--min-box-points", type=int, default=40)
    ap.add_argument("--max-box-dim", type=float, default=6.0)
    ap.add_argument("--min-box-dim", type=float, default=0.04)
    ap.add_argument("--max-z-m", type=float, default=3.2,
                    help="crop input points above floor+this (ScanNet rooms are <~3 m)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t_start = time.time()
    ply_path = resolve_ply(args.space, args.ply)
    print(f"[p4] loading {ply_path}")
    xyz_w, rgb = load_ply(ply_path)
    print(f"[p4] {len(xyz_w):,} points")

    # y_invert (spaces.json, opt-in per space): some raw point clouds have Y
    # increasing DOWNWARD instead of upward — verified concretely for
    # shinhan_space by rendering the actual camera crop CLIP saw for two
    # objects: the one positioned at the low-Y end was visually a ceiling
    # light fixture, and the one at the high-Y end was a table with chairs
    # on the floor — i.e. this space's low-percentile Y is the true
    # ceiling, not the floor. A camera-trajectory height cross-check on
    # this same space earlier looked consistent with normal Y-up and was
    # WRONG; only the direct visual crop is decisive here, so don't trust
    # that heuristic alone for a new space again.
    #
    # y_sign corrects ONLY the internal 3DETR-input orientation (so the
    # detector sees a properly upright room) — it is deliberately NOT baked
    # into xyz_w or the output bbox_min/max/centroid, which stay in xyz_w's
    # own raw convention. A first version of this fix negated xyz_w
    # directly and broke geo_label_clip.py's view-selection: cameras.json's
    # camera positions are defined against the RAW (un-negated) convention,
    # so an object's negated centroid minus a camera's un-negated position
    # is geometric nonsense, producing garbled/wrong CLIP crops (caught by
    # re-rendering the actual crop and finding it showed an unrelated part
    # of the room). Keeping xyz_w untouched keeps every camera-projection
    # consumer correct; only the scalar `y_up_sign` written to the output
    # geo JSON tells elevation-sensitive downstream consumers (geo_label_
    # clip.py's shape_prior, this repo's floor/ceiling-relative rejection
    # checks) which direction is physically "up" for THIS space specifically.
    space_cfg = space_registry().get(args.space, {})
    y_invert = space_cfg.get("y_invert", False)
    y_sign = -1.0 if y_invert else 1.0
    if y_invert:
        print(f"[p4] y_invert=true for {args.space}: raw Y increases "
              f"downward, not upward — correcting the 3DETR input frame "
              f"only (see module docstring)")

    floor_y = float(np.percentile(xyz_w[:, 1], FLOOR_PCTL))
    ceil_y = float(np.percentile(xyz_w[:, 1], CEIL_PCTL))

    y_eff = xyz_w[:, 1] * y_sign          # internal only: properly "up" always
    floor_z = float(np.percentile(y_eff, FLOOR_PCTL))  # internal floor reference

    mdl = world_to_model(xyz_w, y_sign=y_sign)
    mdl[:, 2] -= floor_z
    in_z = mdl[:, 2] <= args.max_z_m
    print(f"[p4] floor_y={floor_y:.2f} ceil_y={ceil_y:.2f}; "
          f"{100 * (1 - in_z.mean()):.1f}% points above z-crop")

    det = Detr3DDetector(args.checkpoint, device=args.device,
                         num_points=args.num_points)

    # tile grid over model-frame x/y — axis_starts() guarantees the far edge of
    # each axis is always covered by some tile, even when the range doesn't
    # divide evenly by stride.
    xmin, ymin = mdl[:, 0].min(), mdl[:, 1].min()
    xmax, ymax = mdl[:, 0].max(), mdl[:, 1].max()
    xs = axis_starts(xmin, xmax, args.window, args.stride)
    ys = axis_starts(ymin, ymax, args.window, args.stride)
    x_centers = [x0 + args.window / 2 for x0 in xs]
    y_centers = [y0 + args.window / 2 for y0 in ys]

    all_boxes, all_prob, all_cls, all_cls_prob = [], [], [], []
    n_tiles = 0
    for xi, x0 in enumerate(xs):
        for yi, y0 in enumerate(ys):
            m = (in_z
                 & (mdl[:, 0] >= x0) & (mdl[:, 0] < x0 + args.window)
                 & (mdl[:, 1] >= y0) & (mdl[:, 1] < y0 + args.window))
            npts = int(m.sum())
            if npts < args.min_tile_points:
                continue
            sub = mdl[m]
            n_tiles += 1

            # responsibility cell of this tile (Voronoi cell of tile centers,
            # open at the scene boundary) — avoids duplicate/truncated boxes.
            # Uses actual neighbor-center midpoints (not a fixed +-stride/2)
            # since axis_starts() can insert a non-uniformly-spaced last tile.
            cx, cy = x_centers[xi], y_centers[yi]
            x_lo = -np.inf if xi == 0 else (x_centers[xi - 1] + cx) / 2
            x_hi = np.inf if xi == len(xs) - 1 else (x_centers[xi + 1] + cx) / 2
            y_lo = -np.inf if yi == 0 else (y_centers[yi - 1] + cy) / 2
            y_hi = np.inf if yi == len(ys) - 1 else (y_centers[yi + 1] + cy) / 2

            for p in range(args.passes):
                r = det.detect(sub, min_prob=args.min_prob, seed=p)
                if not len(r["prob"]):
                    continue
                ctr = (r["boxes_min"] + r["boxes_max"]) / 2
                resp = ((ctr[:, 0] >= x_lo) & (ctr[:, 0] < x_hi)
                        & (ctr[:, 1] >= y_lo) & (ctr[:, 1] < y_hi))
                dims = r["boxes_max"] - r["boxes_min"]
                ok = (resp
                      & (dims.max(axis=1) <= args.max_box_dim)
                      & (dims.min(axis=1) >= args.min_box_dim))
                all_boxes.append(np.hstack([r["boxes_min"][ok], r["boxes_max"][ok]]))
                all_prob.append(r["prob"][ok])
                all_cls.append(r["sem_cls"][ok])
                all_cls_prob.append(r["sem_prob"][ok])
            print(f"[p4] tile ({xi},{yi}) pts={npts:>8} "
                  f"dets={sum(len(a) for a in all_prob)} (cum)")

    if not all_boxes or not sum(len(b) for b in all_boxes):
        raise SystemExit("[p4] no detections — check thresholds / input cloud")
    boxes = np.concatenate(all_boxes)
    prob = np.concatenate(all_prob)
    cls = np.concatenate(all_cls)
    cls_prob = np.concatenate(all_cls_prob)
    print(f"[p4] {n_tiles} tiles, {len(boxes)} raw detections")

    keep = nms_3d(boxes, prob, args.nms_iou)
    boxes, prob, cls, cls_prob = boxes[keep], prob[keep], cls[keep], cls_prob[keep]
    print(f"[p4] {len(boxes)} after 3D NMS (iou<{args.nms_iou})")

    # NOTE: fragment merging (same physical object split into several boxes
    # by 3DETR's query-based prediction) is NOT done here. 3DETR's own raw
    # class prediction is unreliable under the ScanNet->industrial domain gap
    # (see README) — the majority raw class on this dataset is "chair" for
    # almost everything, so merging on raw class either misses real fragments
    # (mislabeled "chair") or, if "chair" is made eligible, chain-merges
    # scores of genuinely distinct nearby objects into one giant box via
    # transitive union-find. Fragment merging instead runs downstream in
    # pipeline2b/scene_graph.py (find_fragment_groups), AFTER CLIP labeling,
    # using the real semantic label.

    # ---- build stage-A nodes: crop points per box in world frame
    nodes, npz_payload = [], {}
    nid = 0
    for b, pr, c, cp in zip(boxes, prob, cls, cls_prob):
        wmin, wmax = model_box_to_world(b[:3], b[3:], floor_z, y_sign=y_sign)
        pad = 0.02
        inb = np.all((xyz_w >= wmin - pad) & (xyz_w <= wmax + pad), axis=1)
        pts = xyz_w[inb]
        if len(pts) < args.min_box_points:
            continue
        # drop the floor band so the node's own points/bbox don't hug the
        # floor — y_sign-aware: "above the floor" means larger raw Y for a
        # normal space, but SMALLER raw Y for a y_invert space (see above).
        body = (pts[:, 1] - floor_y) * y_sign > 0.05
        node_pts = pts[body] if body.sum() >= args.min_box_points else pts
        node_rgb = None
        if rgb is not None:
            node_rgb = rgb[inb]
            if body.sum() >= args.min_box_points:
                node_rgb = node_rgb[body]

        # tighten bbox to the observed points (but never exceed the detector box)
        lo = np.maximum(np.percentile(node_pts, 0.5, axis=0), wmin)
        hi = np.minimum(np.percentile(node_pts, 99.5, axis=0), wmax)
        mean_rgb = ([int(round(v * 255)) for v in node_rgb.mean(axis=0)]
                    if node_rgb is not None else [128, 128, 128])

        nodes.append({
            "id": nid,
            "label": f"obj_{nid}",
            "centroid": [float(v) for v in node_pts.mean(axis=0)],
            "bbox_min": [float(v) for v in lo],
            "bbox_max": [float(v) for v in hi],
            "bbox_size": [float(v) for v in (hi - lo)],
            "n_points": int(len(node_pts)),
            "mean_rgb": mean_rgb,
            "det_class": SCANNET_CLASSES[int(c)],
            "det_prob": float(pr),
            "det_class_prob": float(cp),
        })
        npz_payload[f"xyz_{nid}"] = node_pts.astype(np.float32)
        if node_rgb is not None:
            npz_payload[f"rgb_{nid}"] = node_rgb.astype(np.float32)
        nid += 1

    # probe subsample of the whole cloud (used by table consolidation etc.)
    n_probe = min(200_000, len(xyz_w))
    probe_idx = np.random.default_rng(0).choice(len(xyz_w), n_probe, replace=False)
    npz_payload["_probe_xyz"] = xyz_w[probe_idx].astype(np.float32)

    out_path = (Path(args.out) if args.out
                else REPO / "pipeline4" / "out" / f"{args.space}_p4_geo.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    geo = {
        "space": args.space,
        "source": "pipeline4/3detr",
        "checkpoint": Path(args.checkpoint).name,
        "nodes": nodes,
        "structure_segments": [],
        "floor_y": floor_y,
        "ceil_y": ceil_y,
        # +1 for a normal space (floor_y < ceil_y AND "up" = increasing raw
        # Y, so plain (value - floor_y) already means "elevation"); -1 for
        # a y_invert space, where floor_y/ceil_y are still plain raw-Y
        # percentiles (floor_y ends up > ceil_y numerically) and any
        # elevation-above-floor computation must flip sign — see
        # world_to_model()'s docstring for why this isn't baked into the
        # coordinates themselves. Downstream consumers (geo_label_clip.py's
        # shape_prior, this repo's floor/ceiling-relative rejection checks)
        # read this to get elevation direction right; absent/1.0 is a no-op
        # for every space that predates this field.
        "y_up_sign": y_sign,
    }
    out_path.write_text(json.dumps(geo, indent=2))
    np.savez_compressed(out_path.with_name(out_path.stem + "_points.npz"),
                        **npz_payload)
    print(f"[p4] wrote {out_path} ({len(nodes)} nodes) "
          f"+ points npz  [{time.time() - t_start:.1f}s]")

    by_cls = {}
    for n in nodes:
        by_cls[n["det_class"]] = by_cls.get(n["det_class"], 0) + 1
    print("[p4] class counts:", dict(sorted(by_cls.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
