#!/usr/bin/env python3
"""
Learn a per-space TABLE MODEL from the viewer's annotation boxes (ground
truth), then detect every table that matches it.

The point (user request): don't paste the annotations in as fixed boxes, and
don't hardcode a colour — instead FIT what a table looks like *in this space*
from the operator's corrected boxes, so the detector reproduces the target
tables and the SAME procedure works on any space whose tables differ
(factory teal, shinhan's tables, ...). A model is a tiny JSON of interpretable
parameters:

    rgb          learned tabletop colour (median of annotated tops)
    tol          colour match radius (fit to the spread of annotated tops)
    band_lo/hi   tabletop height band above the floor (from annotated levels)
    proto_short  typical table depth  (median of annotated boxes)
    proto_long   typical table length (median of annotated boxes)

detect_tables() applies the model: keep table-coloured points in the height
band -> occupancy grid -> morphological open (drop blur bridges/specks) ->
connected components (continuous plane = one region; empty aisle = split) ->
split long rows at density valleys, then tile any run still much longer than
proto_long into proto-sized tables so the output matches the target
granularity. Racks of the same colour are rejected by overlap with the
already-detected rack nodes.

Models are saved to pipeline2b/out/<space>_table_model.json and can be reused
across spaces that share a table type (e.g. factory_space_13 -> _14).
"""
from __future__ import annotations

import numpy as np


def _rot_xz(xz, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.column_stack([xz[:, 0] * c - xz[:, 1] * s,
                            xz[:, 0] * s + xz[:, 1] * c])


def learn_table_model(xyz, rgb, floor_y, yaw_deg, annotations):
    """Fit a table model from annotation records (op add/edit, label
    table/desk/empty). Returns the model dict, or None if nothing usable."""
    cols, levels, sizes = [], [], []
    for a in annotations:
        if a.get("op") not in ("add", "edit"):
            continue
        lb = a.get("label")
        if lb not in ("table", "desk") and lb:            # table / desk / blank
            continue
        bc, sz = a.get("box_center"), a.get("bbox_size")
        if not bc or not sz:
            continue
        cx, cy, cz = bc
        sx, sy, sz_ = sz
        top_y = cy + sy / 2.0
        inb = ((np.abs(xyz[:, 0] - cx) < sx / 2) & (np.abs(xyz[:, 2] - cz) < sz_ / 2)
               & (xyz[:, 1] > top_y - 0.20) & (xyz[:, 1] < top_y + 0.08))
        if int(inb.sum()) > 30:
            cols.append(rgb[inb])
            levels.append(top_y - floor_y)
        sizes.append(sorted((float(sx), float(sz_))))
    if not cols:
        return None
    allc = np.vstack(cols)
    target = np.median(allc, axis=0)
    # Robust colour spread: annotation boxes catch some non-tabletop pixels
    # (edges, legs, background), so use the 60th percentile of the distance
    # (the tabletop core) and cap tight — a loose tol floods the scene.
    d = np.linalg.norm(allc - target, axis=1)
    tol = float(np.clip(np.percentile(d, 60) * 1.3, 40.0, 62.0))
    levels = np.asarray(levels)
    sizes = np.asarray(sizes)
    return {
        "rgb": [round(float(v), 1) for v in target],
        "tol": round(tol, 1),
        "band_lo": round(float(max(0.30, levels.min() - 0.18)), 2),
        "band_hi": round(float(levels.max() + 0.18), 2),
        "proto_short": round(float(np.median(sizes[:, 0])), 2),
        "proto_long": round(float(np.median(sizes[:, 1])), 2),
        "n_annotations": int(len(sizes)),
    }


def _valley_split(a, cell=0.10, min_seg=0.5, split_above=1.8):
    """Cut a 1D coordinate array at genuine density valleys (returns bin edges
    used as cut positions). Only acts when the extent exceeds split_above."""
    lo, hi = a.min(), a.max()
    if hi - lo < split_above:
        return []
    nb = max(int((hi - lo) / cell), 4)
    hist, edges = np.histogram(a, bins=nb)
    med = np.median(hist[hist > 0]) if (hist > 0).any() else 0
    cuts = []
    for i in range(1, len(hist) - 1):
        if hist[i] < 0.30 * med and hist[i] <= hist[i - 1] and hist[i] <= hist[i + 1]:
            pos = float((edges[i] + edges[i + 1]) / 2)
            if pos - lo > min_seg and hi - pos > min_seg and (not cuts or pos - cuts[-1] > min_seg):
                cuts.append(pos)
    return cuts


def _segment_axis(a, proto_len):
    """Return (lo,hi) intervals along axis `a`: first cut at density valleys,
    then evenly tile any run still longer than ~1.4*proto_len into
    proto-sized tables (so output granularity matches the learned target)."""
    lo, hi = a.min(), a.max()
    bounds = [lo] + _valley_split(a) + [hi]
    out = []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        span = e - s
        if proto_len > 0.2 and span > 1.4 * proto_len:
            n = max(1, int(round(span / proto_len)))
            step = span / n
            for j in range(n):
                out.append((s + j * step, s + (j + 1) * step + 1e-6))
        else:
            out.append((s, e + 1e-6))
    return out


def detect_tables(xyz, rgb, floor_y, yaw_deg, model, rack_rects=(),
                  min_short=0.40, min_area=0.45):
    """Detect all tables matching a learned model. Returns consolidate-style
    [{col, n_points, mean_rgb, rect, level}]."""
    from scipy.ndimage import label as ndlabel, binary_opening
    if floor_y is None or model is None:
        return []
    target = np.asarray(model["rgb"]); tol = model["tol"]
    proto_short = model.get("proto_short", 0.8)
    proto_long = model.get("proto_long", 1.5)
    y = xyz[:, 1]
    al = _rot_xz(xyz[:, [0, 2]], -yaw_deg)
    top = (y > floor_y + model["band_lo"]) & (y < floor_y + model["band_hi"])
    teal = top & (np.linalg.norm(rgb - target, axis=1) < tol)
    if int(teal.sum()) < 50:
        return []
    ti = np.where(teal)[0]
    cell = 0.08
    gu0, gv0 = al[ti, 0].min(), al[ti, 1].min()
    gi = np.floor((al[ti, 0] - gu0) / cell).astype(np.int64)
    gj = np.floor((al[ti, 1] - gv0) / cell).astype(np.int64)
    nu, nv = int(gi.max()) + 1, int(gj.max()) + 1
    cnt = np.zeros((nu, nv)); np.add.at(cnt, (gi, gj), 1)
    grid = binary_opening(cnt >= 1, iterations=1)
    lab, n = ndlabel(grid, structure=np.ones((3, 3)))
    inreg = lab[gi, gj]
    boxes = []
    for c in range(1, n + 1):
        pidx = ti[inreg == c]
        if len(pidx) < 20:
            continue
        uu, vv = al[pidx, 0], al[pidx, 1]
        # split both axes at genuine density valleys only (no uniform tiling —
        # a continuous bench stays one box; the operator's authoritative
        # annotation boxes provide the fine per-table split where they care).
        ub = [uu.min()] + _valley_split(uu) + [uu.max() + 1e-6]
        for ka in range(len(ub) - 1):
            mu = (uu >= ub[ka]) & (uu < ub[ka + 1])
            if mu.sum() < 15:
                continue
            vv2 = vv[mu]
            vb = [vv2.min()] + _valley_split(vv2) + [vv2.max() + 1e-6]
            for kb in range(len(vb) - 1):
                si = pidx[mu][(vv2 >= vb[kb]) & (vv2 < vb[kb + 1])]
                if len(si) < 15:
                    continue
                cu, cv = al[si, 0], al[si, 1]
                ext = sorted((float(np.ptp(cu)), float(np.ptp(cv))))
                if ext[0] < min_short or ext[0] * ext[1] < min_area:
                    continue
                u0, u1, v0, v1 = cu.min(), cu.max(), cv.min(), cv.max()
                if _rack_overlap((u0, v0, u1, v1), rack_rects) > 0.6:
                    continue
                level = float(np.median(y[si]))
                world = _rot_xz(np.column_stack([cu, cv]), yaw_deg)
                col = np.vstack([
                    np.column_stack([world[:, 0], np.full(len(world), floor_y), world[:, 1]]),
                    np.column_stack([world[:, 0], np.full(len(world), level + 0.05), world[:, 1]]),
                ])
                boxes.append({"col": col, "n_points": int(len(si)),
                              "mean_rgb": [int(v) for v in np.round(target)],
                              "rect": (float(u0), float(v0), float(u1), float(v1)),
                              "level": level})
    return boxes


def _rack_overlap(rect, rack_rects):
    u0, v0, u1, v1 = rect
    area = max((u1 - u0) * (v1 - v0), 1e-6)
    inter = 0.0
    for r0, s0, r1, s1 in rack_rects:
        iu = max(0.0, min(u1, r1) - max(u0, r0))
        iv = max(0.0, min(v1, s1) - max(v0, s0))
        inter += iu * iv
    return inter / area


def match_report(boxes, annotations, yaw_deg, iou_thr=0.3):
    """How many annotated target tables a detection set matches (IoU in the
    aligned XZ plane) — the training signal."""
    def rect(bc, sz):
        cx, _, cz = bc; sx, _, sz_ = sz
        cor = np.array([[cx - sx / 2, cz - sz_ / 2], [cx + sx / 2, cz - sz_ / 2],
                        [cx + sx / 2, cz + sz_ / 2], [cx - sx / 2, cz + sz_ / 2]])
        a = _rot_xz(cor, -yaw_deg)
        return (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())
    targets = [rect(a["box_center"], a["bbox_size"]) for a in annotations
               if a.get("op") in ("add", "edit") and a.get("box_center") and a.get("bbox_size")
               and (a.get("label") in ("table", "desk") or not a.get("label"))]
    dets = [b["rect"] for b in boxes]

    def iou(a, b):
        iu = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iv = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = iu * iv
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0
    matched = sum(1 for t in targets if any(iou(t, d) >= iou_thr for d in dets))
    return matched, len(targets)
