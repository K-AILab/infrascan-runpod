#!/usr/bin/env python
"""Combine a splat-derived scene graph (already re-projected onto a space's
real point cloud - see apply_scenegraph_to_pointcloud.py) with that same
space's pipeline4 (3DETR) scene graph, adding only objects pipeline4 didn't
already find, and only after confirming each candidate actually has real
geometry in its box (not empty space).

Both graphs are already in the SAME frame (the real point_cloud.ply) once
the splat one has been re-projected, so no further alignment is needed here
- this is a spatial DEDUP + MERGE, not another registration problem.

Dedup rule (geometry-first, not a hand-curated label-synonym table - the
two pipelines' label vocabularies genuinely differ, e.g. "chair" vs
"office_chair", "cardboard box" vs "cardboard_box", and a synonym table
would need constant upkeep):
  - same normalized label (case/space/underscore-insensitive) AND
    centroid within DEDUP_CENTROID_M of an existing pipeline4 node, OR
  - large XZ footprint overlap (> DEDUP_OVERLAP_FRAC of the smaller box)
    AND comparable size (ratio < DEDUP_SIZE_RATIO) even across different
    labels - catches the same real object where the two pipelines
    disagree on what to call it, while a small object legitimately
    contained inside a much bigger one (a chair under a table) has a high
    overlap fraction but a large size ratio and is correctly kept as two
    separate objects.

Anything that survives dedup is a CANDIDATE new object - before adding it,
verify real point-cloud density inside its box (own standing instruction:
never add a box without confirming it actually bounds something, not empty
space).

Edges are recomputed with pipeline2b's own build_edges() on the combined
object set (not carried over from either source graph) - see
scene-graph-edges-reuse memory: an empty/stale edge list is an incomplete
deliverable, and edges between an old and a newly-added node don't exist
anywhere to carry over in the first place.

Optionally filters pipeline4's own nodes by 3DETR's own objectness
confidence (`det_prob` in its `<space>_p4_geo.json`, matched to each final
node by nearest centroid - absorption/merging between that raw stage and
the final scene_graph.json means ids don't line up 1:1) before merging.
This was added after direct user feedback that the merged result looked
"too messy... too many false positives" and asked to keep only very
confident pipeline4 detections. Checked against the data first, not
assumed: the FINAL label (e.g. "shelf") often differs from 3DETR's own
raw `det_class` (CLIP can override it), and per-final-label confidence
does NOT cleanly split into "chair/table reliable, everything else isn't"
- office_chair stands out clearly (median det_prob 0.74), but table
(0.30) and shelf (0.32) are unremarkable, in line with the same range as
person/cabinet/machine/pallet. A single continuous det_prob threshold
(not a hardcoded category allowlist) is the honest general fix - it just
happens to keep mostly office_chair/shelf/table at a reasonably high
threshold, without assuming any specific label name is trustworthy.

Usage:
  python pipeline9/merge_splat_with_p4.py \
    --splat-scene-graph ui/_spaces/factory13_pointcloud_scenegraph/scene_graph.json \
    --p4-scene-graph ui/_spaces/factory_space_13_p4/scene_graph.json \
    --p4-geo-json pipeline4/out/factory_space_13_p4_geo.json \
    --min-det-prob 0.6 \
    --pointcloud-ply data/factory_space_13/pointcloud.ply \
    --space factory_space_13_p4_plus_splat \
    --out ui/_spaces/factory_space_13_p4_plus_splat/scene_graph.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline2b"))
import scene_graph as sg  # noqa: E402

DEDUP_CENTROID_M = 0.6
DEDUP_OVERLAP_FRAC = 0.6
DEDUP_SIZE_RATIO = 1.8
MIN_OCCUPANCY_PTS = 40
MIN_OCCUPANCY_DENSITY = 200.0  # points / m^3 - real furniture is dense; empty space isn't


def norm_label(label):
    return label.lower().replace(" ", "_")


def aabb(node):
    c = np.array(node["centroid"], dtype=np.float64)
    s = np.array(node["bbox_size"], dtype=np.float64)
    return c - s / 2, c + s / 2


def footprint_area(node):
    s = node["bbox_size"]
    return max(s[0], 1e-6) * max(s[2], 1e-6)


def xz_overlap_frac(a, b):
    amin, amax = aabb(a)
    bmin, bmax = aabb(b)
    ox = max(0.0, min(amax[0], bmax[0]) - max(amin[0], bmin[0]))
    oz = max(0.0, min(amax[2], bmax[2]) - max(amin[2], bmin[2]))
    inter = ox * oz
    return inter / max(min(footprint_area(a), footprint_area(b)), 1e-6)


def is_duplicate(cand, p4_nodes, p4_tree, p4_centroids):
    d, idx = p4_tree.query(cand["centroid"])
    nearest = p4_nodes[idx]
    if norm_label(cand["label"]) == norm_label(nearest["label"]) and d < DEDUP_CENTROID_M:
        return True, nearest, d
    # cross-label check needs the true best-overlap match, not just nearest-centroid
    best_ov, best_node = 0.0, nearest
    for p in p4_nodes:
        if np.linalg.norm(np.array(p["centroid"]) - np.array(cand["centroid"])) > 3.0:
            continue  # cheap prefilter before the O(n) overlap check
        ov = xz_overlap_frac(cand, p)
        if ov > best_ov:
            best_ov, best_node = ov, p
    if best_ov > DEDUP_OVERLAP_FRAC:
        ratio = max(footprint_area(cand), footprint_area(best_node)) / \
            max(min(footprint_area(cand), footprint_area(best_node)), 1e-6)
        if ratio < DEDUP_SIZE_RATIO:
            return True, best_node, d
    return False, nearest, d


MIN_FOOTPRINT_FILL = 0.5


def verify_occupancy(node, xyz_tree, xyz):
    cmin, cmax = aabb(node)
    lo, hi = cmin - 0.05, cmax + 0.05
    idx = xyz_tree.query_ball_point(node["centroid"], r=float(np.linalg.norm(hi - lo)) / 2 + 0.1)
    if not idx:
        return False, 0
    pts = xyz[idx]
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    n_inside = int(inside.sum())
    vol = max((hi - lo).prod(), 1e-6)
    density = n_inside / vol
    if n_inside < MIN_OCCUPANCY_PTS or density < MIN_OCCUPANCY_DENSITY:
        return False, n_inside
    # A box can pass the volume-averaged density check while still mostly
    # covering empty space - a small dense cluster inside an oversized
    # footprint inflates the average. One box measured 3185 pts/m^3 by
    # volume, comfortably over threshold, while only 36% of its own XZ
    # footprint held any point at any height - a box drawn over mostly empty
    # ground. So footprint fill is checked directly: does any point, at any
    # height, fall in each cell of the box's XZ footprint rasterized at 5cm.
    res = 0.05
    x0, x1 = cmin[0], cmax[0]
    z0, z1 = cmin[2], cmax[2]
    W = max(int((x1 - x0) / res) + 1, 1)
    H = max(int((z1 - z0) / res) + 1, 1)
    col_m = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 2] >= z0) & (pts[:, 2] <= z1)
    col_pts = pts[col_m]
    if len(col_pts) == 0:
        return False, n_inside
    gx = np.clip(((col_pts[:, 0] - x0) / res).astype(int), 0, W - 1)
    gz = np.clip(((col_pts[:, 2] - z0) / res).astype(int), 0, H - 1)
    grid = np.zeros((H, W), dtype=bool)
    grid[gz, gx] = True
    fill = grid.sum() / (W * H)
    return fill >= MIN_FOOTPRINT_FILL, n_inside


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splat-scene-graph", required=True)
    ap.add_argument("--p4-scene-graph", required=True)
    ap.add_argument("--p4-geo-json", default=None,
                     help="pipeline4's own <space>_p4_geo.json, carrying 3DETR's raw per-object "
                          "det_prob (objectness confidence) - matched to final nodes by nearest "
                          "centroid. Required if --min-det-prob is used.")
    ap.add_argument("--min-det-prob", type=float, default=0.0,
                     help="drop pipeline4 nodes below this 3DETR objectness confidence before "
                          "merging - added after direct feedback that unfiltered pipeline4 "
                          "output looked too noisy. A continuous threshold, not a category "
                          "allowlist: checked against the data first (see module docstring) - "
                          "confidence does not cleanly split by final label.")
    ap.add_argument("--pointcloud-ply", required=True)
    ap.add_argument("--space", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    splat_sg = json.loads(Path(args.splat_scene_graph).read_text())
    p4_sg = json.loads(Path(args.p4_scene_graph).read_text())
    p4_nodes = [n for n in p4_sg["nodes"] if not n["label"].startswith("removed_")]
    print(f"[merge] p4: {len(p4_nodes)} real nodes ({len(p4_sg['nodes']) - len(p4_nodes)} "
          f"already-removed dropped)")

    if args.min_det_prob > 0:
        if not args.p4_geo_json:
            raise SystemExit("--min-det-prob needs --p4-geo-json")
        geo = json.loads(Path(args.p4_geo_json).read_text())
        geo_cent = np.array([g["centroid"] for g in geo["nodes"]])
        geo_prob = np.array([g["det_prob"] for g in geo["nodes"]])
        geo_tree = cKDTree(geo_cent)
        kept = []
        for n in p4_nodes:
            _, idx = geo_tree.query(n["centroid"])
            if geo_prob[idx] >= args.min_det_prob:
                kept.append(n)
        print(f"[merge] det_prob>={args.min_det_prob}: keeping {len(kept)}/{len(p4_nodes)} "
              f"pipeline4 nodes ({len(p4_nodes)-len(kept)} low-confidence dropped)")
        p4_nodes = kept

    p = PlyData.read(args.pointcloud_ply)["vertex"]
    xyz = np.column_stack([p["x"], p["y"], p["z"]]).astype(np.float64)
    xyz_tree = cKDTree(xyz)

    # A real, single piece of furniture spanning most of the room's own
    # floor-to-ceiling height is essentially never real - it's a mislabeled
    # structural element (a door/window/curtain that 3DETR/CLIP's own
    # det_class correctly called structural but a downstream CLIP pass
    # overrode with a furniture label like "cabinet"/"shelf", so it never
    # got dropped as structure). In shinhan's pipeline4 output, several raw
    # det_class=window/door/curtain nodes 2.6-2.84 m tall in a 3.1 m room
    # survived into the final graph relabelled as "cabinet"/"shelf". The
    # test is purely geometric and independent of label, so it generalises.
    room_height_m = float(np.percentile(xyz[:, 1], 99) - np.percentile(xyz[:, 1], 1))
    MAX_FURNITURE_HEIGHT_FRAC = 0.75
    max_h = room_height_m * MAX_FURNITURE_HEIGHT_FRAC
    n_before = len(p4_nodes)
    tall_rejected = [n for n in p4_nodes if n["bbox_size"][1] > max_h]
    p4_nodes = [n for n in p4_nodes if n["bbox_size"][1] <= max_h]
    if tall_rejected:
        print(f"[merge] room height ~{room_height_m:.2f}m -> rejecting {len(tall_rejected)}/"
              f"{n_before} pipeline4 nodes taller than {max_h:.2f}m "
              f"({MAX_FURNITURE_HEIGHT_FRAC*100:.0f}% of room height, near-floor-to-ceiling - "
              f"almost certainly a mislabeled structural element):")
        for n in tall_rejected:
            print(f"[merge]   {n['label']:16s} height={n['bbox_size'][1]:.2f}m "
                  f"centroid={[round(x,2) for x in n['centroid']]}")

    print(f"[merge] splat (point-cloud frame): {len(splat_sg['nodes'])} nodes")

    p4_centroids = np.array([n["centroid"] for n in p4_nodes])
    p4_tree = cKDTree(p4_centroids)

    accepted, rejected_dup, rejected_empty = [], [], []
    for n in splat_sg["nodes"]:
        dup, match, d = is_duplicate(n, p4_nodes, p4_tree, p4_centroids)
        if dup:
            rejected_dup.append((n, match, d))
            continue
        ok, n_inside = verify_occupancy(n, xyz_tree, xyz)
        if not ok:
            rejected_empty.append((n, n_inside))
            continue
        accepted.append(n)

    print(f"[merge] {len(accepted)} new objects accepted (not already in p4, real occupancy "
          f"confirmed)")
    print(f"[merge] {len(rejected_dup)} rejected as duplicates of an existing p4 object")
    print(f"[merge] {len(rejected_empty)} rejected for empty/too-sparse occupancy:")
    for n, n_inside in rejected_empty:
        print(f"[merge]   {n['label']:16s} centroid={n['centroid']} n_pts_inside={n_inside}")

    # ---- assign room_id/area_id to new nodes via nearest p4 node ----
    p4_labels_normalized = {norm_label(n["label"]) for n in p4_nodes}
    for n in accepted:
        _, idx = p4_tree.query(n["centroid"])
        n["room_id"] = p4_nodes[idx].get("room_id", 0)
        n["area_id"] = p4_nodes[idx].get("area_id", -1)
        # cosmetic only: if p4 already spells this same category differently
        # (e.g. "cardboard_box" vs "cardboard box"), adopt p4's spelling so
        # the combined graph doesn't render the same real category as two
        # different legend colors/entries.
        nl = norm_label(n["label"])
        if nl in p4_labels_normalized:
            match = next(pn["label"] for pn in p4_nodes if norm_label(pn["label"]) == nl)
            n["label"] = match

    # ---- combined objects dict for build_edges (renumber ids) ----
    combined_nodes = []
    next_id = max((n["id"] for n in p4_nodes), default=-1) + 1
    for n in p4_nodes:
        combined_nodes.append(dict(n))
    for n in accepted:
        n2 = dict(n)
        n2["id"] = next_id
        n2["uid"] = f"splat_merge_{next_id}"
        next_id += 1
        combined_nodes.append(n2)

    # The SPLAT graph's yaw is authoritative, not pipeline4's own - it comes
    # from apply_scenegraph_to_pointcloud.py's direct wall-boundary
    # measurement (min-area-rect over the real point cloud), whereas
    # pipeline4's own building_yaw_deg can be a stale/never-measured 0.0
    # (confirmed directly: factory_space_15_p4's own scene graph has
    # building_yaw_deg=0.0, while the real room measures ~5.4-5.9deg -
    # blindly inheriting p4's value here silently un-rotated every
    # splat-derived box while the real room stayed tilted).
    yaw_deg = splat_sg.get("building_yaw_deg", p4_sg.get("building_yaw_deg", 0.0))
    print(f"[merge] using building_yaw_deg={yaw_deg:.3f} (from the splat graph's own wall "
          f"measurement, not pipeline4's uncalibrated {p4_sg.get('building_yaw_deg', 0.0)})")
    objects = {}
    for n in combined_nodes:
        c = np.array(n["centroid"], dtype=np.float64)
        s = np.array(n["bbox_size"], dtype=np.float64)
        cu, cv = sg._rotate_xz_deg(np.array([[c[0], c[2]]]), -yaw_deg)[0]
        cy = c[1]
        half = s / 2
        obj = dict(n)
        obj["bbox_min"] = [cu - half[0], cy - half[1], cv - half[2]]
        obj["bbox_max"] = [cu + half[0], cy + half[1], cv + half[2]]
        objects[n["id"]] = obj

    print(f"[merge] {len(objects)} combined objects -> building_edges(yaw_deg={yaw_deg:.2f})")
    edges = sg.build_edges(
        objects,
        above_delta=sg.ABOVE_DELTA_M, hanging_delta=sg.HANGING_DELTA_M,
        max_direct_gap=sg.MAX_DIRECT_GAP_M, max_hang_gap=sg.MAX_HANG_GAP_M,
        footprint_iou_thr=sg.FOOTPRINT_IOU_THR, on_floor_m=sg.ON_FLOOR_M,
        wall_blocker=None, yaw_deg=yaw_deg)
    print(f"[merge] {len(edges)} edges computed")

    # ---- final graph: reuse p4's own hierarchy/rooms/areas as the base
    # (same physical rooms either way - not re-segmenting), append the new
    # nodes and their hierarchy_edges, replace nodes/edges wholesale. ----
    out_sg = dict(p4_sg)
    out_sg["space"] = args.space
    out_sg["building_yaw_deg"] = yaw_deg
    out_sg["nodes"] = combined_nodes
    out_sg["edges"] = edges
    existing_ids = {n["id"] for n in p4_nodes}
    new_hier_edges = [
        {"src": f"A{n['area_id']}" if n.get("area_id", -1) >= 0 else f"R{n.get('room_id', 0)}",
         "dst": n["id"], "kind": "contains"}
        for n in combined_nodes if n["id"] not in existing_ids
    ]
    out_sg["hierarchy_edges"] = p4_sg.get("hierarchy_edges", []) + new_hier_edges
    out_sg["_merged_from"] = {
        "p4_scene_graph": str(args.p4_scene_graph),
        "splat_scene_graph": str(args.splat_scene_graph),
        "n_p4_nodes": len(p4_nodes), "n_splat_new_added": len(accepted),
        "n_splat_rejected_duplicate": len(rejected_dup),
        "n_splat_rejected_empty": len(rejected_empty),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_sg, indent=2))
    print(f"[merge] {len(combined_nodes)} total nodes -> {args.out}")


if __name__ == "__main__":
    main()
