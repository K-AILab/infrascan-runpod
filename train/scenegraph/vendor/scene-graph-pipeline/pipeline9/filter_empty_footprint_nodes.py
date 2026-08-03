#!/usr/bin/env python
"""pipeline9: drop scene-graph nodes whose own footprint has zero/near-zero
real point-cloud coverage.

Why: merge_splat_with_p4.py's verify_occupancy() only ever ran on NEWLY
ADDED pipeline4 nodes during the dedup merge - the original splat-derived
nodes (transformed onto the real point cloud by apply_scenegraph_to_pointcloud.py)
were only checked against the point cloud's OVERALL bounding box (+0.3m
margin), never against real LOCAL point density. That whole-scene check is
too coarse for a large multi-bay/multi-room building: big empty gaps
between bays/aisles are still comfortably inside the building's overall
bbox, so a node whose transformed position drifted into such a gap (e.g.
due to imperfect ICP alignment fitness) sails through - and then renders
as a box floating in visibly empty space. Confirmed directly on
factory14_pointcloud_scenegraph: several nodes had ZERO real points
anywhere near their footprint (not just below density threshold - zero
candidates within the query radius at all).

This reuses the exact same verify_occupancy() logic already used for
pipeline4 nodes (volume-density gate + footprint-fill gate), just applied
to every node in the graph, then rebuilds edges the same way
merge_splat_with_p4.py does.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from merge_splat_with_p4 import aabb, verify_occupancy  # noqa: E402
from pipeline2b import scene_graph as sg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-graph-json", required=True,
                    help="served scene_graph.json (point-cloud frame)")
    ap.add_argument("--pointcloud-ply", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sgraph = json.loads(Path(args.scene_graph_json).read_text())
    nodes = sgraph.get("nodes", sgraph.get("objects"))
    print(f"[filter] {len(nodes)} nodes in")

    p = PlyData.read(args.pointcloud_ply)["vertex"]
    xyz = np.stack([p["x"], p["y"], p["z"]], axis=1).astype(np.float64)
    tree = cKDTree(xyz)

    kept, dropped = [], []
    for n in nodes:
        ok, n_inside = verify_occupancy(n, tree, xyz)
        if ok:
            kept.append(n)
        else:
            dropped.append((n.get("label"), n.get("centroid"), n_inside))

    print(f"[filter] dropped {len(dropped)} empty-footprint nodes:")
    for label, c, n_inside in dropped:
        print(f"    {label:20s} center={c} n_inside={n_inside}")
    print(f"[filter] kept {len(kept)} nodes")

    yaw_deg = sgraph.get("building_yaw_deg", 0.0)
    objects = {}
    for n in kept:
        c = np.array(n["centroid"], dtype=np.float64)
        s = np.array(n["bbox_size"], dtype=np.float64)
        cu, cv = sg._rotate_xz_deg(np.array([[c[0], c[2]]]), -yaw_deg)[0]
        cy = c[1]
        half = s / 2
        obj = dict(n)
        obj["bbox_min"] = [cu - half[0], cy - half[1], cv - half[2]]
        obj["bbox_max"] = [cu + half[0], cy + half[1], cv + half[2]]
        objects[n["id"]] = obj

    edges = sg.build_edges(
        objects,
        above_delta=sg.ABOVE_DELTA_M, hanging_delta=sg.HANGING_DELTA_M,
        max_direct_gap=sg.MAX_DIRECT_GAP_M, max_hang_gap=sg.MAX_HANG_GAP_M,
        footprint_iou_thr=sg.FOOTPRINT_IOU_THR, on_floor_m=sg.ON_FLOOR_M,
        wall_blocker=None, yaw_deg=yaw_deg)
    print(f"[filter] {len(edges)} edges recomputed")

    out_sg = dict(sgraph)
    out_sg["nodes"] = kept
    out_sg["edges"] = edges
    Path(args.out).write_text(json.dumps(out_sg))
    print("->", args.out)


if __name__ == "__main__":
    main()
