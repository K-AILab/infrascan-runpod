#!/usr/bin/env python
"""Convert a splat_analyzer viewer-format boxes.json (native splat-frame
center/size/angle, as produced by rotate_and_export.py / closed_surface_flux.py
/ refit_box_extent.py) into a scene_graph.json for `ui/viewer`'s point-cloud
scene-graph viewer (`sg_3d_viewer.html`) — the OTHER viewer in this repo,
used for pipeline2/pipeline9's own from-scratch geometric detections, as
opposed to tri-viewer which renders directly on the Gaussian Splat.

Coordinate transform: this project's "geo" pipeline (pipeline2, and
pipeline9/splat_to_pointcloud.py) works in a Y-up, real-world-METERS frame
derived from the splat's own native (Z-up) frame by
`geo_xyz = splat_xyz[:, SPLAT_AXIS_MAP] * SCALE_TO_METERS`
(SPLAT_AXIS_MAP = (0,2,1): world_x=splat_x, world_y=splat_z, world_z=splat_y;
SCALE_TO_METERS=4.94) — see splat_to_pointcloud.py's own module docstring
and pipeline9/export_boxes_for_splat_viewer.py, which performs the exact
inverse of this same transform. Reused here unchanged rather than
re-derived, since getting the axis map/scale wrong silently produces a
garbled scene with no error.

sg_3d_viewer.html only actually renders each node's `centroid` (as a
point/label sprite, not a wireframe box) plus `label`/`id`/`room_id`/
`n_world_pts`, so this export focuses on getting those right; box_center/
bbox_size/class_hierarchy/attributes are filled in for schema parity with
other consumers (fastsam/sg_explorer) but aren't load-bearing here.

Usage:
  python export_scene_graph_for_point_viewer.py \
    --ply ../data/shinhan_hires_30k.ply \
    --boxes out/shinhan_space_splatanalyzer_derot_v21_boxes.json \
    --space shinhan_space_splatanalyzer \
    --out ../ui/_spaces/shinhan_space_splatanalyzer/scene_graph.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closed_surface_flux import enclosed_mask, load_splat_raw  # noqa: E402
from splat_to_pointcloud import SPLAT_AXIS_MAP  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline2b"))
import scene_graph as sg2b  # noqa: E402 — reused for its build_edges(): the same
# 3DSSG-style support/proximity/comparative relation logic pipeline2b's own
# from-scratch geometric pipeline uses, applied here to our nodes instead of
# re-deriving relationship rules from scratch.

CLASS_HIERARCHY = {
    "chair":  ["chair", "seat", "furniture", "artifact", "entity"],
    "table":  ["table", "surface", "furniture", "artifact", "entity"],
    "light":  ["light", "fixture", "artifact", "entity"],
    "window": ["window", "opening", "architectural", "entity"],
    "plant":  ["plant", "object", "entity"],
    "door":   ["door", "opening", "architectural", "entity"],
}


def to_geo_frame(vec_splat, scale_to_meters):
    """geo_xyz = splat_xyz[SPLAT_AXIS_MAP] * scale_to_meters — see module docstring."""
    v = np.asarray(vec_splat, dtype=np.float64)
    return (v[list(SPLAT_AXIS_MAP)] * scale_to_meters).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ply", required=True, help="original (non-derotated) splat — "
                    "boxes.json is already in that frame")
    ap.add_argument("--yaw-deg", type=float, required=True,
                     help="THIS SPLAT's room yaw — re-derive per splat, don't reuse another "
                          "space's value (no default: a silently-wrong yaw from the wrong space "
                          "is worse than a missing-argument error)")
    ap.add_argument("--scale-to-meters", type=float, required=True,
                     help="THIS SPLAT's native-units-to-meters factor — re-derive per splat "
                          "(no default, same reasoning as --yaw-deg)")
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--space", required=True, help="space name, e.g. shinhan_space_splatanalyzer")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    xyz, _scale, _normals, _area = load_splat_raw(args.ply)
    data = json.loads(Path(args.boxes).read_text())

    nodes = []
    for i, b in enumerate(data["boxes"]):
        idx, _local = enclosed_mask(xyz, b["center"], b["size"], b.get("angle", 0.0))
        centroid = to_geo_frame(b["center"], args.scale_to_meters)
        size = to_geo_frame(b["size"], args.scale_to_meters)  # magnitudes only — permutation still applies, sign is moot
        size = [abs(v) for v in size]
        label = b["label"]
        nodes.append({
            "id": i,
            "uid": f"splatanalyzer_{i:04d}",
            "group_id": -1,
            "label": label,
            "caption": "",
            "label_score": 0.0,
            "label_entropy": 0.0,
            "centroid": [round(v, 4) for v in centroid],
            "box_center": [round(v, 4) for v in centroid],
            "bbox_size": [round(v, 4) for v in size],
            "room_id": 0,
            "area_id": -1,
            "n_proposals": int(len(idx)),
            "n_world_pts": int(len(idx)),
            "on_floor": label not in ("light", "window"),
            "n_absorbed": 0,
            "absorbed_ids": [],
            "class_hierarchy": CLASS_HIERARCHY.get(label, [label, "entity"]),
            "attributes": {},
            "affordances": [],
        })

    all_centroids = np.array([n["centroid"] for n in nodes])
    # Guard against a sparse/empty scene: with 0 nodes all_centroids is 1-D
    # (shape (0,)), so all_centroids[:, 0] raises IndexError. Fall back to origin.
    if all_centroids.ndim == 2 and len(all_centroids):
        centroid_xz = [round(float(all_centroids[:, 0].mean()), 4),
                       round(float(all_centroids[:, 2].mean()), 4)]
    else:
        centroid_xz = [0.0, 0.0]
    room = {
        "id": 0,
        "centroid_xz": centroid_xz,
        "n_pts": len(nodes),
        "area_m2": None,
    }

    # ── 3DSSG-style relation edges (support / proximity / comparative) ──────
    # pipeline2b.scene_graph.build_edges() needs a plain {id: {..}} dict, not
    # our node-list schema — id, centroid, bbox_size, label, volume, room_id,
    # area_id are all it actually reads (everything else defaults gracefully,
    # confirmed by reading its own field access — see module docstring).
    objects_for_edges = {
        n["id"]: {
            "centroid": n["centroid"],
            "bbox_size": n["bbox_size"],
            "label": n["label"],
            "volume": float(np.prod(n["bbox_size"])),
            "room_id": n["room_id"],
            "area_id": n["area_id"],
        }
        for n in nodes
    }
    edges = sg2b.build_edges(
        objects_for_edges,
        above_delta=sg2b.ABOVE_DELTA_M, hanging_delta=sg2b.HANGING_DELTA_M,
        max_direct_gap=sg2b.MAX_DIRECT_GAP_M, max_hang_gap=sg2b.MAX_HANG_GAP_M,
        footprint_iou_thr=sg2b.FOOTPRINT_IOU_THR, on_floor_m=sg2b.ON_FLOOR_M,
        wall_blocker=None, yaw_deg=args.yaw_deg,
    )
    for n in nodes:
        n["on_floor"] = objects_for_edges[n["id"]].get("on_floor", n["on_floor"])

    sg = {
        "space": args.space,
        "building_yaw_deg": args.yaw_deg,
        "hierarchy": {"building": {"id": 0}, "rooms": [0], "areas": []},
        "hierarchy_edges": [],
        "nodes": nodes,
        "edges": edges,
        "rooms": [room],
        "areas": [],
        "coarse_groups": [],
        "fragments": [],
        "stats": {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_rooms": 1,
            "n_areas": 0,
            "n_coarse_groups": 0,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(sg, indent=2))
    from collections import Counter
    print(f"[export_sg] wrote {len(nodes)} nodes, {len(edges)} edges -> {args.out}")
    print(Counter(n["label"] for n in nodes))
    print(Counter(e["relation"] for e in edges))


if __name__ == "__main__":
    main()
