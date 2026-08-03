#!/usr/bin/env python
"""Re-run REAL CLIP labeling (pipeline2b/geo_label_clip.py - crops the
node's own real camera photos, not a geometric heuristic) on any scene
graph already in a point cloud's real frame, replacing whatever labels it
came in with.

Why this exists: multilevel_topdown_v2.py's own labels (table_or_desk/
shelf_or_rack/unclassified_sized_ok) are a pure HEIGHT-BEHAVIOR heuristic -
never looked at a single real photo. Once a scene graph has been
re-projected onto a real point cloud (apply_scenegraph_to_pointcloud.py),
that same room's real camera panos are available, so there is no reason to
keep guessing from geometry alone.

Pipeline:
  1. Build a geo_label_clip.py-compatible "_geo.json" + "_geo_points.npz"
     from the scene graph's own nodes - centroid/bbox_min/bbox_max from
     the node, real point samples from the point cloud itself (used to
     get a tight, accurate crop in each camera view - a 3D box corner
     crop can cover ~2x the object's real screen area from an oblique
     angle and feed CLIP mostly background).
  2. Run geo_label_clip.py for real (--space gives it camera/pano access).
  3. Merge the new label/clip_topk/material/attributes back onto the
     ORIGINAL scene graph nodes by id - position/size/edges are untouched,
     only what CLIP actually re-examined changes. A node CLIP itself
     flags as structural (a real door/window/wall) is dropped, matching
     every other space's own convention - it was never a real "object" to
     begin with.

Usage:
  python pipeline9/relabel_with_clip.py \
    --scene-graph ui/_spaces/factory15_pointcloud_scenegraph/scene_graph.json \
    --pointcloud-ply data/factory_space_15/pointcloud.ply \
    --space factory_space_15 \
    --geo-json-out pipeline9/out/factory15_pointcloud_scenegraph_relabel_geo.json \
    --out ui/_spaces/factory15_pointcloud_scenegraph/scene_graph.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-graph", required=True)
    ap.add_argument("--pointcloud-ply", required=True)
    ap.add_argument("--space", required=True,
                     help="registered space name for camera/pano access, e.g. factory_space_15 "
                          "(NOT one of this project's synthetic _p4/_splatanalyzer spaces - the "
                          "one with real cameras.json/views)")
    ap.add_argument("--geo-json-out", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pts-per-node", type=int, default=400)
    args = ap.parse_args()

    sg = json.loads(Path(args.scene_graph).read_text())
    p = PlyData.read(args.pointcloud_ply)["vertex"]
    xyz = np.column_stack([p["x"], p["y"], p["z"]]).astype(np.float64)
    rgb = np.column_stack([p["red"], p["green"], p["blue"]]).astype(np.float64)
    tree = cKDTree(xyz)

    real_nodes = [n for n in sg["nodes"] if not n["label"].startswith("removed_")]
    geo_nodes = []
    npz_arrays = {}
    for n in real_nodes:
        c = np.array(n["centroid"], dtype=np.float64)
        s = np.array(n["bbox_size"], dtype=np.float64)
        bmin, bmax = (c - s / 2).tolist(), (c + s / 2).tolist()
        r = float(np.linalg.norm(s) / 2 + 0.15)
        idx = tree.query_ball_point(c, r=r)
        pts = xyz[idx]
        cols = rgb[idx]
        if len(pts):
            inside = np.all((pts >= (c - s / 2 - 0.05)) & (pts <= (c + s / 2 + 0.05)), axis=1)
            pts, cols = pts[inside], cols[inside]
        if len(pts) > args.max_pts_per_node:
            sel = np.random.default_rng(0).choice(len(pts), args.max_pts_per_node, replace=False)
            pts, cols = pts[sel], cols[sel]
        nid = n["id"]
        npz_arrays[f"xyz_{nid}"] = pts.astype(np.float32)
        npz_arrays[f"rgb_{nid}"] = cols.astype(np.float32)
        geo_nodes.append({
            "id": nid, "label": n["label"], "centroid": c.tolist(),
            "bbox_min": bmin, "bbox_max": bmax, "bbox_size": s.tolist(),
            "n_points": int(len(pts)), "mean_rgb": cols.mean(axis=0).tolist() if len(cols) else [128, 128, 128],
            "is_structure": False,
        })

    floor_y = float(np.percentile(xyz[:, 1], 1))
    ceil_y = float(np.percentile(xyz[:, 1], 99))
    geo = {
        "space": args.space, "source": "relabel_with_clip.py", "checkpoint": None,
        "nodes": geo_nodes, "structure_segments": [], "floor_y": floor_y, "ceil_y": ceil_y,
    }
    Path(args.geo_json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.geo_json_out).write_text(json.dumps(geo, indent=2))
    npz_path = Path(args.geo_json_out).with_name(Path(args.geo_json_out).stem + "_points.npz")
    np.savez(npz_path, **npz_arrays)
    print(f"[relabel] {len(geo_nodes)} nodes -> {args.geo_json_out} + {npz_path}")

    subprocess.run([
        PY, str(REPO / "pipeline2b" / "geo_label_clip.py"),
        "--space", args.space, "--geo-json", str(args.geo_json_out), "--no-annotations",
    ], check=True, cwd=str(REPO))

    relabeled = json.loads(Path(args.geo_json_out).read_text())["nodes"]
    by_id = {n["id"]: n for n in relabeled}

    kept, dropped = [], []
    for n in sg["nodes"]:
        if n["label"].startswith("removed_"):
            kept.append(n)
            continue
        new = by_id.get(n["id"])
        if new is None:
            kept.append(n)
            continue
        if new.get("is_structure"):
            dropped.append((n["label"], new.get("label"), n["centroid"]))
            continue
        n["label"] = new["label"]
        n["clip_topk"] = new.get("clip_topk", [])
        n["material"] = new.get("material")
        n["material_score"] = new.get("material_score")
        kept.append(n)

    print(f"[relabel] {len(dropped)} nodes CLIP flagged as structural (real door/window/wall/etc) "
          f"- dropped, matching every other space's own convention:")
    for old, new, cen in dropped:
        print(f"[relabel]   {old} -> {new} at {[round(x,2) for x in cen]}")

    sg["nodes"] = kept
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(sg, indent=2))
    from collections import Counter
    print(f"[relabel] final label counts: {Counter(n['label'] for n in kept if not n['label'].startswith('removed_'))}")
    print(f"[relabel] -> {args.out}")


if __name__ == "__main__":
    main()
