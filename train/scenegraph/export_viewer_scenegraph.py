"""Produce the viewer-facing scene graph for our tri-viewer's 3D (splat) mode.

Approach A's final scene graph (`<space>_final.json`) carries the good CLIP
labels + relationship edges, but its box centroids are in the POINT CLOUD frame
(Y-up meters). Our 3D viewer renders the `.ksplat`, which is byte-for-byte the
same coordinate frame as `splat.ply` (ply2ksplat applies no rotation), i.e. the
SPLAT-NATIVE frame — the exact frame `scene_boxes_refit.json` boxes already live
in, and the same frame `cameras.json` `pos` uses (which the viewer already trusts
to place scanpoints).

The refit boxes and the final scene-graph nodes share an identity: a node's `id`
is its index into `scene_boxes_refit.json["boxes"]` (see
`export_scene_graph_for_point_viewer.py`, which builds nodes with
`for i, b in enumerate(data["boxes"])`). So we can attach each surviving node's
FINAL label to its ORIGINAL splat-native box with zero coordinate math — no
scale, no axis permute, no inverse transform, nothing to get subtly wrong.
Structural nodes that CLIP dropped in `_final.json` are simply absent here.

Output (`scene_graph.json`), consumed by the viewer:
    {
      "slug": "...", "coord_frame": "splat", "up_axis": "z", "version": 1,
      "nodes": [{"id", "label", "center":[x,y,z], "size":[x,y,z],
                 "clip_topk"?, "material"?, "on_floor"?}],
      "edges": [{"src", "dst", "relation", "edge_type", "weight"?}],
      "labels": {"chair": 3, ...}          # per-label counts, for a legend
    }
Boxes are axis-aligned in the splat frame (Approach A runs with --yaw-deg 0), so
the viewer draws them as plain AABBs: min = center - size/2, max = center + size/2.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

VERSION = 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refit-boxes", required=True,
                    help="scene_boxes_refit.json — splat-native boxes; node id indexes into it")
    ap.add_argument("--final", required=True,
                    help="<space>_final.json — final CLIP labels + edges (pointcloud frame; "
                         "we take only labels/edges, not its coordinates)")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    boxes = json.loads(Path(args.refit_boxes).read_text())["boxes"]
    final = json.loads(Path(args.final).read_text())

    nodes = []
    for n in final.get("nodes", []):
        i = n["id"]
        if not isinstance(i, int) or i < 0 or i >= len(boxes):
            # id must index the refit boxes; skip anything that doesn't (shouldn't happen)
            print(f"[viewer-export] WARN node id {i} out of range 0..{len(boxes)-1} — skipping")
            continue
        b = boxes[i]
        node = {
            "id": i,
            "label": n.get("label", b.get("label", "object")),
            "center": [float(v) for v in b["center"]],
            "size": [float(abs(v)) for v in b["size"]],
        }
        if n.get("clip_topk"):
            node["clip_topk"] = n["clip_topk"]
        if n.get("material"):
            node["material"] = n["material"]
        if "on_floor" in n:
            node["on_floor"] = bool(n["on_floor"])
        nodes.append(node)

    kept_ids = {n["id"] for n in nodes}
    edges = []
    for e in final.get("edges", []):
        # only keep edges whose endpoints both survived the CLIP structural drop
        if e.get("src") in kept_ids and e.get("dst") in kept_ids:
            edge = {"src": e["src"], "dst": e["dst"],
                    "relation": e.get("relation", ""), "edge_type": e.get("edge_type", "")}
            if "weight" in e:
                edge["weight"] = e["weight"]
            edges.append(edge)

    out = {
        "slug": args.slug,
        "coord_frame": "splat",
        "up_axis": "z",
        "version": VERSION,
        "nodes": nodes,
        "edges": edges,
        "labels": dict(Counter(n["label"] for n in nodes)),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[viewer-export] wrote {args.out}: {len(nodes)} nodes, {len(edges)} edges, "
          f"labels={out['labels']}")


if __name__ == "__main__":
    main()
