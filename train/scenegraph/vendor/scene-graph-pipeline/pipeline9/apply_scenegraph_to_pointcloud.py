#!/usr/bin/env python
"""Re-project the (already merged) factory13 splat-based scene graph onto
data/factory_space_13/pointcloud.ply's own frame, using the rigid transform
already found+validated by align_splat_to_pointcloud.py (ICP, fitness 0.61,
confirmed by direct overlay - see project-factory13-alignment-and-wall-check
memory), so it can be viewed with the real (denser, independently-captured)
point cloud as the background instead of the gaussian splat.

Every node's centroid/box_center is already stored in the splat's "geo"
frame (Y-up, meters: splat_xyz[:,(0,2,1)] * scale_to_meters - the exact
convention export_scene_graph_for_point_viewer.py used to build it), which
is precisely the frame align_splat_to_pointcloud.py's saved transform maps
FROM.

Position vs. orientation are handled differently, matching how
sg_3d_viewer.html actually renders a box - a box's WIREFRAME is drawn at its
own local size around box_center, then given one single extra spin,
`rotation.y = -radians(building_yaw_deg)`, shared by every node:
  - CENTROID/box_center: transformed through the full rotation+translation,
    since real position needs the true 3D transform.
  - bbox_size: kept as the ORIGINAL, un-rotated size (just rescaled) -
    recomputing an axis-aligned bounding box from 8 rotated corners was
    tried first and rejected: it silently inflates size for any object
    whose rotated bounding box doesn't line up with the world axes (a
    near-180 deg yaw with a ~1 deg residual tilt still inflates some
    boxes up to 2x - confirmed directly on this data).
  - building_yaw_deg: set to the transform's own extracted Y-yaw (the
    rotation matrix here is a pure Y-yaw to within ~0.01 rad residual -
    confirmed directly), so the viewer's existing per-box spin does the
    orientation work instead of baking it into a recomputed AABB. Without
    this, every box renders axis-aligned to world X/Z while the real room
    (and the real point cloud drawn behind it) is visibly tilted - which
    is exactly what "still rotated by an angle" looked like even after
    the mirrored-axis and double-yaw bugs were fixed.

Edges (support/proximity/comparative relations) are relative-geometry
statements between nodes and are preserved as-is: a single rigid transform
applied uniformly to every node changes no distances or relative positions
that those relations were computed from.

Usage:
  python pipeline9/apply_scenegraph_to_pointcloud.py \
    --scene-graph ui/_spaces/factory13_100k_sharpen_splatanalyzer/scene_graph.json \
    --transform pipeline9/out/factory13_splat_to_pc_transform_v3.json \
    --source-scale-to-meters 5.37 \
    --space factory13_pointcloud_scenegraph \
    --out ui/_spaces/factory13_pointcloud_scenegraph/scene_graph.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat_pointcloud_align_utils import measure_wall_yaw_deg, transform_points  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-graph", required=True)
    ap.add_argument("--transform", required=True)
    ap.add_argument("--source-scale-to-meters", type=float, required=True,
                     help="scale_to_meters the SOURCE scene_graph.json's centroid/bbox_size were "
                          "already baked with (export_scene_graph_for_point_viewer.py's own "
                          "to_geo_frame) - rescaled to the transform's own scale before applying "
                          "it, since geo=raw_permuted*scale is a pure linear scaling from the "
                          "origin and the two need not match (confirmed directly: the splat's "
                          "long-assumed scale_to_meters=5.37 turned out to cap ICP fitness at "
                          "0.61 regardless of rotation; sweeping scale directly found a real "
                          "peak of 0.80 fitness at ~7.95 - the transform file's own scale)")
    ap.add_argument("--space", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pointcloud-ply", required=True,
                     help="used for two things: (1) measuring the room's real wall angle "
                          "directly (robust min-area-rect over the wall boundary) for box "
                          "ORIENTATION, when --transform doesn't already carry a "
                          "'building_yaw_deg' - extracting a yaw from the ICP rotation matrix "
                          "directly was tried and rejected, it conflates the reflection-undo "
                          "with the true in-plane tilt; (2) dropping any transformed node that "
                          "falls outside the point cloud's real captured extent - splat "
                          "background/floater geometry can reconstruct well beyond the room "
                          "actually scanned, confirmed directly on factory_space_15")
    args = ap.parse_args()

    sg = json.loads(Path(args.scene_graph).read_text())
    tdata = json.loads(Path(args.transform).read_text())
    T = np.array(tdata["transform_4x4"])
    transform_scale = tdata.get("true_scale_to_meters", args.source_scale_to_meters)
    rescale = transform_scale / args.source_scale_to_meters
    # The stored geo frame's own axis permutation (0,2,1) is, by itself, a
    # REFLECTION (determinant -1) - confirmed directly: ICP fitness jumped
    # from 0.77 to 0.99 once a sign flip was added to restore a proper
    # rotation. Without undoing that reflection here, no rigid rotation can
    # ever correctly align this graph to the point cloud - it would come out
    # mirrored (which is exactly what a first attempt without this looked
    # like: every box roughly the right SIZE and in the right REGION, but
    # not truly overlapping real objects).
    axis_sign = np.array(tdata.get("axis_sign", [1.0, 1.0, 1.0]), dtype=np.float64)
    print(f"[apply_sg_to_pc] rescaling stored geo-frame positions by "
          f"{transform_scale}/{args.source_scale_to_meters} = {rescale:.4f}, "
          f"axis_sign={axis_sign.tolist()}, before transforming")

    if "building_yaw_deg" in tdata:
        yaw_deg = float(tdata["building_yaw_deg"])
        print(f"[apply_sg_to_pc] using building_yaw_deg from transform file = {yaw_deg:.3f} deg")
    else:
        if not args.pointcloud_ply:
            raise SystemExit("--transform has no building_yaw_deg - pass --pointcloud-ply "
                              "so it can be measured directly")
        yaw_deg = measure_wall_yaw_deg(args.pointcloud_ply)
        print(f"[apply_sg_to_pc] measured real wall angle (box orientation) = {yaw_deg:.3f} deg")

    for n in sg["nodes"]:
        c = np.array(n["centroid"], dtype=np.float64) * rescale * axis_sign
        s = np.array(n["bbox_size"], dtype=np.float64) * rescale
        new_center = transform_points(c[None, :], T)[0]
        n["centroid"] = [round(float(v), 4) for v in new_center]
        n["box_center"] = [round(float(v), 4) for v in new_center]
        n["bbox_size"] = [round(float(v), 4) for v in s]

    # A gaussian splat commonly reconstructs some stray background/floater
    # geometry beyond the room actually captured by the point cloud scanner
    # - any source detection built from that (e.g. multilevel_topdown_v2.py
    # scanning the splat's own auto-detected room bounds) can land WELL
    # outside the point cloud's real extent once transformed here. Confirmed
    # directly on factory_space_15: a whole ring of boxes appeared outside
    # the point cloud's real footprint on every side after transforming -
    # nothing physically real is out there, since the point cloud is the
    # actual scan. Drop anything outside the real captured volume (+ a
    # margin, since a real object right at the boundary shouldn't be cut).
    if args.pointcloud_ply:
        p_pc = PlyData.read(args.pointcloud_ply)["vertex"]
        pc_xyz = np.column_stack([p_pc["x"], p_pc["y"], p_pc["z"]]).astype(np.float64)
        margin = 0.3
        pc_lo, pc_hi = pc_xyz.min(axis=0) - margin, pc_xyz.max(axis=0) + margin
        before = len(sg["nodes"])
        kept, dropped = [], []
        for n in sg["nodes"]:
            c = np.array(n["centroid"])
            if np.all(c >= pc_lo) and np.all(c <= pc_hi):
                kept.append(n)
            else:
                dropped.append(n)
        sg["nodes"] = kept
        if dropped:
            print(f"[apply_sg_to_pc] dropped {len(dropped)}/{before} nodes outside the real "
                  f"point cloud's captured extent (+{margin}m margin) - splat background/floater "
                  f"artifacts, not real objects:")
            for n in dropped:
                print(f"[apply_sg_to_pc]   {n['label']:20s} centroid={n['centroid']}")

        # The whole-extent clip above only catches nodes outside the
        # BUILDING's overall bounding box - too coarse for a large multi-bay
        # building, where big empty gaps between bays/aisles are still
        # comfortably inside that box. A node whose transformed position
        # drifted into such a gap (e.g. from imperfect ICP alignment
        # fitness) sails through and renders as a box floating in visibly
        # empty space. In factory 14, 14 of 41 nodes had zero real points
        # anywhere near their own footprint despite passing the extent clip,
        # so merge_splat_with_p4.verify_occupancy() (volume-density plus
        # footprint-fill) is applied here too, not only to merge candidates.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from merge_splat_with_p4 import verify_occupancy  # noqa: E402
        from scipy.spatial import cKDTree
        tree = cKDTree(pc_xyz)
        before2 = len(sg["nodes"])
        occ_kept, occ_dropped = [], []
        for n in sg["nodes"]:
            ok, n_inside = verify_occupancy(n, tree, pc_xyz)
            (occ_kept if ok else occ_dropped).append((n, n_inside))
        sg["nodes"] = [n for n, _ in occ_kept]
        if occ_dropped:
            print(f"[apply_sg_to_pc] dropped {len(occ_dropped)}/{before2} nodes with empty/too-"
                  f"sparse real point coverage in their own footprint:")
            for n, n_inside in occ_dropped:
                print(f"[apply_sg_to_pc]   {n['label']:20s} centroid={n['centroid']} "
                      f"n_pts_inside={n_inside}")

    for r in sg.get("rooms", []):
        if r.get("centroid_xz") is not None:
            cx, cz = r["centroid_xz"]
            v = np.array([cx, 0.0, cz]) * rescale * axis_sign
            pt = transform_points(v[None, :], T)[0]
            r["centroid_xz"] = [round(float(pt[0]), 4), round(float(pt[2]), 4)]

    sg["space"] = args.space
    # sg_3d_viewer.html spins every box wireframe by rotation.y =
    # -radians(building_yaw_deg) - set directly to the measured wall angle
    # (verified: rendering with this value puts box edges parallel to the
    # room's real, independently-measured wall boundary; 0 left every box
    # axis-aligned to world X/Z, visibly mismatched against the real tilted
    # room, and the stale splat-frame value double-rotated on top of
    # already-transformed positions).
    sg["building_yaw_deg"] = float(yaw_deg)
    sg["_reprojected_from"] = {
        "source_space": sg.get("space"), "source_scene_graph": str(args.scene_graph),
        "transform_file": str(args.transform),
        "note": "nodes/edges/labels identical to the source graph - only centroid/box_center/"
                "bbox_size were re-projected onto data/factory_space_13/pointcloud.ply's frame "
                "via the saved splat<->pointcloud ICP transform.",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(sg, indent=2))
    print(f"[apply_sg_to_pc] {len(sg['nodes'])} nodes re-projected -> {args.out}")


if __name__ == "__main__":
    main()
