#!/usr/bin/env python
"""Fully automatic splat<->pointcloud alignment - given any space's own
gaussian splat ply and its corresponding independently-captured
data/<space>/pointcloud.ply, find the rigid transform (+ real scale, since
"scale_to_meters" is often just a rough initial guess) that maps the
splat's own detection frame onto the point cloud's frame, so a scene graph
built from the splat can be re-projected onto the point cloud and viewed/
combined with point-cloud-native detections (e.g. pipeline4's 3DETR).

Bakes in every lesson learned doing this manually for factory13 (see
project-factory13-alignment-and-wall-check / bug-splat-pc-yaw-sign-mismatch
memories for the history) so it doesn't need to be re-discovered by hand
each time:

  1. REFLECTION: the axis permutation used to build a splat's "geo" frame
     (x,y,z) -> (x,z,y), swapping Y<->Z to get Y-up, is by itself a
     REFLECTION (determinant -1), not a rotation. Confirmed directly on
     factory13: ICP fitness capped at 0.77 with the plain permutation vs.
     0.99 once one axis's sign was flipped to restore a proper rotation.
     Silently skipping this check would converge to a *plausible-looking*
     but mirrored result - boxes roughly the right size/region, never
     truly overlapping real objects.
  2. SCALE: a splat's own "scale_to_meters" is frequently just a rough
     estimate (e.g. cross-referenced against one known dimension). Treat it
     as an INITIAL GUESS and let ICP fitness pick the true value directly -
     confirmed directly on factory13: fitness was capped at 0.61 across
     the *entire* rotation search space at the assumed scale, and only
     climbed to 0.80+ once the scale itself was swept and corrected
     (5.37 -> ~7.95, a 48% correction).
  3. Cheap-then-expensive staging: reflection is checked first with a
     tiny, fast probe (the effect size is large - no need for a fine
     search to see it); then scale+yaw are coarse-swept only for the
     winning reflection; then the single best candidate gets a real
     multi-stage ICP refinement (progressively tighter correspondence
     threshold). Searching all of {reflection x scale x yaw} at full
     refinement cost would take a long time for no extra robustness.

Usage:
  python pipeline9/align_splat_to_pointcloud.py \
    --splat-ply data/factory13_100k_sharpen.ply \
    --pointcloud-ply data/factory_space_13/pointcloud.ply \
    --scale-to-meters-guess 5.37 \
    --out pipeline9/out/factory13_splat_to_pc_transform.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat_pointcloud_align_utils import measure_wall_yaw_deg, rot_y  # noqa: E402

SPLAT_AXIS_PERMUTE = (0, 2, 1)  # x,y,z (Z-up) -> x,z,y (Y-up) - a REFLECTION alone
AXIS_SIGN_CANDIDATES = [
    (1.0, 1.0, 1.0),    # unflipped (a reflection - kept only as a probe baseline)
    (-1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, -1.0),
]


def load_splat_opacity_filtered(ply_path, opacity_thresh=0.3):
    p = PlyData.read(str(ply_path))["vertex"]
    xyz = np.column_stack([p["x"], p["y"], p["z"]]).astype(np.float64)
    opacity = 1 / (1 + np.exp(-np.array(p["opacity"], dtype=np.float64)))
    return xyz[opacity >= opacity_thresh]


def to_o3d(xyz, voxel):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd.voxel_down_sample(voxel) if voxel else pcd


def best_fitness_over_yaws(src, dst, dst_c, threshold, yaws_deg, max_iter=50):
    src_c = src.get_center()
    best = None
    for ang in yaws_deg:
        R = rot_y(ang)
        T0 = np.eye(4)
        T0[:3, :3] = R
        T0[:3, 3] = dst_c - R @ src_c
        result = o3d.pipelines.registration.registration_icp(
            src, dst, threshold, T0,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
        if best is None or result.fitness > best[0].fitness:
            best = (result, ang)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splat-ply", required=True, help="the RAW (non-derotated) splat ply")
    ap.add_argument("--pointcloud-ply", required=True)
    ap.add_argument("--scale-to-meters-guess", type=float, required=True,
                     help="the splat's own assumed scale_to_meters - used only as a starting "
                          "point for the scale sweep below, not trusted directly")
    ap.add_argument("--opacity-thresh", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t_start = time.time()

    raw = load_splat_opacity_filtered(args.splat_ply, args.opacity_thresh)
    p = PlyData.read(args.pointcloud_ply)["vertex"]
    dst_xyz = np.column_stack([p["x"], p["y"], p["z"]]).astype(np.float64)
    print(f"[align] splat pts={len(raw):,}  pointcloud pts={len(dst_xyz):,}")

    dst_probe = to_o3d(dst_xyz, 0.10)
    dst_c_probe = dst_probe.get_center()

    # ---- stage 1: joint (reflection x scale) coarse probe ----
    # Reflection and scale are NOT separable: confirmed directly - probing
    # reflection candidates at a fixed (wrong) scale guess gave fitnesses
    # all bunched at ~0.57-0.59 with NO clear winner (the correct sign
    # picked by a hair, wrongly, in one run), because a genuinely wrong
    # scale caps fitness for every candidate similarly and hides the real
    # reflection signal. Only once tested at the TRUE scale did the correct
    # sign separate cleanly (0.99 vs 0.77). So scale must be swept for
    # EVERY reflection candidate here, not fixed at the initial guess.
    print("[align] stage 1/4: joint reflection x scale probe (cheap, coarse)")
    probe_yaws = list(range(0, 360, 60))
    probe_scale_mults = np.array([0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7, 2.0])
    sign_scores = []
    for sign in AXIS_SIGN_CANDIDATES:
        best_for_sign = None
        for mult in probe_scale_mults:
            scale = args.scale_to_meters_guess * mult
            xyz = raw[:, list(SPLAT_AXIS_PERMUTE)] * np.array(sign) * scale
            src_probe = to_o3d(xyz, 0.10)
            best, ang = best_fitness_over_yaws(src_probe, dst_probe, dst_c_probe, 0.3,
                                                probe_yaws, max_iter=25)
            if best_for_sign is None or best.fitness > best_for_sign[0].fitness:
                best_for_sign = (best, scale, ang)
        print(f"[align]   axis_sign={sign} -> best probe fitness={best_for_sign[0].fitness:.4f} "
              f"@ scale={best_for_sign[1]:.3f} yaw={best_for_sign[2]}")
        sign_scores.append((best_for_sign[0].fitness, sign, best_for_sign[1]))
    sign_scores.sort(key=lambda x: -x[0])
    best_sign = sign_scores[0][1]
    probe_scale_hint = sign_scores[0][2]
    if sign_scores[0][0] < sign_scores[1][0] * 1.05:
        print(f"[align]   WARNING: top two axis_sign candidates are close "
              f"({sign_scores[0][0]:.4f} vs {sign_scores[1][0]:.4f}) - reflection may be ambiguous")
    print(f"[align] -> best axis_sign={best_sign} (scale hint={probe_scale_hint:.3f})")

    # ---- stage 2: finer scale x yaw sweep for the winning reflection ----
    print("[align] stage 2/4: coarse scale x yaw sweep")
    dst_c = dst_c_probe
    scale_mults = probe_scale_hint / args.scale_to_meters_guess * np.array(
        [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    coarse_yaws = list(range(0, 360, 20))
    best_overall = None
    for mult in scale_mults:
        scale = args.scale_to_meters_guess * mult
        xyz = raw[:, list(SPLAT_AXIS_PERMUTE)] * np.array(best_sign) * scale
        src = to_o3d(xyz, 0.06)
        best, ang = best_fitness_over_yaws(src, dst_probe, dst_c, 0.25, coarse_yaws, max_iter=60)
        print(f"[align]   scale={scale:.3f} (x{mult:.2f}) -> fitness={best.fitness:.4f} "
              f"rmse={best.inlier_rmse:.4f} @ yaw={ang}")
        if best_overall is None or best.fitness > best_overall[0].fitness:
            best_overall = (best, scale, ang)

    # ---- stage 3: refine scale finely around the coarse peak ----
    print("[align] stage 3/4: refining scale around the coarse peak")
    coarse_scale = best_overall[1]
    fine_mults = np.linspace(0.9, 1.1, 9)
    for mult in fine_mults:
        scale = coarse_scale * mult
        xyz = raw[:, list(SPLAT_AXIS_PERMUTE)] * np.array(best_sign) * scale
        src = to_o3d(xyz, 0.04)
        yaw_window = [best_overall[2] + d for d in range(-15, 16, 5)]
        best, ang = best_fitness_over_yaws(src, dst_probe, dst_c, 0.20, yaw_window, max_iter=80)
        print(f"[align]   scale={scale:.3f} -> fitness={best.fitness:.4f} rmse={best.inlier_rmse:.4f} "
              f"@ yaw={ang}")
        if best.fitness > best_overall[0].fitness:
            best_overall = (best, scale, ang)

    # ---- stage 4: multi-stage threshold-tightening refinement at the winner ----
    print("[align] stage 4/4: multi-stage refinement at the best (sign, scale, yaw)")
    final_scale = best_overall[1]
    xyz = raw[:, list(SPLAT_AXIS_PERMUTE)] * np.array(best_sign) * final_scale
    src = to_o3d(xyz, 0.04)
    dst = to_o3d(dst_xyz, 0.04)
    src_c, dst_c = src.get_center(), dst.get_center()
    R0 = rot_y(best_overall[2])
    T0 = np.eye(4)
    T0[:3, :3] = R0
    T0[:3, 3] = dst_c - R0 @ src_c

    result = o3d.pipelines.registration.registration_icp(
        src, dst, 0.25, T0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    print(f"[align]   coarse (thresh=0.25): fitness={result.fitness:.4f} rmse={result.inlier_rmse:.4f}")
    for thresh in (0.10, 0.05):
        result = o3d.pipelines.registration.registration_icp(
            src, dst, thresh, result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=150))
        print(f"[align]   refined (thresh={thresh}): fitness={result.fitness:.4f} rmse={result.inlier_rmse:.4f}")

    T = result.transformation
    box_yaw_deg = measure_wall_yaw_deg(args.pointcloud_ply)
    print(f"[align] measured real wall angle (for box orientation) = {box_yaw_deg:.3f} deg")
    print(f"[align] TOTAL runtime: {time.time()-t_start:.1f}s")

    out = {
        "transform_4x4": T.tolist(),
        "fitness": result.fitness, "inlier_rmse": result.inlier_rmse,
        "true_scale_to_meters": final_scale,
        "scale_to_meters_guess": args.scale_to_meters_guess,
        "axis_map": list(SPLAT_AXIS_PERMUTE), "axis_sign": list(best_sign),
        "building_yaw_deg": box_yaw_deg,
        "note": "apply as: pc_xyz = (T[:3,:3] @ v.T).T + T[:3,3], where "
                f"v = splat_xyz[:,{list(SPLAT_AXIS_PERMUTE)}] * {list(best_sign)} * {final_scale} "
                "- found fully automatically (reflection probe -> scale+yaw sweep -> multi-stage "
                "ICP refine), not assumed. building_yaw_deg is the point cloud's OWN measured "
                "wall angle (independent of this transform's rotation) for rendering box "
                "orientation - see apply_scenegraph_to_pointcloud.py's own docstring for why "
                "extracting it from the transform's rotation matrix instead was tried and wrong.",
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
