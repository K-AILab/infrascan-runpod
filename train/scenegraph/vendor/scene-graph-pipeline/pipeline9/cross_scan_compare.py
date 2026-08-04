#!/usr/bin/env python
"""Measure how far two independent scans of the same room agree.

Where two captures cover one space, their overlap gives a quality metric that
needs no hand-labelled ground truth. Register the two clouds, match the
detections, and any label disagreement is by construction an error in one scan
or the other — it is the same physical object.

Reports three numbers:
  * matched pairs        — recall consistency
  * label agreement      — labelling stability
  * only-in-A / only-in-B — misses or false positives

Run it before and after a change and the numbers say whether the change
helped. Note that it measures CONSISTENCY, not correctness: two scans can
agree on the same wrong label. Treat a drop as a warning worth investigating
rather than a verdict.

This is measurement only. It copies nothing between scans and hard-codes no
object or label — transferring detections across would destroy the
independence that gives the comparison its value.

Registration is a coarse yaw sweep followed by point-to-point ICP. Scale is
not searched, since both captures are already in metres.

Usage:
  python cross_scan_compare.py --space-a factory_space_13 \
    --space-b factory_space_14 \
    --boxes-a out/factory_space_13_sg_pointcloud.json \
    --boxes-b out/factory_space_14_sg_pointcloud.json \
    --out out/crossscan.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import open3d as o3d


def load_cloud(path, voxel):
    pcd = o3d.io.read_point_cloud(str(path))
    return pcd.voxel_down_sample(voxel) if voxel > 0 else pcd


def nodes_of(path):
    d = json.loads(Path(path).read_text())
    if "nodes" in d:
        return [{"label": n["label"],
                 "c": np.asarray(n.get("box_center", n["centroid"]), dtype=float),
                 "s": np.asarray(n["bbox_size"], dtype=float),
                 "id": n["id"]} for n in d["nodes"]], d
    return [{"label": b["label"], "c": np.asarray(b["center"], dtype=float),
             "s": np.asarray(b["size"], dtype=float), "id": i}
            for i, b in enumerate(d["boxes"])], d


def register(a, b, yaw_steps, icp_thresh):
    """Coarse yaw sweep about the vertical axis, then ICP. The captures are in
    metres already, so only rotation and translation are unknown."""
    ca = a.get_center()
    cb = b.get_center()
    best = (None, -1.0, None)
    for deg in np.arange(0, 360, 360 / yaw_steps):
        th = np.radians(deg)
        R = np.array([[np.cos(th), 0, np.sin(th)],
                      [0, 1, 0],
                      [-np.sin(th), 0, np.cos(th)]])          # Y-up captures
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = cb - R @ ca
        ev = o3d.pipelines.registration.evaluate_registration(a, b, icp_thresh, T)
        if ev.fitness > best[1]:
            best = (deg, ev.fitness, T)
    deg, fit, T0 = best
    print(f"[cross] coarse yaw {deg:.0f}° -> fitness {fit:.3f}")
    res = o3d.pipelines.registration.registration_icp(
        a, b, icp_thresh, T0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    print(f"[cross] ICP fitness {res.fitness:.3f}  rmse {res.inlier_rmse:.4f} m")
    return np.asarray(res.transformation), res.fitness, res.inlier_rmse


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space-a", required=True)
    ap.add_argument("--space-b", required=True)
    ap.add_argument("--boxes-a", required=True, help="scene graph / boxes in A's cloud frame")
    ap.add_argument("--boxes-b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.08)
    ap.add_argument("--icp-thresh", type=float, default=0.35)
    ap.add_argument("--yaw-steps", type=int, default=24)
    ap.add_argument("--match-dist-m", type=float, default=0.9,
                    help="centre distance within which two boxes are the same object")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    pa = root / "data" / args.space_a / "pointcloud.ply"
    pb = root / "data" / args.space_b / "pointcloud.ply"
    print(f"[cross] loading {pa.name} / {pb.name} at {args.voxel} m voxel …")
    A, B = load_cloud(pa, args.voxel), load_cloud(pb, args.voxel)
    print(f"[cross] {len(A.points):,} vs {len(B.points):,} points after downsampling")

    T, fit, rmse = register(A, B, args.yaw_steps, args.icp_thresh)
    if fit < 0.3:
        print(f"[cross] WARNING registration fitness {fit:.3f} is low — the two scans "
              f"may not overlap as assumed; treat everything below as unreliable")

    na, da = nodes_of(args.boxes_a)
    nb, db = nodes_of(args.boxes_b)
    for n in na:                                  # A's boxes into B's frame
        n["c"] = (T[:3, :3] @ n["c"]) + T[:3, 3]

    used_b = set()
    matched, only_a = [], []
    for n in na:
        best, bd = None, np.inf
        for j, m in enumerate(nb):
            if j in used_b:
                continue
            d = float(np.linalg.norm(n["c"] - m["c"]))
            if d < bd:
                best, bd = j, d
        if best is not None and bd <= args.match_dist_m:
            used_b.add(best)
            matched.append((n, nb[best], bd))
        else:
            only_a.append(n)
    only_b = [m for j, m in enumerate(nb) if j not in used_b]

    agree = [(x, y, d) for x, y, d in matched if x["label"] == y["label"]]
    disagree = [(x, y, d) for x, y, d in matched if x["label"] != y["label"]]

    print(f"\n[cross] {len(na)} boxes in {args.space_a}, {len(nb)} in {args.space_b}")
    print(f"[cross] matched {len(matched)} pairs within {args.match_dist_m} m "
          f"(median centre offset {np.median([d for _x, _y, d in matched]):.2f} m)"
          if matched else "[cross] no pairs matched")
    print(f"[cross]   label agrees    : {len(agree)}")
    print(f"[cross]   label disagrees : {len(disagree)}")
    print(f"[cross] only in {args.space_a}: {len(only_a)}   "
          f"only in {args.space_b}: {len(only_b)}")

    if matched:
        rate = len(agree) / len(matched)
        print(f"\n[cross] label agreement on co-detected objects: {rate:.0%} — every "
              f"disagreement is an error in one scan or the other, by construction")
    if disagree:
        print("\n  disagreements:")
        for a_lbl, b_lbl in Counter((x["label"], y["label"])
                                    for x, y, _d in disagree).most_common(20):
            print(f"    {a_lbl[0]:<20} vs {a_lbl[1]:<20} x{b_lbl}")
    if only_a:
        print(f"\n  present only in {args.space_a} (missed by {args.space_b}, or false "
              f"positives here): {dict(Counter(n['label'] for n in only_a))}")
    if only_b:
        print(f"  present only in {args.space_b}: "
              f"{dict(Counter(m['label'] for m in only_b))}")

    Path(args.out).write_text(json.dumps({
        "space_a": args.space_a, "space_b": args.space_b,
        "transform_a_to_b": T.tolist(), "icp_fitness": fit, "icp_rmse_m": rmse,
        "n_a": len(na), "n_b": len(nb), "n_matched": len(matched),
        "n_label_agree": len(agree), "n_label_disagree": len(disagree),
        "label_agreement": (len(agree) / len(matched)) if matched else None,
        "disagreements": [{"a": x["label"], "b": y["label"], "dist_m": round(d, 3),
                           "at": [round(v, 3) for v in y["c"]]}
                          for x, y, d in disagree],
        "only_in_a": [{"label": n["label"], "at": [round(v, 3) for v in n["c"]]}
                      for n in only_a],
        "only_in_b": [{"label": m["label"], "at": [round(v, 3) for v in m["c"]]}
                      for m in only_b],
    }, indent=2))
    print(f"\n-> {args.out}")

    if matched:
        print(f"\n[cross] USE: re-run this after a code change. Higher matched-pair "
              f"count means better recall consistency; higher label agreement means "
              f"more stable labelling. Neither needs any ground truth.")


if __name__ == "__main__":
    main()
