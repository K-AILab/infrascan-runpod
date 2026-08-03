#!/usr/bin/env python3
"""
Offline step 3: Add 3D world positions to each proposal.

For each proposal, samples pixels inside the bbox from the DA3 depth map,
filters by depth confidence, backprojects to world coordinates using the
camera pose from cameras.json.

Usage:
    conda activate sam3
    python sandbox/offline/03_backproject.py

Input:  sandbox/offline/out/metadata.json
Output: sandbox/offline/out/metadata.json  (updated in-place — adds world_pos)

Camera convention (from SETUP.md): +Y down (OpenCV). DA3 depth is
unreliable on thin objects, glass, mirrors — those will have NaN world_pos.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
PROJ        = Path(os.environ.get("INFRASCAN_TAGGING_MODELS", str(ROOT.parent / "external")))

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paths import space, space_choices  # noqa: E402

# ── Config ─────────────────────────────────────────────────────────────────
CONF_MIN       = 0.4        # DA3 confidence threshold — ignore unreliable depth
GRID           = 7          # sample GRID×GRID pixels inside bbox (49 samples)
DEPTH_Q_LO     = 0.25       # trim depth outliers: lower quantile
DEPTH_Q_HI     = 0.75       # upper quantile (use median band)
MIN_VALID_PTS  = 8          # need at least this many confident depth samples
DBSCAN_EPS_M   = 0.15       # cluster radius after backprojection (metres)
DBSCAN_MIN_PTS = 4          # min cluster size — drops scattered depth bleeds
MIN_CLUSTER_PTS = 6         # final survivors required; else world_pos = None


def load_npz(da3_dir: Path, frame_idx: int):
    path = da3_dir / f"frame_{frame_idx}.npz"
    if not path.exists():
        return None
    d = np.load(str(path))
    return d["depth"].astype(np.float32), d["conf"].astype(np.float32), d["intrinsics"].astype(np.float32)


def backproject(bbox, depth, conf, K, R, pos) -> list | None:
    """Backproject bbox center depth sample to world XYZ.

    Returns [x, y, z] or None if depth is unreliable.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = depth.shape

    # Sample grid inside bbox
    xs = np.linspace(x1, x2 - 1, GRID, dtype=int).clip(0, W - 1)
    ys = np.linspace(y1, y2 - 1, GRID, dtype=int).clip(0, H - 1)
    uu, vv = np.meshgrid(xs, ys)
    uu = uu.ravel()
    vv = vv.ravel()

    depths = depth[vv, uu]
    confs  = conf[vv, uu]

    # Filter by confidence and positive depth
    valid = (confs >= CONF_MIN) & (depths > 0.01)
    if valid.sum() < MIN_VALID_PTS:
        return None

    d_vals = depths[valid]
    u_vals = uu[valid].astype(np.float32)
    v_vals = vv[valid].astype(np.float32)

    # Trim depth outliers
    lo = np.quantile(d_vals, DEPTH_Q_LO)
    hi = np.quantile(d_vals, DEPTH_Q_HI)
    band = (d_vals >= lo) & (d_vals <= hi)
    if band.sum() < 1:
        band = np.ones(len(d_vals), dtype=bool)

    d_vals = d_vals[band]
    u_vals = u_vals[band]
    v_vals = v_vals[band]

    # Camera-space rays: p_cam = K_inv @ [u, v, 1] * depth
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X_cam = (u_vals - cx) / fx * d_vals
    Y_cam = (v_vals - cy) / fy * d_vals
    Z_cam = d_vals
    pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)   # (N, 3)

    # World-space: p_world = R @ p_cam + pos
    R_arr = np.array(R, dtype=np.float32)
    pos_arr = np.array(pos, dtype=np.float32)
    pts_world = (pts_cam @ R_arr.T) + pos_arr

    # DBSCAN-keep-best-cluster: when the bbox straddles foreground and
    # background, backprojected points split into two depth clusters
    # (object surface vs wall behind). Take only the largest dense cluster
    # so the median lands on the object, not between it and the background.
    if len(pts_world) >= DBSCAN_MIN_PTS:
        labels = DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_PTS).fit_predict(pts_world)
        valid = labels[labels >= 0]
        if len(valid) > 0:
            best_label = int(np.bincount(valid).argmax())
            pts_world = pts_world[labels == best_label]

    if len(pts_world) < MIN_CLUSTER_PTS:
        return None

    center = np.median(pts_world, axis=0)
    return center.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, choices=space_choices(),
                        help="Space (must be registered in spaces.json).")
    parser.add_argument("--out-dir", default=None,
                        help="Directory containing metadata.json "
                             "(default: spaces.json out_dir).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute world_pos for every proposal, ignoring existing values.")
    args = parser.parse_args()
    args.proposer = "fastsam"   # fastsam-only build
    space_paths = space(args.space)
    da3_dir       = space_paths["da3"]
    cameras_json  = space_paths["cameras"]
    out_dir = Path(args.out_dir) if args.out_dir else space(args.space)["out_dir"]
    METADATA_FILE = out_dir / "metadata.json"
    print(f"[backproject] space={args.space} da3={da3_dir}")
    print(f"[backproject] Using {METADATA_FILE}")

    if not METADATA_FILE.exists():
        print(f"[backproject] ERROR: {METADATA_FILE} not found — run 02_embed.py first")
        return

    metadata = json.loads(METADATA_FILE.read_text())
    print(f"[backproject] {len(metadata)} proposals to backproject")

    cams = json.loads(cameras_json.read_text())
    cam_by_id = {c["id"]: c for c in cams}

    success = 0
    fail = 0
    cached_npz = {}   # frame_idx → (depth, conf, K)

    for i, item in enumerate(metadata):
        if "world_pos" in item and not args.force:
            success += 1
            continue

        frame_idx = item["frame_idx"]
        if frame_idx not in cached_npz:
            npz = load_npz(da3_dir, frame_idx)
            if npz is None:
                item["world_pos"] = None
                fail += 1
                continue
            cached_npz[frame_idx] = npz
            if len(cached_npz) > 50:    # keep cache small
                oldest = next(iter(cached_npz))
                del cached_npz[oldest]

        depth, conf, K = cached_npz[frame_idx]
        cam = cam_by_id.get(item["view_id"])
        if cam is None:
            item["world_pos"] = None
            fail += 1
            continue

        wp = backproject(item["bbox"], depth, conf, K, cam["R"], cam["pos"])
        item["world_pos"] = wp
        if wp is not None:
            success += 1
        else:
            fail += 1

        if (i + 1) % 10000 == 0:
            METADATA_FILE.write_text(json.dumps(metadata))
            print(f"[backproject] {i+1}/{len(metadata)} | success={success} fail={fail}")

    METADATA_FILE.write_text(json.dumps(metadata))
    print(f"\n[backproject] Done.")
    print(f"  Success : {success}")
    print(f"  Failed  : {fail}  (unreliable depth / missing npz)")
    print(f"  Output  : {METADATA_FILE}")


if __name__ == "__main__":
    main()
