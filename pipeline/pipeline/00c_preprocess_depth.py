#!/usr/bin/env python3
"""
Preprocess NPZ depth files into browser-friendly binary format and generate
a scanpoints.json manifest for the Matterport-style pano viewer.

Binary format per view (.bin):
  - float32 xyz: 504*504*3 values (3,048,192 bytes)
  - uint8  valid: 504*504 values   (254,016 bytes)
  Total: 3,302,208 bytes

scanpoints.json groups 372 views into 31 scanpoints (12 yaw views each).

Usage:
    python preprocess_depth.py \
        --depth_dir  data/experiments/v2_cubicmap_rgb/depth \
        --rgb_dir    data/experiments/v2_cubicmap_rgb/rgb \
        --poses_txt  data/experiments/v2_cubicmap_rgb/da3_output/camera_poses.txt \
        --output_dir data/experiments/v2_cubicmap_rgb/web
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def parse_view_name(name: str):
    """Extract frame_id and yaw from a filename like 000000_pz000_y030_normal."""
    m = re.match(r"(\d{6})_pz\d+_y(\d{3})_normal", name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def detect_yaw_calibration(depth_dir, npz_files):
    """Auto-detect the mapping from filename yaw to scanner-frame yaw.

    Returns (offset, direction) such that:
        scanner_yaw = (offset + direction * filename_yaw) % 360
    where direction is +1 (normal) or -1 (reversed).
    """
    # Sample y000 and y090 from the first scanpoint to determine offset + direction
    frame0 = None
    for f in npz_files:
        fid, yaw = parse_view_name(f.stem)
        if fid is not None:
            frame0 = fid
            break
    if frame0 is None:
        return 0.0, 1  # fallback

    scanner_yaws = {}
    for fy in [0, 90]:
        fname = depth_dir / f"{frame0}_pz000_y{fy:03d}_normal.npz"
        if not fname.exists():
            continue
        d = np.load(fname)
        if not d["valid"][252, 252]:
            continue
        direction = d["xyz"][252, 252] - d["scanner_pos"]
        scanner_yaw = np.degrees(np.arctan2(direction[0], direction[1])) % 360
        scanner_yaws[fy] = scanner_yaw

    if 0 not in scanner_yaws:
        return 0.0, 1  # fallback

    offset_if_normal = scanner_yaws[0]  # scanner = offset + 1*0
    if 90 in scanner_yaws:
        # Check if direction is +1 or -1
        predicted_normal = (offset_if_normal + 90) % 360
        predicted_reverse = (offset_if_normal - 90) % 360
        actual = scanner_yaws[90]
        err_normal = min(abs(actual - predicted_normal), 360 - abs(actual - predicted_normal))
        err_reverse = min(abs(actual - predicted_reverse), 360 - abs(actual - predicted_reverse))
        if err_reverse < err_normal:
            return round(offset_if_normal, 1), -1
    return round(offset_if_normal, 1), 1


def build_serpentine_mapping(num_scanpoints, views_per_sp=12):
    """Build DA3 frame index -> (scanpoint_idx, yaw_index) mapping.

    DA3 processes views in serpentine order:
      even scanpoints: y000, y030, ..., y330 (forward)
      odd  scanpoints: y330, y300, ..., y000 (reverse)
    """
    mapping = {}  # da3_frame_idx -> (scanpoint_idx, yaw_deg)
    yaws_forward = list(range(0, 360, 30))  # [0, 30, 60, ..., 330]
    yaws_reverse = list(reversed(yaws_forward))

    for sp_idx in range(num_scanpoints):
        yaws = yaws_forward if sp_idx % 2 == 0 else yaws_reverse
        for v_idx, yaw in enumerate(yaws):
            da3_idx = sp_idx * views_per_sp + v_idx
            mapping[da3_idx] = (sp_idx, yaw)

    return mapping


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth_dir", default=None,
                   help="Path to GT depth dir (optional for video-based experiments)")
    p.add_argument("--rgb_dir", required=True)
    p.add_argument("--poses_txt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--da3_results", default=None,
                   help="Path to DA3 results_output/ dir (for dense depth)")
    args = p.parse_args()

    depth_dir = Path(args.depth_dir) if args.depth_dir else None
    rgb_dir = Path(args.rgb_dir)
    output_dir = Path(args.output_dir)

    # Read camera poses
    poses_lines = [
        ln.strip()
        for ln in Path(args.poses_txt).read_text().splitlines()
        if ln.strip()
    ]

    rgb_files = sorted(rgb_dir.glob("*.jpg"))

    has_gt_depth = depth_dir is not None and depth_dir.exists()

    if has_gt_depth:
        bin_dir = output_dir / "depth_bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        npz_files = sorted(depth_dir.glob("*.npz"))
        assert len(npz_files) == len(rgb_files) == len(poses_lines), (
            f"Mismatch: {len(npz_files)} npz, {len(rgb_files)} rgb, {len(poses_lines)} poses"
        )
        yaw_offset, yaw_direction = detect_yaw_calibration(depth_dir, npz_files)
    else:
        npz_files = []
        # da3 poses are in serpentine order, not the sorted glob. Rebuild
        # rgb_files from the recorded order so rgb_files[i] ↔ poses_lines[i].
        order_file = Path(args.poses_txt).parent / "img_list.txt"
        rgb_files = [
            rgb_dir / l.strip()
            for l in order_file.read_text().splitlines()
            if l.strip()
        ]
        assert len(rgb_files) == len(poses_lines), (
            f"Mismatch: {len(rgb_files)} rgb, {len(poses_lines)} poses"
        )
        yaw_offset, yaw_direction = 0.0, 1
    print(f"Yaw calibration: scanner_yaw = ({yaw_offset} + {yaw_direction} * filename_yaw) % 360")

    # Group views by scanpoint (frame_id)
    scanpoints = {}  # frame_id -> list of view dicts
    scanpoint_order = []  # preserve ordering

    if has_gt_depth:
        # Process GT depth
        for i, npz_path in enumerate(npz_files):
            stem = npz_path.stem
            frame_id, yaw = parse_view_name(stem)
            if frame_id is None:
                print(f"  SKIP unrecognized: {stem}")
                continue

            data = np.load(npz_path)
            xyz = data["xyz"].astype(np.float32)
            valid = data["valid"].astype(np.uint8)
            scanner_pos = data["scanner_pos"].astype(np.float32)

            # GT depth is bottom-to-top (OpenGL) — flip to match image orientation
            xyz = np.flip(xyz, axis=0).copy()
            valid = np.flip(valid, axis=0).copy()

            bin_name = f"{frame_id}_pz000_y{yaw:03d}.bin"
            with open(bin_dir / bin_name, "wb") as f:
                f.write(xyz.tobytes())
                f.write(valid.tobytes())

            vals = np.fromstring(poses_lines[i], sep=" ").reshape(4, 4)
            slam_pos = vals[:3, 3].tolist()

            view = {
                "yaw": yaw,
                "file": stem,
                "rgb": f"panos/{rgb_files[i].name}",
                "depth_bin": f"depth_bin/{bin_name}",
            }

            if frame_id not in scanpoints:
                scanpoints[frame_id] = {
                    "frame_id": frame_id,
                    "scanner_pos": scanner_pos.tolist(),
                    "slam_pos": slam_pos,
                    "views": [],
                }
                scanpoint_order.append(frame_id)

            scanpoints[frame_id]["views"].append(view)
    else:
        # No GT depth — build scanpoints from RGB files + poses only
        for i, rgb_path in enumerate(rgb_files):
            stem = rgb_path.stem
            frame_id, yaw = parse_view_name(stem)
            if frame_id is None:
                continue

            vals = np.fromstring(poses_lines[i], sep=" ").reshape(4, 4)
            slam_pos = vals[:3, 3].tolist()

            view = {
                "yaw": yaw,
                "file": stem,
                "rgb": f"panos/{rgb_path.name}",
            }

            if frame_id not in scanpoints:
                scanpoints[frame_id] = {
                    "frame_id": frame_id,
                    "scanner_pos": slam_pos,  # use SLAM pos as scanner pos
                    "slam_pos": slam_pos,
                    "views": [],
                }
                scanpoint_order.append(frame_id)

            scanpoints[frame_id]["views"].append(view)

    # ── DA3 dense depth export ──────────────────────────────────────────
    has_da3 = False
    da3_intrinsics = None
    if args.da3_results:
        da3_dir = Path(args.da3_results)
        da3_bin_dir = output_dir / "da3_depth_bin"
        da3_bin_dir.mkdir(parents=True, exist_ok=True)

        num_sp = len(scanpoint_order)
        serpentine = build_serpentine_mapping(num_sp)

        # Build reverse lookup: (frame_id, yaw) -> scanpoint dict key
        fid_yaw_to_key = {}
        for fid in scanpoint_order:
            for v in scanpoints[fid]["views"]:
                fid_yaw_to_key[(fid, v["yaw"])] = fid

        da3_count = 0
        for da3_idx in sorted(serpentine.keys()):
            sp_idx, yaw = serpentine[da3_idx]
            fid = scanpoint_order[sp_idx]

            da3_path = da3_dir / f"frame_{da3_idx}.npz"
            if not da3_path.exists():
                continue

            da3_data = np.load(da3_path)
            depth = da3_data["depth"].astype(np.float32)  # (504, 504), dense
            conf = da3_data["conf"].astype(np.float32)     # (504, 504)

            if da3_intrinsics is None:
                da3_intrinsics = da3_data["intrinsics"].astype(np.float32).tolist()

            # Write: float32 depth (504*504*4 bytes) + float32 conf (504*504*4 bytes)
            bin_name = f"{fid}_pz000_y{yaw:03d}_da3.bin"
            with open(da3_bin_dir / bin_name, "wb") as f:
                f.write(depth.tobytes())
                f.write(conf.tobytes())

            # Add da3 depth path to the matching view
            for v in scanpoints[fid]["views"]:
                if v["yaw"] == yaw:
                    v["da3_depth_bin"] = f"da3_depth_bin/{bin_name}"
                    break

            da3_count += 1

        has_da3 = da3_count > 0
        print(f"Wrote {da3_count} DA3 depth binaries to {da3_bin_dir}")

    # Build manifest
    manifest = {
        "yaw_offset": float(yaw_offset),
        "yaw_direction": int(yaw_direction),
        "has_da3_depth": has_da3,
        "scanpoints": [],
    }
    if da3_intrinsics:
        manifest["da3_intrinsics"] = da3_intrinsics

    for idx, fid in enumerate(scanpoint_order):
        sp = scanpoints[fid]
        sp["id"] = idx
        # Sort views by yaw
        sp["views"].sort(key=lambda v: v["yaw"])
        manifest["scanpoints"].append(sp)

    manifest_path = output_dir / "scanpoints.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {len(manifest['scanpoints'])} scanpoints to {manifest_path}")
    if has_gt_depth:
        print(f"Wrote {len(npz_files)} GT depth binaries to {bin_dir}")


if __name__ == "__main__":
    main()
