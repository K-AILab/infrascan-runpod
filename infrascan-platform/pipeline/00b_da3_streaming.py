"""Stage `da3` for video-input spaces.

Wraps da3_streaming.py — it estimates depth + camera poses jointly from
the perspective views, then we reshape its outputs into the layout the
rest of the pipeline expects:

    data/<slug>/depth/frame_<i>.npz   from results_output/*.npz
    data/<slug>/cameras.json          built from camera_poses.txt + views
    data/<slug>/pointcloud.ply        from pcd/combined_pcd.ply

Usage:
    python pipeline/00b_da3_streaming.py --space <slug>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Use the platform's _paths.py shim
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import space   # noqa: E402


REPO = Path(__file__).resolve().parents[1]
DA3_STREAMING_DIR = Path(__file__).resolve().parent / "da3_streaming"


def run_da3_streaming(views_dir: Path, out_dir: Path) -> None:
    cfg = DA3_STREAMING_DIR / "configs" / "base_config.yaml"
    if not cfg.exists():
        # pick the first available
        cfg = next(DA3_STREAMING_DIR.glob("configs/*.yaml"))
    cmd = [
        sys.executable, str(DA3_STREAMING_DIR / "da3_streaming.py"),
        "--image_dir", str(views_dir),
        "--config", str(cfg),
        "--output_dir", str(out_dir),
    ]
    print("[da3] " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    # da3_streaming imports loop_utils.* — needs its own dir on PYTHONPATH
    env["PYTHONPATH"] = str(DA3_STREAMING_DIR) + ":" + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env, cwd=str(DA3_STREAMING_DIR))


def parse_view_name(name: str):
    """Filename like '000000_pz000_y030_normal' → (frame, pitch, yaw)."""
    m = re.match(r"(\d+)_pz(\d+)_y(\d+)_normal", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def reshape_outputs(slug: str, paths: dict, da3_out: Path) -> None:
    """Promote da3_streaming outputs into the per-space layout."""
    data_root: Path = paths["data_root"]

    # 1. Move per-view depth NPZs from results_output/ into depth/
    src_npz_dir = da3_out / "results_output"
    dst_npz_dir = paths["da3"]   # data/<slug>/depth
    dst_npz_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(src_npz_dir.glob("frame_*.npz"))
    if not npz_files:
        sys.exit("[da3] no per-frame .npz produced by da3_streaming")
    print(f"[da3] linking {len(npz_files)} depth files → {dst_npz_dir}")
    for src in npz_files:
        dst = dst_npz_dir / src.name
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())

    # 2. Build cameras.json from camera_poses.txt
    poses_txt = da3_out / "camera_poses.txt"
    if not poses_txt.exists():
        sys.exit(f"[da3] camera_poses.txt missing at {poses_txt}")

    poses = []
    with poses_txt.open() as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) != 16:
                continue
            mat = np.array(vals, dtype=np.float64).reshape(4, 4)
            poses.append(mat)

    # Pair poses with the EXACT order da3_streaming processed them in
    # (serpentine, not alphabetical). da3_streaming records it in img_list.txt;
    # poses[k] and frame_<k>.npz both correspond to that file.
    order_file = da3_out / "img_list.txt"
    if not order_file.exists():
        sys.exit(
            f"[da3] {order_file} missing — cannot recover the pose↔image order. "
            f"Delete {da3_out / 'results_output'} and re-run so da3_streaming "
            f"re-emits img_list.txt."
        )
    view_names = [
        Path(l).stem for l in order_file.read_text().splitlines() if l.strip()
    ]
    if len(view_names) != len(poses):
        print(f"[da3] WARNING views={len(view_names)} poses={len(poses)} — truncating to min",
              file=sys.stderr)
    n = min(len(view_names), len(poses))

    cameras = []
    for view_id in range(n):
        name = view_names[view_id]
        T = poses[view_id]
        R = T[:3, :3].tolist()
        pos = T[:3, 3].tolist()
        parsed = parse_view_name(name)
        if parsed:
            frame, pitch, yaw = parsed
        else:
            frame, pitch, yaw = view_id, 0, 0
        cameras.append({
            "id": view_id,
            "pos": pos,
            "R": R,
            "pano": f"panos/{name}.jpg",
            "xy": [pos[0], pos[2]],
            "frame": frame,
            "pitch": pitch,
            "yaw": yaw,
        })

    cameras_json = paths["cameras"]
    cameras_json.write_text(json.dumps(cameras))
    print(f"[da3] wrote {cameras_json} ({len(cameras)} entries)")

    # 3. Save the combined point cloud
    src_ply = da3_out / "pcd" / "combined_pcd.ply"
    dst_ply = paths["pointcloud"]
    if src_ply.exists():
        if dst_ply.exists() or dst_ply.is_symlink():
            dst_ply.unlink()
        dst_ply.symlink_to(src_ply.resolve())
        print(f"[da3] linked pointcloud {dst_ply} → {src_ply}")
    else:
        print(f"[da3] WARNING combined_pcd.ply missing at {src_ply}", file=sys.stderr)

    # 4. intrinsics.json — if da3_streaming wrote intrinsic.txt, convert.
    # DA3 writes 4 floats per line (fx fy cx cy) in the version we ship;
    # legacy runs used 9 floats (row-major 3x3 K). Handle both.
    intrinsic_txt = da3_out / "intrinsic.txt"
    if intrinsic_txt.exists():
        K = None
        with intrinsic_txt.open() as f:
            line = f.readline().strip()
            vals = [float(x) for x in line.split()]
            if len(vals) == 4:
                fx, fy, cx, cy = vals
                K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
            elif len(vals) == 9:
                K = np.array(vals).reshape(3, 3).tolist()
        if K:
            paths["intrinsics"].write_text(json.dumps({
                "K": K,
                "fx": K[0][0], "fy": K[1][1], "cx": K[0][2], "cy": K[1][2],
                "image_size": [504, 504],
                "width": 504, "height": 504,
            }))
            print(f"[da3] wrote {paths['intrinsics']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    paths = space(args.space)
    views_dir: Path = paths["views"]
    if not views_dir.exists() or not any(views_dir.iterdir()):
        sys.exit(f"[da3] views/ is empty for space {args.space}: {views_dir}")

    da3_out = paths["data_root"] / "_da3_streaming"
    da3_out.mkdir(parents=True, exist_ok=True)

    if not args.resume or not (da3_out / "results_output").exists():
        run_da3_streaming(views_dir, da3_out)
    else:
        print(f"[da3] reusing existing run at {da3_out}")

    reshape_outputs(args.space, paths, da3_out)


if __name__ == "__main__":
    main()
