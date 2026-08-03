#!/usr/bin/env python3
"""
Offline step 0b: Generate DA3 depth maps for an icc-style space.

Writes one `.npz` per view into `<space.da3>/frame_<view_id>.npz`, matching
the format consumed by `03_backproject.py`:

    keys: depth (H, W), conf (H, W), intrinsics (3, 3)

Views are processed scanpoint-by-scanpoint (36 views per scanpoint by
default — 12 yaws × 3 pitches), so DA3 can exploit multi-view consistency
within each location. Camera poses from `cameras.json` are passed to DA3
so the predicted depth is aligned to the input world scale.

Usage:
    conda activate sam3      # or whatever env has Depth-Anything-3 installed
    python 00b_gen_da3.py --space icc1
    python 00b_gen_da3.py --space icc1 --resume         # skip frames already on disk
    python 00b_gen_da3.py --space icc1 --max-scanpoints 5   # smoke test

Note: only needed for spaces that don't already ship DA3 outputs (i.e. not
v1/v2). After running this, `03_backproject.py --space icc1` works.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from _paths import space, space_choices  # noqa: E402

# Depth-Anything-3 lives outside this repo. Default path = the conventional
# external/ checkout written by `scripts/install_da3.sh`; override with the
# DA3_SRC env var or `--da3-src` CLI.
_REPO = Path(__file__).resolve().parents[1]
DA3_SRC_DEFAULT = Path(os.environ.get(
    "DA3_SRC", _REPO / "external" / "depth-anything-3" / "src"
))
if str(DA3_SRC_DEFAULT) not in sys.path:
    sys.path.insert(0, str(DA3_SRC_DEFAULT))

VIEW_H = 504
VIEW_W = 504


def _parse_scanpoint(pano: str) -> int:
    """Extract scanpoint id from a pano filename like 'panos/000049_pz000_y060_normal.jpg'."""
    m = re.search(r"(\d+)_pz", Path(pano).name)
    return int(m.group(1)) if m else -1


def _build_extrinsics(R, pos) -> np.ndarray:
    """cam-to-world (R, pos) → world-to-camera 4x4 extrinsics for DA3."""
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    pos = np.asarray(pos, dtype=np.float32).reshape(3)
    Rwc = R.T                            # rotation world → cam
    twc = -Rwc @ pos                     # translation world → cam
    ext = np.eye(4, dtype=np.float32)
    ext[:3, :3] = Rwc
    ext[:3, 3] = twc
    return ext


def _load_intrinsics(space_paths) -> np.ndarray:
    """Read intrinsics.json next to cameras.json, fall back to fx=fy=H/2."""
    K_path = space_paths["cameras"].parent / "intrinsics.json"
    if K_path.exists():
        d = json.loads(K_path.read_text())
        if "matrix_K" in d:
            return np.asarray(d["matrix_K"], dtype=np.float32)
        return np.asarray(
            [[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]],
            dtype=np.float32,
        )
    # Fallback for spaces without a sidecar intrinsics file.
    f = VIEW_H / 2.0
    return np.asarray([[f, 0, VIEW_W / 2], [0, f, VIEW_H / 2], [0, 0, 1]], dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--space", required=True, choices=space_choices(),
                   help="Space (must be registered in spaces.json).")
    p.add_argument("--out-dir", default=None,
                   help="Override output directory (default: <data_root>/depth).")
    p.add_argument("--model-name", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                   help="HF hub id (default: depth-anything/DA3NESTED-GIANT-LARGE-1.1, "
                        "already cached locally). Alternatives: "
                        "depth-anything/DA3NESTED-GIANT-LARGE")
    p.add_argument("--resume", action="store_true",
                   help="Skip view ids whose npz already exists on disk")
    p.add_argument("--max-scanpoints", type=int, default=0,
                   help="Process at most N scanpoints (0 = all) — smoke test helper")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    space_paths = space(args.space)
    views_dir   = space_paths["views"]
    cameras_p   = space_paths["cameras"]
    out_dir     = Path(args.out_dir) if args.out_dir else space_paths["da3"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[da3] space={args.space} views={views_dir} → {out_dir}")
    if not views_dir.exists():
        sys.exit(f"[da3] views_dir does not exist: {views_dir}")
    if not cameras_p.exists():
        sys.exit(f"[da3] cameras.json missing: {cameras_p}")

    cameras = json.loads(cameras_p.read_text())
    K = _load_intrinsics(space_paths)
    print(f"[da3] {len(cameras)} views, K = fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    by_sp: dict[int, list[dict]] = defaultdict(list)
    for cam in cameras:
        by_sp[_parse_scanpoint(cam["pano"])].append(cam)
    sp_ids = sorted(by_sp.keys())
    if args.max_scanpoints > 0:
        sp_ids = sp_ids[: args.max_scanpoints]
    print(f"[da3] grouped into {len(sp_ids)} scanpoints ({len(by_sp[sp_ids[0]])} views in sp{sp_ids[0]})")

    # Lazy import — heavy dep
    from depth_anything_3.api import DepthAnything3
    print(f"[da3] Loading {args.model_name} on {args.device} ...")
    model = DepthAnything3.from_pretrained(args.model_name).to(args.device).eval()
    print("[da3] model ready")

    total_frames = 0
    t0 = time.time()
    for sp_idx, sp in enumerate(sp_ids):
        cams = by_sp[sp]
        # Filter out frames already on disk
        if args.resume:
            cams = [c for c in cams if not (out_dir / f"frame_{c['id']}.npz").exists()]
        if not cams:
            continue

        image_paths = [str(views_dir / Path(c["pano"]).name) for c in cams]
        exts = np.stack([_build_extrinsics(c["R"], c["pos"]) for c in cams], axis=0)
        ks   = np.broadcast_to(K, (len(cams), 3, 3)).copy()

        # DA3 multi-view inference: same scene, multiple poses → metric-consistent depth.
        pred = model.inference(
            image=image_paths,
            extrinsics=exts,
            intrinsics=ks,
            align_to_input_ext_scale=True,
            process_res=VIEW_H,
            process_res_method="upper_bound_resize",
        )

        # prediction.depth: (N, H, W). Persist one npz per view.
        depths = np.asarray(pred.depth, dtype=np.float32)
        confs  = (np.asarray(pred.conf,  dtype=np.float32)
                  if pred.conf is not None
                  else np.ones_like(depths, dtype=np.float32))
        # Per-view K (DA3 may rescale slightly if process_res changes shape)
        K_out  = np.asarray(pred.intrinsics, dtype=np.float32) if pred.intrinsics is not None else None

        for i, c in enumerate(cams):
            out_npz = out_dir / f"frame_{c['id']}.npz"
            np.savez(
                out_npz,
                depth=depths[i],
                conf=confs[i],
                intrinsics=K_out[i] if K_out is not None else K,
            )
            total_frames += 1

        if (sp_idx + 1) % 10 == 0 or sp_idx == len(sp_ids) - 1:
            elapsed = time.time() - t0
            rate = total_frames / max(elapsed, 1e-3)
            print(f"[da3] sp {sp_idx+1}/{len(sp_ids)} · "
                  f"{total_frames} frames · {rate:.1f} frame/s · "
                  f"{elapsed/60:.1f} min")

    print(f"[da3] done — wrote {total_frames} frames in {(time.time()-t0)/60:.1f} min")
    print(f"[da3] next: python 03_backproject.py --space {args.space}")


if __name__ == "__main__":
    main()
