"""Generate a top-down floor-plan PNG from a space's point cloud.

Reads `data/<space>/pointcloud.ply` (or the downsampled web PLY if available),
picks the two axes with the largest spread (horizontal plane), rasterizes the
points into a 2D density image, and writes:

    ui/_spaces/<space>/topdown/topdown.png
    ui/_spaces/<space>/topdown/bounds.json

The viewer reads bounds.json to map world (x, y) → pixel (px, py) for the
dynamic dots layer.

Run once per space (or any time the PLY is refreshed):
    python pipeline/gen_topdown.py --space <name>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).parent))
from _paths import space, space_choices  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SIZE = 1024            # output is up to SIZE × SIZE
DEFAULT_BG   = (12, 12, 12)    # near-black, matches viewer dark theme
DEFAULT_FG   = (160, 230, 240) # cyan-ish for the room
GAMMA        = 0.5             # density → brightness curve (low γ = boost faint regions)


def pick_axes(xyz: np.ndarray) -> tuple[int, int]:
    """Return indices of the two axes with the largest spread."""
    spreads = xyz.max(axis=0) - xyz.min(axis=0)
    order = np.argsort(-spreads)   # descending
    return int(order[0]), int(order[1])


def rasterize(xyz: np.ndarray, ax_u: int, ax_v: int, size: int) -> tuple[Image.Image, dict]:
    """Bin (u, v) into a size×size density image; preserve aspect ratio."""
    u = xyz[:, ax_u]
    v = xyz[:, ax_v]

    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())

    u_range = u_max - u_min
    v_range = v_max - v_min

    # Pick output dims so aspect is preserved and the longer side = size.
    if u_range >= v_range:
        out_w = size
        out_h = max(1, int(round(size * v_range / u_range)))
    else:
        out_h = size
        out_w = max(1, int(round(size * u_range / v_range)))

    # Histogram with one bin per pixel column/row.
    H, _, _ = np.histogram2d(
        u, v,
        bins=(out_w, out_h),
        range=[[u_min, u_max], [v_min, v_max]],
    )

    # Density → 0..1, then gamma-corrected.
    if H.max() > 0:
        H = H / H.max()
    H = np.power(H, GAMMA)

    # Compose RGB: lerp BG → FG by density. Transpose so origin is top-left.
    bg = np.array(DEFAULT_BG, dtype=np.float32)
    fg = np.array(DEFAULT_FG, dtype=np.float32)
    # H is (out_w, out_v) in (u, v); we want image (rows = v, cols = u).
    dens = H.T[::-1]    # flip v so larger v draws toward the top
    rgb = bg[None, None, :] + (fg - bg)[None, None, :] * dens[:, :, None]
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")

    bounds = {
        "axis_u": ax_u,
        "axis_v": ax_v,
        "u_min": u_min,
        "u_max": u_max,
        "v_min": v_min,
        "v_max": v_max,
        "width":  img.size[0],
        "height": img.size[1],
        # The PNG draws +u right, +v UP (we flipped above), so the JS
        # mapping is straightforward — see viewer/topdown.js worldToPx().
        "v_flipped": True,
    }
    return img, bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, choices=space_choices(),
                    help="Space (must be registered in spaces.json).")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE,
                    help="longer side of output PNG, in pixels")
    ap.add_argument("--ply", default=None,
                    help="Override PLY path (default: prefer downsampled "
                         "ui asset, fall back to the source pointcloud).")
    args = ap.parse_args()

    sp = space(args.space)
    if args.ply:
        ply_path = Path(args.ply)
    else:
        # Prefer the (smaller) downsampled web PLY if it already exists,
        # otherwise fall back to the raw source pointcloud.
        candidates = [
            REPO / "ui" / "_spaces" / args.space / "Data_" / "downsampled_web.ply",
            sp["pointcloud"],
        ]
        ply_path = next((p for p in candidates if p.exists()), candidates[-1])
    out_dir = REPO / "ui" / "_spaces" / args.space / "topdown"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[topdown:{args.space}] reading {ply_path}")
    ply = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float32)
    print(f"[topdown:{args.space}] {len(xyz):,} points")

    ax_u, ax_v = pick_axes(xyz)
    print(f"[topdown:{args.space}] axes: u=axis{ax_u}  v=axis{ax_v}  "
          f"(skipping vertical axis{3 - ax_u - ax_v})")

    img, bounds = rasterize(xyz, ax_u, ax_v, args.size)
    img.save(out_dir / "topdown.png", optimize=True)
    (out_dir / "bounds.json").write_text(json.dumps(bounds, indent=2))
    print(f"[topdown:{args.space}] wrote {out_dir/'topdown.png'}  ({img.size[0]}×{img.size[1]})")
    print(f"[topdown:{args.space}] wrote {out_dir/'bounds.json'}")


if __name__ == "__main__":
    main()
