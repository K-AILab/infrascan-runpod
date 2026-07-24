"""Reproject the scanner point cloud into each training view to get a per-view SPARSE,
metric, multi-view-consistent depth signal (no monocular scale drift) — the "Tier 3"
depth lever from SHINHAN_SPLATFACTO_PLAN.md §3.

For each frame: world2camera via nerfstudio's OWN get_viewmat (so this uses the exact
same camera convention gsplat's rasterizer/rendered-depth uses — no convention-mismatch
risk), project points through the pinhole intrinsics, z-buffer to the nearest point per
pixel (torch scatter_reduce amin). Output: depth_scanner/frame_XXXXXX.npz per view
({"depth": (H,W) float32, "mask": (H,W) bool}).

CPU-only by design (this is offline, one-time preprocessing — must not contend with a
concurrent GPU training job).

Usage:
  python reproject_scanner_depth.py --selftest
  python reproject_scanner_depth.py --data <dataset dir with transforms.json> \
      --pointcloud <scanner pointcloud.ply> --out <output dir> [--every N] [--overlay K]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch


def get_viewmat(c2w):
    """Identical to nerfstudio.models.splatfacto.get_viewmat (reimplemented here so this
    script has no nerfstudio/CUDA import dependency and can run standalone on CPU)."""
    R = c2w[:, :3, :3]
    T = c2w[:, :3, 3:4]
    R = R * torch.tensor([[[1., -1., -1.]]])
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4)
    viewmat[:, 3, 3] = 1.0
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


def project_and_zbuffer(points_world, viewmat, fx, fy, cx, cy, W, H, near=0.02):
    """points_world: (N,3) torch tensor. Returns (depth (H,W), mask (H,W))."""
    ones = torch.ones(points_world.shape[0], 1)
    hom = torch.cat([points_world, ones], dim=1)          # (N,4)
    cam = (viewmat[0] @ hom.T).T[:, :3]                    # (N,3) camera space
    z = cam[:, 2]
    valid = z > near
    cam = cam[valid]; z = z[valid]
    u = fx * (cam[:, 0] / z) + cx
    v = fy * (cam[:, 1] / z) + cy
    iu = u.floor().long(); iv = v.floor().long()
    inb = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    iu, iv, z = iu[inb], iv[inb], z[inb]
    flat = (iv * W + iu).long()
    depth_flat = torch.full((H * W,), float("inf"))
    if flat.numel() > 0:
        depth_flat.scatter_reduce_(0, flat, z, reduce="amin", include_self=True)
    mask_flat = torch.isfinite(depth_flat)
    depth_flat = torch.where(mask_flat, depth_flat, torch.zeros_like(depth_flat))
    return depth_flat.view(H, W).numpy(), mask_flat.view(H, W).numpy()


def _selftest():
    # Identity c2w (nerfstudio/OpenGL convention: camera looks down -Z, +Y up, +X right).
    c2w = torch.eye(4)[None]
    vm = get_viewmat(c2w)
    fx = fy = 100.0; cx = cy = 50.0; W = H = 100

    # A point 3 units in front of the camera, dead-centre -> expect z=3, pixel (cx,cy).
    pts = torch.tensor([[0.0, 0.0, -3.0]])
    depth, mask = project_and_zbuffer(pts, vm, fx, fy, cx, cy, W, H)
    iy, ix = int(cy), int(cx)
    print(f"centre point: depth[{iy},{ix}]={depth[iy,ix]:.3f} (expect 3.0) mask={mask[iy,ix]}")
    assert mask[iy, ix] and abs(depth[iy, ix] - 3.0) < 1e-4

    # A point shifted +1 in X at z=-3 -> expect u = fx*1/3+cx = 133.33 -> outside a 100-wide
    # image (correctly culled). Shift by +0.3 instead -> u = 100*0.3/3+50 = 60.
    pts2 = torch.tensor([[0.3, 0.0, -3.0]])
    depth2, mask2 = project_and_zbuffer(pts2, vm, fx, fy, cx, cy, W, H)
    print(f"offset point lands at u=60? mask[50,60]={mask2[50,60]} depth={depth2[50,60]:.3f}")
    assert mask2[50, 60] and abs(depth2[50, 60] - 3.0) < 1e-4

    # Z-buffer: two points at the same pixel, nearer one (z=2) must win over farther (z=5).
    pts3 = torch.tensor([[0.0, 0.0, -5.0], [0.0, 0.0, -2.0]])
    depth3, mask3 = project_and_zbuffer(pts3, vm, fx, fy, cx, cy, W, H)
    print(f"z-buffer picks nearest: depth[{iy},{ix}]={depth3[iy,ix]:.3f} (expect 2.0, not 5.0)")
    assert abs(depth3[iy, ix] - 2.0) < 1e-4

    # A point behind the camera (z would be negative) must be culled entirely.
    pts4 = torch.tensor([[0.0, 0.0, 5.0]])   # behind: +Z is backward in this c2w convention
    depth4, mask4 = project_and_zbuffer(pts4, vm, fx, fy, cx, cy, W, H)
    print(f"behind-camera point culled: any valid? {mask4.any()} (expect False)")
    assert not mask4.any()

    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", type=Path)
    ap.add_argument("--pointcloud", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame (debug speed)")
    a = ap.parse_args()

    if a.selftest:
        _selftest(); return

    assert a.data and a.pointcloud and a.out, "need --data --pointcloud --out"
    from plyfile import PlyData
    tj = json.loads((a.data / "transforms.json").read_text())
    frames = tj["frames"][::a.every]
    top = {k: tj[k] for k in ("fl_x", "fl_y", "cx", "cy", "w", "h") if k in tj}

    print(f"loading point cloud {a.pointcloud} ...")
    ply = PlyData.read(str(a.pointcloud))
    v = ply["vertex"]
    pts_world = torch.from_numpy(
        np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
    )
    print(f"{len(pts_world):,} points loaded")

    a.out.mkdir(parents=True, exist_ok=True)
    import time
    t_start = time.time()
    for i, fr in enumerate(frames):
        fx = fr.get("fl_x", top.get("fl_x")); fy = fr.get("fl_y", top.get("fl_y"))
        cx = fr.get("cx", top.get("cx")); cy = fr.get("cy", top.get("cy"))
        W = fr.get("w", top.get("w")); H = fr.get("h", top.get("h"))
        c2w = torch.tensor(fr["transform_matrix"], dtype=torch.float32)[None]
        vm = get_viewmat(c2w)
        depth, mask = project_and_zbuffer(pts_world, vm, fx, fy, cx, cy, W, H)
        stem = Path(fr["file_path"]).stem
        np.savez_compressed(a.out / f"{stem}.npz",
                             depth=depth.astype(np.float32), mask=mask.astype(bool))
        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(frames) - i - 1) / rate if rate > 0 else float("inf")
            print(f"  {i+1}/{len(frames)}  ({rate:.2f} frames/s, ETA {eta/60:.1f} min)"
                  f"  hit-rate this frame: {mask.mean()*100:.2f}%", flush=True)
    print(f"done: {len(frames)} frames -> {a.out}  ({(time.time()-t_start)/60:.1f} min total)")


if __name__ == "__main__":
    main()
