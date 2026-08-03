#!/usr/bin/env python
"""Score detected 3D boxes by how well their enclosed Gaussians form a
CLOSED SURFACE — a training-free geometric quality signal borrowed from
Gaussian-Det (Yan, Zheng, Duan; ICLR 2025; arXiv:2410.01404), specifically
its Theorem 1: the flux of a constant vector field through a closed
surface is exactly zero. A box that tightly and correctly encloses one
real object has Gaussians whose (outward-oriented) normals point in all
directions roughly evenly, so their flux nets close to zero. A box that's
misaligned, oversized, or a false positive tends to capture an incomplete/
open slice of surface (e.g. one side of a wall plus empty air), so the
normals don't cancel out and the flux is large.

Gaussian-Det itself is a fully supervised, TRAINED detector (VoteNet-style
proposal head, learned Closure Inferring Module) with no released code —
not something we can just drop in. This script borrows only the paper's
actual mathematical core (Theorem 1 / Eqn. 8-9, and the per-Gaussian
normal/area definitions in Section 3.2 and Appendix B.2) as a pure-
geometry, no-training-needed scoring pass on top of boxes this project's
OWN pipelines already produced (splat_analyzer, pipeline9, etc.) — not a
replacement for how those boxes get created, a filter on the output.

Validated directly on this splat before trusting it: a box KNOWN to be
wrong (a "door" detection that was actually a real plant, boxed too
loosely — see external/splat_analyzer/RUN_NOTES.md's 17th round) scored
|flux|=0.18, roughly 30x every KNOWN-correct box checked (table/light/
plant, |flux| 0.003-0.008). This script generalizes that one-off check
to every box in a boxes.json.

Usage:
  python closed_surface_flux.py --ply ../data/shinhan_hires_30k.ply \
    --boxes ../pipeline9/out/shinhan_space_splatanalyzer_derot_v17_boxes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat_normals import quat_to_rotmat  # noqa: E402 — read-only reuse


def load_splat_raw(ply_path: str, opacity_thresh: float = 0.1):
    """Per-Gaussian position, scale, UNORIENTED normal, and cross-sectional
    area (Eqn. 3 in Gaussian-Det: A = pi * s1*s2*s3 / min(s1,s2,s3)).
    Normals are deliberately left unoriented here — Gaussian-Det orients
    each Gaussian's normal relative to the SPECIFIC proposal/box it's
    being scored against (center -> Gaussian direction), not once
    globally, so orientation happens per-box in `box_flux` instead."""
    p = PlyData.read(ply_path)["vertex"]
    xyz = np.stack([p["x"], p["y"], p["z"]], axis=1).astype(np.float64)
    q = np.stack([p["rot_0"], p["rot_1"], p["rot_2"], p["rot_3"]], axis=1).astype(np.float64)
    scale = np.exp(np.stack([p["scale_0"], p["scale_1"], p["scale_2"]], axis=1).astype(np.float64))
    opacity = 1 / (1 + np.exp(-p["opacity"].astype(np.float64)))
    keep = opacity >= opacity_thresh
    xyz, q, scale = xyz[keep], q[keep], scale[keep]

    R = quat_to_rotmat(q)
    shortest_axis = np.argmin(scale, axis=1)
    normals = R[np.arange(len(R)), :, shortest_axis]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)

    area = np.pi * np.prod(scale, axis=1) / np.maximum(scale.min(axis=1), 1e-12)
    return xyz, scale, normals, area


def enclosed_mask(xyz, center, size, angle):
    """Indices of gaussians whose CENTER falls inside an oriented box,
    and their positions in the box's own local (unrotated) frame."""
    center = np.asarray(center, dtype=np.float64)
    c, s = np.cos(-angle), np.sin(-angle)
    r_inv = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    local = (xyz - center) @ r_inv.T
    half = np.asarray(size, dtype=np.float64) / 2
    inside = np.all(np.abs(local) <= half, axis=1)
    idx = np.where(inside)[0]
    return idx, local[idx]


def box_flux(xyz, normals, area, center, size, angle, T=None):
    """|flux| for one oriented box — low means a closed, coherent real
    surface inside it; high means an open/incomplete/misaligned one."""
    if T is None:
        T = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    center = np.asarray(center, dtype=np.float64)
    idx, _ = enclosed_mask(xyz, center, size, angle)
    if len(idx) == 0:
        return None, 0
    n = normals[idx].copy()
    to_gaussian = xyz[idx] - center
    flip = np.sum(n * to_gaussian, axis=1) < 0
    n[flip] *= -1
    flux = float(np.sum((T @ n.T) * area[idx]))
    return flux, len(idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ply", required=True, help="the ORIGINAL (not de-rotated) splat — "
                    "boxes.json is already in that frame")
    ap.add_argument("--boxes", required=True, help="a viewer-format boxes.json")
    ap.add_argument("--out", default=None, help="optional: write boxes.json annotated with flux/n_gaussians")
    ap.add_argument("--filtered-out", default=None,
                     help="optional: write a SECOND boxes.json with boxes rejected per "
                     "--min-gaussians/--max-flux removed")
    ap.add_argument("--min-gaussians", type=int, default=0,
                     help="reject boxes enclosing fewer than this many gaussians (0 = a box "
                     "floating in empty space — the clearest possible defect, no threshold-tuning "
                     "needed). Default 0 means no filtering.")
    ap.add_argument("--max-flux", type=float, default=None,
                     help="reject boxes with |flux| above this. Established from direct "
                     "crop-verified ground truth on this splat: confirmed-good boxes measured "
                     "0.003-0.04, a confirmed-bad box (oversized 'door' detection actually "
                     "seeing a plant) measured 0.18 — pick a value in that gap, e.g. 0.1. "
                     "Unset by default (no established general-purpose value yet).")
    args = ap.parse_args()

    xyz, _scale, normals, area = load_splat_raw(args.ply)
    print(f"[flux] {len(xyz):,} gaussians loaded from {args.ply}")

    data = json.loads(Path(args.boxes).read_text())
    scored = []
    for b in data["boxes"]:
        flux, n_gauss = box_flux(xyz, normals, area, b["center"], b["size"], b.get("angle", 0.0))
        scored.append((b["label"], abs(flux) if flux is not None else float("nan"), n_gauss, b))

    scored.sort(key=lambda t: t[1])
    print(f"\n{'label':10s} {'|flux|':>8s} {'n_gauss':>8s}")
    for label, flux, n_gauss, _ in scored:
        print(f"{label:10s} {flux:8.4f} {n_gauss:8d}")

    for (label, flux, n_gauss, b) in scored:
        b["flux"] = flux
        b["n_gaussians"] = n_gauss

    if args.out:
        Path(args.out).write_text(json.dumps(data, indent=2))
        print(f"\n[flux] wrote annotated boxes -> {args.out}")

    if args.filtered_out:
        kept_boxes, dropped = [], []

        def reject(flux, n_gauss):
            return n_gauss < args.min_gaussians or (
                args.max_flux is not None and not np.isnan(flux) and flux > args.max_flux)

        for (label, flux, n_gauss, b) in scored:
            if reject(flux, n_gauss):
                dropped.append((label, flux, n_gauss))
                continue
            kept_box = dict(b)
            kept_box.pop("flux", None)
            kept_box.pop("n_gaussians", None)
            kept_boxes.append(kept_box)

        Path(args.filtered_out).write_text(json.dumps({"boxes": kept_boxes}, indent=2))
        print(f"\n[flux] kept {len(kept_boxes)}/{len(scored)} boxes -> {args.filtered_out}")
        if dropped:
            print("[flux] dropped:")
            for label, flux, n_gauss in dropped:
                print(f"  {label:10s} |flux|={flux:.4f} n_gauss={n_gauss}")


if __name__ == "__main__":
    main()
