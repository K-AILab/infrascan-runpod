#!/usr/bin/env python
"""Refit detected table boxes' geometry (position + size) using the
reconstructed MESH surface instead of raw Gaussian centers.

Motivation (explicit user request): the existing detections came from the
Gaussian splat via splat_analyzer (OWLv2 2D detection + depth-lift + 3D
clustering - see external/splat_analyzer/CURRENT_BEST_RESULT.md) and should
stay as the base — WHICH tables exist and roughly where. But their box
GEOMETRY (size/center) was fit from sparse, noisy raw Gaussian centers
(refit_box_extent.py). The mesh (pipeline9/out/whole_room/, alpha-shape
reconstruction + Taubin smoothing - see that pipeline's own history) is a
much cleaner, CLOSED-surface representation of the same room, so refitting
table geometry against mesh vertices instead of raw Gaussians should define
each table's true footprint/extent more accurately.

Frame handling: the detected boxes are in the ORIGINAL (non-derotated)
splat frame with a per-box "angle" field that's uniformly +28.072deg (see
rotate_and_export.py - every box gets the SAME single yaw, not an
individually-fit angle). The mesh was built from the DEROTATED splat, so
it's already axis-aligned in that frame. Rather than re-rotating the whole
783k-vertex mesh, this rotates just the (few) table box centers back by
-yaw into the derotated frame, refits there with angle=0 (genuinely
axis-aligned, matching how splat_analyzer's whole pipeline was designed),
then rotates the refit result forward again by +yaw for the output format.

No Gaussian-scale-based padding (refit_box_extent.py's `pad`) is applied -
that padding exists specifically to compensate for a Gaussian's rendered
extent beyond its stored center point; a mesh vertex already lies ON the
reconstructed surface, so the enclosed extent itself IS the boundary,
modulo a small fixed pad for mesh sampling resolution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refit_box_extent import ransac_horizontal_plane, refit_is_plausible  # noqa: E402

YAW_DEG = 28.072  # established shinhan_space room yaw


def yaw_matrix(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def enclosed_mask_axis_aligned(xyz, center, size, search_expand=1.0):
    """angle=0 special case of closed_surface_flux.enclosed_mask - the
    derotated frame's boxes are genuinely axis-aligned, so skip the
    (identity) rotation. search_expand widens the box used to SELECT
    candidate points beyond the original detection's own size - if the
    original box under-covers the real table (the complaint motivating
    this whole script), searching only inside it can never recover the
    missing part no matter how good the mesh is; the final fit (percentile
    trim / RANSAC plane) still runs on whatever's found, so this only
    WIDENS what's visible to it, it doesn't force a bigger result."""
    local = xyz - center
    half = np.asarray(size, dtype=np.float64) / 2 * search_expand
    inside = np.all(np.abs(local) <= half, axis=1)
    idx = np.where(inside)[0]
    return idx, local[idx]


def largest_connected_component_at_origin(xy, cell_size, origin_radius_cells=2):
    """Rasterize (local) xy points into a grid, connected-component-label
    the occupied cells, and return a boolean mask selecting only the
    component that actually touches the ORIGINAL box's own center (the
    grid cell(s) within origin_radius_cells of local (0,0)).

    Why: a RANSAC horizontal-plane fit only checks that a point is close
    to the SAME plane by height - it says nothing about whether that point
    is part of the same CONTIGUOUS physical surface. Floor and a raised
    tabletop can sit at heights close enough for a shared plane fit to
    accept both. Confirmed directly: every one of 18 refit tables grew by
    a suspiciously uniform ~3x FOOTPRINT AREA (i.e. ~2x per linear axis,
    exactly matching the search window's own expansion factor) rather than
    varying per-table extent - a strong sign growth was hitting the search
    boundary/plane-membership alone, not each table's real edge, and the
    user directly confirmed some boxes ended up spanning empty floor.
    Requiring the kept region to be the connected component reachable from
    the detector's own original center means a genuine GAP (no mesh
    points, or points on a different local plane) stops growth there,
    while real contiguous surface (e.g. the same long bench, confirmed via
    real oblique splat renders to genuinely be one continuous physical
    platform) still grows into it, since there's no gap to stop at."""
    mins = xy.min(axis=0)
    grid_idx = np.floor((xy - mins) / cell_size).astype(np.int64)
    shape = grid_idx.max(axis=0) + 1
    occ = np.zeros(shape, dtype=bool)
    occ[grid_idx[:, 0], grid_idx[:, 1]] = True

    structure = np.ones((3, 3), dtype=bool)  # 8-connected
    labeled, _ = ndimage.label(occ, structure=structure)

    origin_cell = np.floor((np.array([0.0, 0.0]) - mins) / cell_size).astype(np.int64)
    oy, ox = origin_cell
    r = origin_radius_cells
    y0, y1 = max(0, oy - r), min(shape[0], oy + r + 1)
    x0, x1 = max(0, ox - r), min(shape[1], ox + r + 1)
    nearby_labels = labeled[y0:y1, x0:x1]
    nearby_labels = nearby_labels[nearby_labels > 0]
    if len(nearby_labels) == 0:
        return None  # no occupied cell near the original center at all
    origin_label = np.bincount(nearby_labels).argmax()

    point_labels = labeled[grid_idx[:, 0], grid_idx[:, 1]]
    return point_labels == origin_label


def principal_axis_is_x(xy, min_eigval_ratio=1.5):
    """PCA on (local) xy points; returns True if the dominant direction of
    spread aligns more with local X, False for Y, None if ambiguous (the
    two eigenvalues are too close to confidently call an orientation, or
    too few points).

    Why PCA instead of the original detection's own size (x vs y, which is
    what the first version of this script used): directly compared the two
    on all 18 tables and found 2 outright DISAGREEMENTS (tables where the
    original noisy box size said one axis was "long" but PCA on the actual
    mesh surface said the perpendicular one is) - confirmed as the exact
    cause of a reported bug where a box grew long in what a rendered view
    showed was clearly the table's SHORT side: capping was applied to
    whichever axis the (wrong) size-based guess called "short", which was
    actually the true long axis, leaving the true short axis free to grow.
    PCA is far more confident in practice: 16 of 18 tables have a principal
    axis within ~10deg of exactly 0 or 90, vs. several original size
    ratios sitting close to 1.0 (near-coin-flip)."""
    if len(xy) < 20:
        return None
    xy = xy - xy.mean(axis=0)
    cov = xy.T @ xy / len(xy)
    evals, evecs = np.linalg.eigh(cov)
    if evals[0] <= 0 or evals[1] / max(evals[0], 1e-12) < min_eigval_ratio:
        return None
    principal = evecs[:, 1]  # eigh returns ascending order
    return abs(principal[0]) >= abs(principal[1])


def refit_box_from_mesh(mesh_xyz, center_derot, size, lo_pct=0.5, hi_pct=99.5,
                        min_mesh_pts=30, fixed_pad=0.003, search_expand=1.0,
                        connectivity_cell_size=0.02, height_tolerance_factor=1.5,
                        short_axis_cap_factor=1.15, long_axis_cap_factor=1.8):
    """short_axis_cap_factor caps growth on the table's own SHORT horizontal
    axis to this factor times its original half-size, regardless of what
    the plane fit found; long_axis_cap_factor does the same (looser) for
    the long axis. Motivation (reported directly against a rendered view):
    these tables sit in parallel rows of a continuous desk bench (see this
    script's growth-plateau investigation - even the LONG axis never
    plateaus within 8x search expansion, i.e. genuinely one continuous
    surface with no internal gap in EITHER direction), and the connected-
    component gate does NOT reliably separate one row from the next
    parallel row when they're pushed flush together (no real height/
    planarity discontinuity between adjacent rows' desktops either) -
    confirmed visually: boxes were growing in the SHORT (cross-row)
    direction and overlapping the next parallel row. Since neither axis has
    a true recoverable boundary, both are hard-capped - short more tightly
    (that's the one where growth causes visible overlap), long more
    loosely (that's where real under-detection happens, the whole reason
    this script exists), rather than trusting the plane fit's raw extent
    on either.

    Orientation (axis_is_x) is determined from a TIGHT local sample
    (orientation_expand, independent of search_expand), not the wide growth
    window - confirmed directly that PCA on the wide window is unstable
    (flips depending on how much of the neighboring rows got pulled in by
    then), while a tight sample close to the original detection reflects
    just this object's own local shape, before any cross-row contamination
    is possible."""
    orientation_expand = min(search_expand, 1.3)
    o_idx, o_local = enclosed_mask_axis_aligned(mesh_xyz, center_derot, size, orientation_expand)
    axis_is_x = None
    if len(o_local) >= min_mesh_pts:
        o_inl = ransac_horizontal_plane(o_local, max_height_dev=size[2] / 2 * height_tolerance_factor)
        if o_inl is not None:
            axis_is_x = principal_axis_is_x(o_local[o_inl][:, :2])

    idx, local = enclosed_mask_axis_aligned(mesh_xyz, center_derot, size, search_expand)
    if len(idx) < min_mesh_pts:
        return None, None, len(idx)

    lo = np.percentile(local, lo_pct, axis=0)
    hi = np.percentile(local, hi_pct, axis=0)

    # constrain the plane search to near the ORIGINAL detection's own height
    # (half its own Z-size, with margin) so a nearby larger surface at a
    # different height (e.g. floor next to a raised table) can't out-vote
    # the real tabletop just by having more points in the widened window -
    # see ransac_horizontal_plane's docstring for why this was added.
    max_height_dev = size[2] / 2 * height_tolerance_factor
    inl = ransac_horizontal_plane(local, max_height_dev=max_height_dev)
    if inl is not None:
        plane_pts = local[inl]
        conn_mask = largest_connected_component_at_origin(
            plane_pts[:, :2], connectivity_cell_size)
        if conn_mask is not None and conn_mask.sum() >= 20:
            plane_pts = plane_pts[conn_mask]
        footprint_lo = plane_pts[:, :2].min(axis=0)
        footprint_hi = plane_pts[:, :2].max(axis=0)
        lo[:2] = footprint_lo
        hi[:2] = footprint_hi

        # recompute Z from points actually WITHIN the refined footprint
        # (captures the object's real height - legs+top - instead of the
        # whole wide search window's incidental height range, which grows
        # with search_expand regardless of what's really there)
        in_footprint = np.all(
            (local[:, :2] >= footprint_lo) & (local[:, :2] <= footprint_hi), axis=1)
        if in_footprint.sum() >= min_mesh_pts:
            z_vals = local[in_footprint, 2]
            lo[2] = np.percentile(z_vals, lo_pct)
            hi[2] = np.percentile(z_vals, hi_pct)

    if axis_is_x is None:
        axis_is_x = size[0] >= size[1]  # fallback: original (noisy) size guess
    long_axis = 0 if axis_is_x else 1
    short_axis = 1 - long_axis

    short_cap_half = size[short_axis] / 2 * short_axis_cap_factor
    lo[short_axis] = max(lo[short_axis], -short_cap_half)
    hi[short_axis] = min(hi[short_axis], short_cap_half)

    long_cap_half = size[long_axis] / 2 * long_axis_cap_factor
    lo[long_axis] = max(lo[long_axis], -long_cap_half)
    hi[long_axis] = min(hi[long_axis], long_cap_half)

    lo = lo - fixed_pad
    hi = hi + fixed_pad

    new_size_local = hi - lo
    new_center_local = (hi + lo) / 2.0
    new_center_derot = center_derot + new_center_local
    return new_center_derot, new_size_local, len(idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mesh", default=str(Path(__file__).resolve().parent
                     / "out" / "whole_room" / "whole_room_mesh_smoothed.ply"))
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default="table")
    ap.add_argument("--yaw-deg", type=float, default=YAW_DEG)
    ap.add_argument("--lo-pct", type=float, default=0.5)
    ap.add_argument("--hi-pct", type=float, default=99.5)
    ap.add_argument("--min-mesh-pts", type=int, default=30)
    ap.add_argument("--search-expand", type=float, default=1.0,
                     help="widen the search box beyond the original detection's own size before "
                     "fitting, so an under-sized original detection doesn't cap the refit")
    ap.add_argument("--connectivity-cell-size", type=float, default=0.02,
                     help="grid cell size (native units) for the connected-component gate that "
                     "stops growth at a genuine gap instead of the search window's own edge")
    ap.add_argument("--min-footprint-ratio", type=float, default=0.9,
                     help="reject a refit whose footprint area is smaller than this fraction of "
                     "the ORIGINAL box's - a mesh coverage gap at this location (sparse/incomplete "
                     "reconstruction) can make the fit shrink to a small fragment even though the "
                     "real object is at least as big as already-established by the original "
                     "detection; keep the original box in that case rather than trust it")
    ap.add_argument("--short-axis-cap-factor", type=float, default=1.15,
                     help="hard cap on growth for the table's own SHORT horizontal axis (cross-row "
                     "direction), regardless of what the plane/connectivity fit finds - prevents "
                     "growing into the next parallel row/desk (see refit_box_from_mesh's docstring)")
    ap.add_argument("--long-axis-cap-factor", type=float, default=1.8,
                     help="looser hard cap on growth for the table's own LONG horizontal axis - "
                     "still capped (not unlimited) because this axis was also confirmed to never "
                     "plateau within a much wider search, i.e. has no true recoverable boundary")
    args = ap.parse_args()
    only_labels = set(args.labels.split(","))

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh_xyz = np.asarray(mesh.vertices, dtype=np.float64)
    print(f"[refit-mesh] {len(mesh_xyz):,} mesh vertices from {args.mesh}")

    R_fwd = yaw_matrix(args.yaw_deg)     # derotated -> original
    R_inv = R_fwd.T                       # original -> derotated (orthonormal)

    data = json.loads(Path(args.boxes).read_text())
    n_refit = n_skipped = 0
    for b in data["boxes"]:
        if b["label"] not in only_labels:
            continue
        center_orig = np.asarray(b["center"], dtype=np.float64)
        center_derot = R_inv @ center_orig

        new_center_derot, new_size, n_pts = refit_box_from_mesh(
            mesh_xyz, center_derot, b["size"],
            lo_pct=args.lo_pct, hi_pct=args.hi_pct, min_mesh_pts=args.min_mesh_pts,
            search_expand=args.search_expand,
            connectivity_cell_size=args.connectivity_cell_size,
            short_axis_cap_factor=args.short_axis_cap_factor,
            long_axis_cap_factor=args.long_axis_cap_factor)

        if new_center_derot is None:
            n_skipped += 1
            print(f"  skip {b['label']:8s} n_mesh_pts={n_pts} (below --min-mesh-pts {args.min_mesh_pts})")
            continue
        if not refit_is_plausible(b["label"], new_size):
            n_skipped += 1
            print(f"  skip {b['label']:8s} n_mesh_pts={n_pts:5d} refit size {new_size} implausibly tiny")
            continue

        old_footprint = b["size"][0] * b["size"][2]
        new_footprint = new_size[0] * new_size[2]
        if new_footprint < args.min_footprint_ratio * old_footprint:
            n_skipped += 1
            print(f"  skip {b['label']:8s} n_mesh_pts={n_pts:5d} refit footprint {new_footprint:.4f} "
                  f"< {args.min_footprint_ratio}x original {old_footprint:.4f} - likely a mesh "
                  f"coverage gap here, not a real smaller object; keeping original box")
            continue

        new_center_orig = R_fwd @ new_center_derot
        old_size = np.asarray(b["size"])
        print(f"  refit {b['label']:8s} n_mesh_pts={n_pts:5d} size {old_size} -> {new_size}")
        b["center"] = [float(v) for v in new_center_orig]
        b["size"] = [float(v) for v in new_size]
        n_refit += 1

    Path(args.out).write_text(json.dumps(data, indent=2))
    print(f"\n[refit-mesh] refit {n_refit}, skipped {n_skipped} -> {args.out}")


if __name__ == "__main__":
    main()
