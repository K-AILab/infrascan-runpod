#!/usr/bin/env python
"""Refit a detected box's size and center to the ACTUAL spatial extent of
the Gaussians it encloses, instead of trusting the size/position that
came out of the detector's single-depth-sample 2D->3D lift.

Motivation: user-reported "box size doesn't match the real object, and
boxes are sometimes offset from the object's true center." Investigated
alongside closed_surface_flux.py's filtering pass (see
external/splat_analyzer/RUN_NOTES.md's 18th-19th rounds) and found the
same root cause underlies both the worst cases (empty/near-empty boxes)
and this milder, more common one: the detector's box size/position is a
rough guess from ONE depth sample, not a fit to the real geometry that's
already sitting right there once you know roughly where to look.

This is NOT a trick from the Gaussian-Det paper — that paper's flux
(Theorem 1) is used purely as a TRAINING loss for a learned box-regression
head; it has no direct test-time formula for fixing a given box's size.
This script is a separate, more direct geometric idea: use the box we
already trust enough to have SOME real object inside it (post
closed_surface_flux.py filtering) as a rough seed, then look at what's
really there.

Method, per box:
1. Find gaussians whose center falls inside the box (closed_surface_flux's
   enclosed_mask), in the box's own local (unrotated) frame.
2. Take a robust (not min/max) extent per local axis — the
   `--lo-pct`/`--hi-pct` percentiles — so a handful of stray outlier
   gaussians don't blow the box out.
3. Pad outward by each gaussian's own physical size (median of its two
   LARGEST scale axes among the enclosed gaussians — the two axes that
   span its surface disk, excluding the thin normal-direction axis) since
   a Gaussian's rendered surface extends beyond its stored center point.
4. Keep the box's rotation `angle` unchanged (it isn't in question here);
   recompute center/size from the padded local extent.

22nd-23rd round history (see RUN_NOTES.md for the full story): an
earlier version of this script kept only the LARGEST spatially-connected
DBSCAN cluster of enclosed points before step 2, meant to stop a box that
straddles two real separate objects from blending both into one nonsense
extent. Directly crop-verified (not assumed) on a concrete case where
this "fix" was actively WRONG: a table whose enclosed points split into
a dense 757-point cluster plus a sparse 58-point one — cropping the
source frame showed ONE continuous physical table, not two — the
cluster split was caused by uneven Gaussian reconstruction density
across a single large flat surface, not a real gap. Tested whether any
single `eps_factor` could both bridge that gap AND still split the one
case DBSCAN was originally added for (two chairs) — found NO working
middle ground: the eps needed to bridge the table's sparse region was
loose enough to merge essentially every object's points into one
cluster everywhere (i.e., equivalent to no clustering at all), and the
original "two separate chairs" case turned out to be inconclusive on
re-inspection (blurry crops, plausibly one row of attached desk-chair
units, not confirmed as two chairs the way the table was confirmed as
one). Given real, confirmed evidence on one side and weak/uncertain
evidence on the other, dropped the clustering step entirely — see
`largest_cluster()`, kept in this file but unused by default, in case a
future case needs it with better-calibrated logic (e.g. an absolute,
not density-relative, gap threshold — note that a simple absolute-
distance threshold ALSO didn't cleanly separate these two cases: the
confirmed-single table's gap (0.097) was larger than the
inconclusive-chair case's gap (0.040), the opposite of what a distance
threshold would need).

Usage:
  python refit_box_extent.py --ply ../data/shinhan_hires_30k.ply \
    --boxes ../pipeline9/out/shinhan_space_splatanalyzer_derot_v19_boxes.json \
    --out ../pipeline9/out/shinhan_space_splatanalyzer_derot_v20_boxes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closed_surface_flux import enclosed_mask, load_splat_raw  # noqa: E402


def largest_cluster(local, min_samples=5, eps_factor=3.0):
    """Indices (into `local`) of the largest spatially-connected cluster.
    eps is derived per-call from each point's own k-th-nearest-neighbor
    distance rather than a fixed absolute unit, so it adapts to however
    dense (or sparse) this particular object's gaussians happen to be."""
    if len(local) < min_samples + 1:
        return np.arange(len(local))
    nn = NearestNeighbors(n_neighbors=min_samples).fit(local)
    dists, _ = nn.kneighbors(local)
    eps = float(np.median(dists[:, -1])) * eps_factor
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(local).labels_
    if (labels < 0).all():
        return np.arange(len(local))
    vals, counts = np.unique(labels[labels >= 0], return_counts=True)
    best = vals[np.argmax(counts)]
    return np.where(labels == best)[0]


def ransac_horizontal_plane(local, thresh=0.015, n_iter=500, seed=0, min_normal_z=0.85,
                             min_inlier_frac=0.25, max_height_dev=None):
    """RANSAC-fit the dominant near-HORIZONTAL plane (normal within ~32 deg
    of local Z) among enclosed points, return inlier indices or None.

    Motivation: a table's real footprint (X/Y) can be recovered from
    percentile-trimmed extent just fine MOST of the time, but that method
    has no notion of "this is a flat surface" — it treats each axis as an
    independent 1D distribution. Directly tested (not assumed) against the
    same confirmed-single-continuous-table case used throughout this file's
    history: a horizontal-plane RANSAC fit recovers the CONFIRMED-correct
    width (0.163) exactly, same as the percentile method, but is more
    robust in principle (a real point close to the tabletop's own plane is
    an inlier regardless of local point density; a point from an unrelated
    nearby surface at a different height/orientation is rejected as an
    outlier regardless of how many other similar points surround it).
    Requiring min_normal_z rules out picking a vertical (wall-like) plane
    by accident — confirmed this happens without the constraint (an
    unconstrained RANSAC fit a vertical slice through the SAME table,
    giving a nonsense 0.039 width).

    max_height_dev (optional, None preserves original behavior): reject any
    candidate plane whose height AT THE LOCAL ORIGIN (the box's own center)
    deviates from 0 by more than this — added when searching a much wider
    window than the original detection (refit_box_extent_from_mesh.py):
    without it, a nearby larger flat surface at a DIFFERENT height (e.g.
    open floor next to a raised table) can out-vote the real tabletop
    plane simply by having more points in the wider window, since a
    same-height check was never needed when the search region was already
    tight around the object.
    """
    n_pts = len(local)
    if n_pts < 20:
        return None
    rng = np.random.default_rng(seed)
    best_inliers = None
    for _ in range(n_iter):
        sample = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = local[sample]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if abs(n[2]) < min_normal_z:
            continue
        d = -n.dot(p0)
        if max_height_dev is not None and abs(-d / n[2]) > max_height_dev:
            continue
        dist = np.abs(local @ n + d)
        inliers = np.where(dist < thresh)[0]
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
    if best_inliers is None or len(best_inliers) < max(20, n_pts * min_inlier_frac):
        return None
    return best_inliers


def refit_box(xyz, scale, center, size, angle, lo_pct=0.5, hi_pct=99.5, pad_factor=1.0,
              min_gaussians=20, label=None):
    """Returns (new_center, new_size, n_gaussians) or (None, None, n) if
    there aren't enough enclosed gaussians to refit responsibly."""
    center = np.asarray(center, dtype=np.float64)
    idx, local = enclosed_mask(xyz, center, size, angle)
    if len(idx) < min_gaussians:
        return None, None, len(idx)

    lo = np.percentile(local, lo_pct, axis=0)
    hi = np.percentile(local, hi_pct, axis=0)

    if label == "table":
        inl = ransac_horizontal_plane(local)
        if inl is not None:
            lo[:2] = local[inl, :2].min(axis=0)
            hi[:2] = local[inl, :2].max(axis=0)
            # keep the height (Z) extent from the FULL enclosed set (the plane
            # inliers are just the thin tabletop surface, not the whole object)

    surface_axes = np.sort(scale[idx], axis=1)[:, 1:]  # drop the thinnest (normal) axis
    radii = np.sqrt(surface_axes[:, 0] * surface_axes[:, 1])  # geometric mean of the two surface axes
    pad = float(np.median(radii)) * pad_factor
    lo = lo - pad
    hi = hi + pad

    new_size_local = hi - lo
    new_center_local = (hi + lo) / 2.0

    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    new_center = center + r @ new_center_local
    return new_center, new_size_local, len(idx)


MIN_FOOTPRINT = 0.05  # splat-native units (~0.25m). General safety net for
# table/chair specifically: if the enclosed points are, for whatever
# reason, spatially tiny (e.g. depth-lift landed on just a leg or a small
# fragment), refit can still produce a physically-implausible box. Found
# via one concrete case (a "table" refit to [0.025, 0.018, 0.07]) where
# BOTH horizontal dims came out under 0.03 while every other TABLE/CHAIR
# had at least one horizontal dim >= 0.055 — a clean, data-driven gap for
# THOSE labels specifically. NOT applied to "light" (or other labels):
# ceiling lights are legitimately physically small (already crop-
# validated at similar dims in an earlier round — see RUN_NOTES.md's
# 20th round), so this floor would wrongly reject correct light fits.
FOOTPRINT_CHECK_LABELS = {"table", "chair"}


def refit_is_plausible(label, new_size):
    if label not in FOOTPRINT_CHECK_LABELS:
        return True
    return not (new_size[0] < MIN_FOOTPRINT and new_size[1] < MIN_FOOTPRINT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ply", required=True)
    ap.add_argument("--boxes", required=True, help="a viewer-format boxes.json, ideally already "
                     "passed through closed_surface_flux.py's --min-gaussians filter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo-pct", type=float, default=0.5)
    ap.add_argument("--hi-pct", type=float, default=99.5)
    ap.add_argument("--pad-factor", type=float, default=1.0)
    ap.add_argument("--min-gaussians", type=int, default=20,
                     help="skip refit (keep original box) below this — too few points to trust "
                     "a percentile-based extent")
    ap.add_argument("--labels", default=None,
                     help="comma-separated labels to refit (others pass through unchanged). "
                     "Default: all. Use this to exclude labels where a single detected box "
                     "commonly spans TWO real adjacent objects (confirmed for 'table' in this "
                     "room's continuous desk rows — see RUN_NOTES.md's 21st round) — refit can "
                     "only ever recover ONE of the two objects, silently dropping the other, "
                     "which is worse than leaving the (imprecise but at least covering both) "
                     "original box alone.")
    args = ap.parse_args()
    only_labels = set(args.labels.split(",")) if args.labels else None

    xyz, scale, _normals, _area = load_splat_raw(args.ply)
    print(f"[refit] {len(xyz):,} gaussians loaded from {args.ply}")

    data = json.loads(Path(args.boxes).read_text())
    n_refit = n_skipped = 0
    skipped_boxes = []          # (box_dict) for the fallback pass below
    refit_sizes_by_label = {}   # label -> list of successfully-refit sizes
    for b in data["boxes"]:
        if only_labels is not None and b["label"] not in only_labels:
            n_skipped += 1
            continue
        angle = b.get("angle", 0.0)
        new_center, new_size, n_gauss = refit_box(
            xyz, scale, b["center"], b["size"], angle,
            lo_pct=args.lo_pct, hi_pct=args.hi_pct, pad_factor=args.pad_factor,
            min_gaussians=args.min_gaussians, label=b["label"])
        if new_center is None:
            n_skipped += 1
            skipped_boxes.append(b)
            print(f"  skip {b['label']:8s} n_gauss={n_gauss} (below --min-gaussians {args.min_gaussians})")
            continue
        if not refit_is_plausible(b["label"], new_size):
            n_skipped += 1
            skipped_boxes.append(b)
            print(f"  skip {b['label']:8s} n_gauss={n_gauss:5d} refit size {new_size} "
                  f"implausibly tiny (both horizontal dims < {MIN_FOOTPRINT}) — keeping original box")
            continue
        old_size = np.asarray(b["size"])
        print(f"  refit {b['label']:8s} n_gauss={n_gauss:5d} "
              f"size {old_size} -> {new_size}")
        b["center"] = [float(v) for v in new_center]
        b["size"] = [float(v) for v in new_size]
        n_refit += 1
        refit_sizes_by_label.setdefault(b["label"], []).append(new_size)

    # A box skipped above (too few enclosed gaussians, or an implausible
    # refit) keeps the detector's own raw size - which, per this project's
    # established finding (CURRENT_BEST_RESULT.md), is frequently a crude
    # same-as-width-and-height single-sample guess, floor-clamped to a
    # fixed 0.1-native-unit CUBE when the real estimate came out too small.
    # That's not just cosmetically wrong: a downstream CLIP relabel step
    # samples real points / frames a photo crop using this box's OWN size,
    # so a ~5x-oversized cube (0.1 native units can be ~1m real) pulls in
    # whatever's behind/around the real object - confirmed directly on
    # factory14: a real office chair with this exact degenerate size
    # produced a crop dominated by the workbench behind it, and CLIP
    # (reasonably, given what it was shown) called it "workbench" instead
    # of "chair". Detect the same round-number-cube signature here and
    # replace ONLY the size (never the center) with this run's own
    # per-label median of successfully-refit boxes - a real, data-derived
    # size for that label in THIS room, not another universal placeholder.
    DEGENERATE_CUBE_TOL = 0.003  # native units; the raw default is exactly 0.1^3
    n_fallback = 0
    for b in skipped_boxes:
        s = b["size"]
        if max(s) - min(s) > DEGENERATE_CUBE_TOL or abs(s[0] - 0.1) > DEGENERATE_CUBE_TOL:
            continue  # not the degenerate-cube pattern - a real (if unrefit) size, leave alone
        candidates = refit_sizes_by_label.get(b["label"])
        if not candidates:
            continue  # no same-label reference in this run either - nothing better to fall back to
        median_size = np.median(np.asarray(candidates), axis=0)
        print(f"  fallback {b['label']:8s} degenerate cube size {s} -> per-label median {median_size}")
        b["size"] = [float(v) for v in median_size]
        n_fallback += 1

    Path(args.out).write_text(json.dumps(data, indent=2))
    print(f"\n[refit] refit {n_refit}, skipped {n_skipped} ({n_fallback} of those given a "
          f"per-label median-size fallback) -> {args.out}")


if __name__ == "__main__":
    main()
