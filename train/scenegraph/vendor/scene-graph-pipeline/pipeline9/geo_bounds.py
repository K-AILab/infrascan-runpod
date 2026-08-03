#!/usr/bin/env python
"""Auto-derive the per-space geometric constants the topdown-segmentation
pipeline needs (floor height, room footprint bounds) directly from a splat's
own point cloud, so a new space doesn't require hand-measuring these the
way this project's first pass at shinhan_space did.

Unlike yaw/scale-to-meters (which need an external real-world reference and
genuinely can't be derived from the points alone - see run_full_pipeline.py's
docstring), floor height and room footprint ARE directly observable in any
indoor scan: the floor is the densest horizontal surface near the bottom of
the point cloud, and the footprint is simply the visible extent. Both are
exposed as CLI overrides by callers so a measured value can still replace
the auto-derived one, same convention as yaw-deg/scale-to-meters.
"""
from __future__ import annotations

import numpy as np


def auto_floor_z(xyz: np.ndarray, opacity: np.ndarray, opacity_thresh: float = 0.3,
                  search_frac: float = 0.20, z_percentile: float = 0.5) -> float:
    """Densest ~2cm-equivalent z-band within the bottom `search_frac` of the
    (percentile-trimmed, to ignore stray noise points) z-range. The floor is
    always the largest flat horizontal surface near the bottom of an indoor
    scan, so its z-histogram bin dominates within that bottom slice."""
    vis = xyz[opacity >= opacity_thresh]
    z = vis[:, 2]
    z_lo, z_hi = np.percentile(z, [z_percentile, 100 - z_percentile])
    band = z[(z >= z_lo) & (z <= z_lo + search_frac * (z_hi - z_lo))]
    bins = max(int((band.max() - band.min()) / 0.004), 5)
    counts, edges = np.histogram(band, bins=bins)
    i = int(counts.argmax())
    return float((edges[i] + edges[i + 1]) / 2)


def auto_ceiling_z(xyz: np.ndarray, opacity: np.ndarray, opacity_thresh: float = 0.3,
                    search_frac: float = 0.20, z_percentile: float = 0.5) -> float:
    """Mirror of auto_floor_z: densest z-band within the TOP `search_frac` of
    the (percentile-trimmed) z-range - the ceiling is the largest flat
    horizontal surface near the top of an indoor scan."""
    vis = xyz[opacity >= opacity_thresh]
    z = vis[:, 2]
    z_lo, z_hi = np.percentile(z, [z_percentile, 100 - z_percentile])
    band = z[(z <= z_hi) & (z >= z_hi - search_frac * (z_hi - z_lo))]
    bins = max(int((band.max() - band.min()) / 0.004), 5)
    counts, edges = np.histogram(band, bins=bins)
    i = int(counts.argmax())
    return float((edges[i] + edges[i + 1]) / 2)


def auto_room_bounds(xyz: np.ndarray, opacity: np.ndarray, opacity_thresh: float = 0.3,
                      pad_frac: float = 0.02) -> tuple[float, float, float, float]:
    """Raw min/max of visible points' x,y with a small safety pad - tested
    against shinhan_space's own hand-measured bounds and found to match
    within a few cm (raw min/max, not percentile-trimmed: trimming turned
    out to clip real wall/room extent too aggressively)."""
    vis = xyz[opacity >= opacity_thresh]
    x0, x1 = vis[:, 0].min(), vis[:, 0].max()
    y0, y1 = vis[:, 1].min(), vis[:, 1].max()
    px, py = pad_frac * (x1 - x0), pad_frac * (y1 - y0)
    return float(x0 - px), float(x1 + px), float(y0 - py), float(y1 + py)
