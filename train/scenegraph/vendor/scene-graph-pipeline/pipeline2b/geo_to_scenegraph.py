#!/usr/bin/env python3
"""
Pipeline 2, step 3: build the scene graph and viewer assets from
geo_label_clip.py's labeled nodes.

Reuses scene_graph.py's room/area/edge/serialise machinery unchanged (room
detection, area subdivision, building-yaw alignment, and the geometric
relation edges — standing_on/above/below/left/right/... — all only need
centroid/bbox/volume, not how the object was originally detected). CLIP
labels and the structure backstop come from geo_label_clip.py's output;
this script does not run CLIP itself.

Usage:
    conda activate infrascan
    python pipeline2/geo_cluster.py       --space factory_space_14
    python pipeline2/geo_label_clip.py    --space factory_space_14
    python pipeline2/geo_to_scenegraph.py --space factory_space_14
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))
from _paths import space, space_choices  # noqa: E402
import scene_graph as sg  # noqa: E402


TABLE_LABELS = {"table", "desk"}


def _rot_xz(xz, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.column_stack([xz[:, 0] * c - xz[:, 1] * s,
                            xz[:, 0] * s + xz[:, 1] * c])


def _tabletop_level(ys, floor_y):
    """Densest 2 cm y-histogram bin within a plausible tabletop height band
    above the floor. Using the DENSEST (not the highest) bin, restricted to
    the table-height range, avoids latching onto a shelf top or the ceiling
    that happens to fall inside a node's points."""
    lo, hi = floor_y + 0.35, floor_y + 1.45
    ys = ys[(ys > lo) & (ys < hi)]
    if len(ys) < 30:
        return None
    bins = max(int((ys.max() - ys.min()) / 0.02), 3)
    counts, edges = np.histogram(ys, bins=bins)
    i = int(counts.argmax())
    return float((edges[i] + edges[i + 1]) / 2)


def _depth_split(mask, cell, depth_max):
    """Split a footprint MASK (2D bool grid) along its shorter axis wherever
    it is deeper than a single table can be. A real table row is never more
    than ~depth_max deep, so an over-deep region is two rows back-to-back (or
    a cluttered blob): cut it at the deepest density valley on that axis, or
    the midline if there is none, until every piece is within depth_max.
    Yields sub-masks. The LONGER axis is never capped — pushed-together
    benches legitimately run several metres."""
    stack = [mask]
    out = []
    while stack:
        m = stack.pop()
        ci, cj = np.nonzero(m)
        if not len(ci):
            continue
        ext_i = (ci.max() - ci.min() + 1) * cell
        ext_j = (cj.max() - cj.min() + 1) * cell
        # shorter axis is the "depth"
        ax = 0 if ext_i <= ext_j else 1
        depth = min(ext_i, ext_j)
        if depth <= depth_max:
            out.append(m)
            continue
        prof = m.sum(axis=1 - ax).astype(float)   # occupancy along depth axis
        idx = np.nonzero(prof)[0]
        lo, hi = idx.min(), idx.max()
        interior = prof[lo + 1:hi]
        cut = None
        if len(interior):
            med = np.median(prof[prof > 0])
            k = lo + 1 + int(interior.argmin())
            if prof[k] < 0.4 * med and (k - lo) * cell > 0.3 and (hi - k) * cell > 0.3:
                cut = k
        if cut is None:
            cut = (lo + hi) // 2
        a, b = m.copy(), m.copy()
        if ax == 0:
            a[cut:, :] = False
            b[:cut, :] = False
        else:
            a[:, cut:] = False
            b[:, :cut] = False
        if a.any() and b.any():
            stack += [a, b]
        else:
            out.append(m)
    return out


def consolidate_table_footprints(nodes, node_xyz, probe, floor_y, yaw_deg=0.0):
    """One box per table via PER-SEED region growing on the tabletop plane
    (Poux et al. 2020 in spirit; hardened for multi-view blur). For each
    table/desk node: take its densest tabletop level, then grow a connected
    region over the full cloud WITHIN A WINDOW around the seed, on an
    occupancy grid where a cell counts only if it holds >= MIN_CELL points
    (density high-pass drops the sparse blur that bridged aisles). The
    window caps how far a region can reach, so a distant table can never be
    merged in; the connected component stops at any real aisle. Interior
    holes (items/occlusion) are closed and filled; the rim the high-pass
    trims is grown back so the box covers the whole top.

    Recall-first: every table node that has points yields a box (no seed is
    silently dropped), because the component is chosen by overlap with the
    node's OWN tabletop points, not a single centroid cell. Near-duplicate
    boxes from two nodes on one table are removed by centroid dedup.

    Returns (boxes, drop): boxes=[{col,n_points,mean_rgb}], col = floor ->
    tabletop+items column feeding apply_building_yaw; drop = every table
    node id (all replaced by these boxes)."""
    from scipy.ndimage import (label as ndlabel, binary_fill_holes,
                               binary_closing, binary_dilation)
    CELL, MIN_CELL, BAND, ITEM_H, GROW, DEPTH_MAX = 0.08, 3, 0.06, 0.9, 1.8, 1.4
    if probe is None or floor_y is None:
        return [], set()
    tables = [n for n in nodes
              if n.get("label") in TABLE_LABELS and not n.get("is_structure")]
    tables.sort(key=lambda n: -n.get("n_points", 0))
    drop = {n["id"] for n in tables}
    al_all = _rot_xz(probe[:, [0, 2]], -yaw_deg)
    boxes = []
    claimed = []                       # (bmin,bmax) aligned, for overlap dedup

    def emit(mask, u0, v0, level, rgb):
        ci, cj = np.nonzero(mask)
        if not len(ci):
            return
        us = u0 + (ci + 0.5) * CELL
        vs = v0 + (cj + 0.5) * CELL
        bmin = np.array([us.min(), vs.min()])
        bmax = np.array([us.max(), vs.max()])
        ext = bmax - bmin + CELL
        short, long = sorted(ext)
        if short < 0.35 or long < 0.45:
            return
        for b0, b1 in claimed:          # skip a near-duplicate of an emitted box
            ix = max(0.0, min(bmax[0], b1[0]) - max(bmin[0], b0[0]))
            iz = max(0.0, min(bmax[1], b1[1]) - max(bmin[1], b0[1]))
            inter = ix * iz
            amin = min(float(np.prod(bmax - bmin)), float(np.prod(b1 - b0)))
            if inter > 0.5 * max(amin, 1e-9):
                return
        claimed.append((bmin, bmax))
        col_top = level + 0.04
        infp = ((al_all[:, 0] >= us.min() - CELL) & (al_all[:, 0] <= us.max() + CELL)
                & (al_all[:, 1] >= vs.min() - CELL) & (al_all[:, 1] <= vs.max() + CELL))
        above = probe[infp & (probe[:, 1] > level + 0.04)
                      & (probe[:, 1] < level + ITEM_H)]
        if len(above) >= 15:
            col_top = float(np.percentile(above[:, 1], 95))
        world = _rot_xz(np.column_stack([us, vs]), yaw_deg)
        col = np.vstack([
            np.column_stack([world[:, 0], np.full(len(world), floor_y), world[:, 1]]),
            np.column_stack([world[:, 0], np.full(len(world), col_top), world[:, 1]]),
        ])
        boxes.append({"col": col, "n_points": int(mask.sum() * 8), "mean_rgb": rgb})

    for n in tables:
        own = node_xyz.get(n["id"])
        if own is None or len(own) < 40:
            continue
        level = _tabletop_level(own[:, 1], floor_y)
        if level is None:
            continue
        seed_a = _rot_xz(np.array([[n["centroid"][0], n["centroid"][2]]]), -yaw_deg)[0]
        band = (np.abs(probe[:, 1] - level) < BAND) \
            & (np.abs(al_all[:, 0] - seed_a[0]) < GROW) \
            & (np.abs(al_all[:, 1] - seed_a[1]) < GROW)
        w = al_all[band]
        if len(w) < 20:
            continue
        u0, v0 = w.min(0)
        gu = ((w[:, 0] - u0) / CELL).astype(int)
        gv = ((w[:, 1] - v0) / CELL).astype(int)
        nu, nv = int(gu.max()) + 1, int(gv.max()) + 1
        cnt = np.zeros((nu, nv), int)
        np.add.at(cnt, (gu, gv), 1)
        occ = binary_closing(cnt >= MIN_CELL, np.ones((3, 3), bool), iterations=2)
        if not occ.any():
            continue
        lab, _ = ndlabel(occ, np.ones((3, 3), int))
        # choose the component with the most of THIS node's own tabletop cells
        own_top = own[np.abs(own[:, 1] - level) < BAND]
        if len(own_top) < 10:
            continue
        oa = _rot_xz(own_top[:, [0, 2]], -yaw_deg)
        ou = np.clip(((oa[:, 0] - u0) / CELL).astype(int), 0, nu - 1)
        ov = np.clip(((oa[:, 1] - v0) / CELL).astype(int), 0, nv - 1)
        labs_hit = lab[ou, ov]
        labs_hit = labs_hit[labs_hit > 0]
        if not len(labs_hit):
            continue
        L = np.bincount(labs_hit).argmax()
        comp = binary_dilation(binary_fill_holes(lab == L),
                               np.ones((3, 3), bool), iterations=2)
        for piece in _depth_split(comp, CELL, DEPTH_MAX):
            emit(piece, u0, v0, level, n.get("mean_rgb"))
    print(f"[geo→sg] table consolidation: {len(boxes)} table boxes "
          f"(per-seed region grow, window {GROW}m, cell={CELL}m, "
          f"min {MIN_CELL} pts/cell; depth cap {DEPTH_MAX}m), "
          f"{len(drop)} original table nodes replaced")
    return boxes, drop


# Labels whose boxes commonly overshoot the real object (a loose percentile
# bbox keeps a sparse tail of leaked-in neighbour/wall points, or bridges to
# an adjacent unit). Same density-high-pass footprint idea as the tabletop
# grow, but applied to the node's OWN points and NOT grown through the full
# cloud — so it only SHRINKS to the object's dense body, never merges
# neighbours.
FOOTPRINT_TIGHTEN_LABELS = {"shelf", "storage_rack", "cabinet",
                            "bookshelf", "locker", "machine", "whiteboard"}
# Floor-standing furniture: its box must reach the floor (a storage rack
# does not float — user directive). Whiteboards are wall-mounted and can sit
# off the floor, so they are tightened but NOT floor-anchored.
FLOOR_STANDING_LABELS = {"shelf", "storage_rack", "cabinet",
                         "bookshelf", "locker", "machine"}


def tighten_footprint(pts, yaw_deg, cell=0.08, min_cell=3, min_keep=0.5):
    """Return the subset of a node's points forming its dense, connected
    body — drops the sparse peripheral cells that inflate the box beyond the
    real object. Projects to the wall-aligned XZ plane, keeps only cells
    with >= min_cell points (density high-pass), takes the connected
    component containing the centroid, fills holes and grows back one cell.
    Falls back to the original points if that would discard more than
    (1 - min_keep) of them (i.e. the high-pass fragmented a genuinely sparse
    object rather than trimming a leaked tail)."""
    from scipy.ndimage import (label as ndlabel, binary_fill_holes,
                               binary_closing, binary_dilation)
    if pts is None or len(pts) < 40:
        return pts
    al = _rot_xz(pts[:, [0, 2]], -yaw_deg)
    u0, v0 = al.min(0)
    gu = ((al[:, 0] - u0) / cell).astype(int)
    gv = ((al[:, 1] - v0) / cell).astype(int)
    nu, nv = int(gu.max()) + 1, int(gv.max()) + 1
    cnt = np.zeros((nu, nv), int)
    np.add.at(cnt, (gu, gv), 1)
    occ = cnt >= min_cell
    if not occ.any():
        return pts
    occ = binary_closing(occ, np.ones((3, 3), bool))
    lab, nlab = ndlabel(occ, np.ones((3, 3), int))
    if nlab == 0:                      # closing eroded every cell away
        return pts
    cen = al.mean(0)
    cu = int(np.clip((cen[0] - u0) / cell, 0, nu - 1))
    cv = int(np.clip((cen[1] - v0) / cell, 0, nv - 1))
    if lab[cu, cv] == 0:
        occi = np.argwhere(lab > 0)
        if not len(occi):
            return pts
        d = np.abs(occi[:, 0] - cu) + np.abs(occi[:, 1] - cv)
        cu, cv = occi[d.argmin()]
    comp = binary_dilation(binary_fill_holes(lab == lab[cu, cv]),
                           np.ones((3, 3), bool))
    keep = comp[gu, gv]
    if keep.mean() < min_keep:
        return pts
    return pts[keep]


def build_wall_blocker(wall_meta: dict | None) -> dict | None:
    """Convert geo_cluster's exported wall grid into the wall_blocker
    consumed INSIDE sg.build_edges — wall suppression happens in the
    scene-graph construction itself, not as a post-hoc edge filter. Cells
    are dilated by one so occlusion gaps in the wall's point coverage
    (doorway-sized holes in the scan, not real doorways) can't let an edge
    slip through."""
    if not wall_meta or not wall_meta.get("cells"):
        return None
    cells = set()
    for gx, gz in wall_meta["cells"]:
        for ax in (-1, 0, 1):
            for az in (-1, 0, 1):
                cells.add((gx + ax, gz + az))
    return {"cells": cells,
            "x0": float(wall_meta["x_min"]), "z0": float(wall_meta["z_min"]),
            "cell_m": float(wall_meta["cell_m"])}


# ── (pipeline2b) box rejection filters ────────────────────────────────────
FLOOR_SLAB_H_M   = 0.30   # a flat box (height < this) sitting low to the floor
FLOOR_NEAR_M     = 0.35   # (bottom within this of the floor) is floor residue
FLOOR_SLAB_WIDE_M = 1.30  # ...but only drop it if it's a LARGE flat patch
FLOOR_THIN_H_M   = 0.10   # (footprint wider than this, e.g. nodes 177/172) or
                          # a very thin sheet (< this tall). Compact low boxes
                          # (real pallets ~0.8-1.2m, h~0.15) are SPARED. Tables
                          # survive regardless: their box is ~0.8m tall.
WALL_PANEL_H_M   = 1.30   # a thin, tall, WIDE box is a wall/partition fragment
WALL_PANEL_THIN_M = 0.20  # (e.g. a wall patch mislabeled "shelf"). Whiteboards
WALL_PANEL_WIDE_M = 1.80  # are thin+tall but narrower than this, so survive.
FILL_MIN         = 0.03   # min fraction of the box's 0.12m voxel grid that
FILL_CELL_M      = 0.12   # holds points; below this the box encloses empty
                          # space (nothing actually there) and is dropped.


def _reject_floor_wall_empty(objects: dict, floor_y):
    """Drop boxes that (a) lie flat on the floor, (b) are thin tall wall
    panels, or (c) enclose mostly empty space (low point fill ratio). Uses
    each object's own world points (bbox_pts) and wall-aligned bbox_size."""
    drop = []
    for oid, o in objects.items():
        if o.get("_ann"):        # user-drawn box is authoritative — never reject
            continue
        sz = o.get("bbox_size")
        pts = o.get("bbox_pts")
        if sz is None:
            continue
        h = float(sz[1])
        horiz = sorted((float(sz[0]), float(sz[2])))
        bottom = float(np.asarray(pts)[:, 1].min()) if pts is not None and len(pts) else None
        if (floor_y is not None and bottom is not None
                and bottom < floor_y + FLOOR_NEAR_M and h < FLOOR_SLAB_H_M
                and (horiz[1] > FLOOR_SLAB_WIDE_M or h < FLOOR_THIN_H_M)):
            drop.append((oid, "floor_slab")); continue
        if (h > WALL_PANEL_H_M and horiz[0] < WALL_PANEL_THIN_M
                and horiz[1] > WALL_PANEL_WIDE_M):
            drop.append((oid, "wall_panel")); continue
        if pts is not None and len(pts) >= 8:
            p = np.asarray(pts)
            k = np.floor((p - p.min(0)) / FILL_CELL_M).astype(np.int64)
            occ = len(np.unique(k, axis=0))
            tot = int(np.prod(np.maximum(np.ceil(np.asarray(sz) / FILL_CELL_M), 1)))
            if tot > 0 and occ / tot < FILL_MIN:
                drop.append((oid, "empty")); continue
    for oid, _ in drop:
        del objects[oid]
    return drop


# ── (pipeline2b) wall / partition slab rejection ──────────────────────────
# The WALL_PANEL test in _reject_floor_wall_empty only catches a very specific
# thin+tall+wide box. In practice many wall segments get boxed as "shelf"
# without hitting all three thresholds (e.g. sp13 #38/#64/#65/#129), and
# VerModule under-detects the walls themselves, so we cannot rely on a wall
# mask. Instead we judge each object by its own WALL-ALIGNED, trimmed box
# (bbox_size — what the viewer draws): a wall/partition fragment is a TALL,
# thin, ELONGATED panel that reaches high up the wall. Crucially the panel
# must be elongated (long horizontal >> short horizontal); this keeps blocky /
# near-square objects with a thin side — office chairs, ladders, cabinets,
# columns/posts — which is what separates a wall from real furniture. Uses no
# hardcoded wall position, so it generalises across spaces.
WALL_THIN_M       = 0.35   # a horizontal side this thin alone marks a panel
WALL_PANEL_THIN_M = 0.65   # ...or this thin, if the panel is clearly elongated
WALL_PANEL_ASPECT = 1.6    #    (long horizontal / short horizontal > this)
WALL_MIN_H_M      = 1.10   # tall enough to be a wall, not a chair / low box
WALL_MIN_TOP_M    = 1.80   # its top reaches this far above the floor (up the wall)


def _reject_wall_slabs(objects: dict, floor_y):
    """Drop boxes that are tall, thin, elongated vertical panels (wall /
    partition fragments), judged from the wall-aligned trimmed box the viewer
    draws. Never touches a user-drawn (_ann) box."""
    if floor_y is None:
        return []
    drop = []
    for oid, o in objects.items():
        if o.get("_ann"):
            continue
        sz = o.get("bbox_size")
        if sz is None:
            continue
        sy = float(sz[1])
        short_h, long_h = sorted((float(sz[0]), float(sz[2])))
        centre = o.get("box_center") or o.get("centroid")
        if centre is None:
            continue
        top = float(centre[1]) + sy / 2.0 - floor_y
        is_panel = (short_h < WALL_THIN_M
                    or (short_h < WALL_PANEL_THIN_M
                        and long_h / max(short_h, 1e-6) > WALL_PANEL_ASPECT))
        if is_panel and sy > WALL_MIN_H_M and top > WALL_MIN_TOP_M:
            drop.append((oid, o.get("label")))
    for oid, _ in drop:
        del objects[oid]
    return drop


CEIL_CONFINE_M = 1.0   # a box whose BOTTOM is within this of the ceiling lives
                       # entirely in the top ~1 m of the room = a duct / beam /
                       # ceiling panel / hanging fixture. Measured from the
                       # CEILING (not the floor): even a high wall shelf's bottom
                       # sits far below this line, so real objects are spared;
                       # only ceiling-confined clutter qualifies.


WALL_PLANE_THIN   = 0.030   # PCA λ_min/λ_mid below this = a flat planar slab
WALL_PLANE_NY     = 0.35    # …whose plane normal is horizontal = vertical wall
WALL_PLANE_MIN_H  = 1.0     # …and it is tall. A rack/shelf has depth or
                            # horizontal shelf surfaces, so its points are NOT
                            # planar (λ_min not tiny) and it is spared.


def _reject_wall_planes(objects: dict, floor_y):
    """Drop residual wall / partition / window / panel boxes: tall boxes whose
    own points form a THIN VERTICAL PLANE (PCA smallest axis ≪ middle axis, and
    that axis — the surface normal — is horizontal). This is the user's normals
    cue applied at the box level: a wall face is a vertical plane; a shelf/rack
    standing against it has front-to-back depth and horizontal shelf surfaces,
    so its points are not planar and it survives. Geometric, not semantic."""
    drop = []
    for oid, o in objects.items():
        if o.get("_ann"):
            continue
        pts = o.get("bbox_pts")
        sz = o.get("bbox_size")
        if pts is None or sz is None or len(pts) < 50:
            continue
        if float(sz[1]) < WALL_PLANE_MIN_H:
            continue
        p = np.asarray(pts, float)
        try:
            w, V = np.linalg.eigh(np.cov((p - p.mean(0)).T))
        except Exception:
            continue
        w = np.maximum(w, 1e-12)              # ascending eigenvalues
        planar = (w[0] / w[1]) < WALL_PLANE_THIN
        horiz_normal = abs(float(V[1, 0])) < WALL_PLANE_NY   # n_y of λ_min axis
        if planar and horiz_normal:
            drop.append((oid, o.get("label")))
    for oid, _ in drop:
        del objects[oid]
    return drop


def _reject_structure_2nd_pass(objects, floor_y, ceil_y, wall_grid):
    """Second geometric pass (after CLIP): drop boxes that are STRUCTURE-SHAPED
    regardless of the CLIP label they were given — the user's observation that
    walls/floor get boxed as 'wall'/'person'/etc. Three geometric tests, none
    semantic:
      A) a thin vertical panel whose footprint sits ON a detected wall segment
         (wall_grid) — a residual wall/partition strip. A free-standing shelf
         is not on a wall line; a shelf against a wall has real depth (not thin).
      B) a flat slab at floor level — floor residue.
      C) a paper-thin vertical plane anywhere — a real object has some depth, so
         a plane this thin (e.g. a 'person' or 'shelf' 8 cm thick and 1.5 m tall)
         is a wall/panel, not an object.
    """
    wcells = set()
    cm = xm = zm = None
    if wall_grid:
        cm = wall_grid.get("cell_m"); xm = wall_grid.get("x_min")
        zm = wall_grid.get("z_min")
        wcells = set((int(a), int(b)) for a, b in wall_grid.get("cells", []))

    def wall_frac(o):
        if not wcells:
            return 0.0
        c = o.get("box_center") or o.get("centroid"); s = o.get("bbox_size")
        if not c or not s:
            return 0.0
        gx0 = int((c[0] - s[0] / 2 - xm) / cm); gx1 = int((c[0] + s[0] / 2 - xm) / cm)
        gz0 = int((c[2] - s[2] / 2 - zm) / cm); gz1 = int((c[2] + s[2] / 2 - zm) / cm)
        tot = hit = 0
        for a in range(gx0, gx1 + 1):
            for b in range(gz0, gz1 + 1):
                tot += 1; hit += ((a, b) in wcells)
        return hit / max(tot, 1)

    if floor_y is None:
        return []
    # LEVELS learned from the operator's wall/floor relabels (round 15):
    #   floor = a flat slab low to the ground (their floor boxes: height ≤0.15,
    #           bottom 0.13–0.42 above floor)
    #   wall  = elevated structure reaching high (their wall boxes: bottom ≥0.63
    #           above floor AND top ≥1.9 — NOT floor-standing). A real object
    #           rests on the floor (bottom ≈ 0), so it is spared.
    FLOOR_H_MAX   = 0.18   # box height at/under this + low = floor slab. Their
                           # floor boxes span 0.08–0.16 (small-room floor is a
                           # touch thicker/blurrier), so 0.18 catches every room's
                           # floor; floors sit at the same level in all rooms.
    FLOOR_BOT_MAX = 0.45   # bottom above floor at/under this
    WALL_BOT_MIN  = 0.60   # bottom this far above floor = FLOATS (not floor-
                           # standing; a real rack/shelf sits at bottom≈0)
    WALL_TOP_MIN  = 1.90   # …and top reaches this high = wall level
    WALL_SHORT_MAX = 0.55  # …and it is not deep (their wall boxes ≤0.54 short;
                           # a deep free-standing rack exceeds this and is kept)
    STRUCTURE_LBL = {"wall", "partition_panel", "window", "door", "curtain",
                     "air_duct", "ceiling", "ceiling_light", "pillar", "floor"}

    drop = []
    for oid, o in objects.items():
        if o.get("_ann"):
            continue
        sz = o.get("bbox_size"); c = o.get("box_center") or o.get("centroid")
        if sz is None or c is None:
            continue
        lbl = o.get("label", "")
        short = min(float(sz[0]), float(sz[2])); sy = float(sz[1])
        long_ = max(float(sz[0]), float(sz[2]))
        bottom = float(c[1]) - sy / 2.0 - floor_y
        top = float(c[1]) + sy / 2.0 - floor_y
        # keep real ceiling lights (labelled + high in the hanging-light band)
        if lbl == "ceiling_light" and top > 2.8:
            continue
        wf = wall_frac(o)
        def _rec(reason):
            return {"oid": oid, "label": lbl, "reason": reason,
                    "box_center": list(c), "bbox_size": list(sz),
                    "pts": o.get("bbox_pts")}
        # (F) FLOOR: a flat slab. A pallet, a low workbench, and a tabletop are
        #     ALSO flat and low, and the old "flat + near-ground → floor" rule
        #     ate them (it dropped ~10 pallets, low tables, chairs as floor). A
        #     GENUINE floor residue patch is distinguishable: it is thin AND
        #     hugs the ground AND spans real area. A pallet has thickness
        #     (sy>0.12); a low table/bench is lifted clearly off the floor
        #     (bottom≥0.20); a small flat item lacks the area — all kept.
        area = long_ * short
        if sy <= 0.12 and bottom < 0.20 and area > 0.8:
            drop.append(_rec("floor-level")); continue
        # (W) WALL / wall-mounted structure: FLOATS (does NOT stand on the
        #     floor), reaches high, and is not a deep 3-D object. A floor-
        #     standing rack (bottom≈0) or a deep unit (short>0.55) is spared.
        if bottom > WALL_BOT_MIN and top > WALL_TOP_MIN and short < WALL_SHORT_MAX:
            drop.append(_rec("wall-level")); continue
        # (S) CLIP-labelled structure whose geometry is thin/planar/wall-aligned.
        struct_geom = (short < 0.15 or (sy < 0.15 and long_ > 0.5)
                       or (wf > 0.5 and short < 0.25))
        if lbl in STRUCTURE_LBL and struct_geom:
            drop.append(_rec("struct")); continue
        # (A) thin tall panel sitting on a detected wall line.
        if short < 0.16 and sy > 0.8 and wf > 0.4:
            drop.append(_rec("wall-aligned")); continue
    for r in drop:
        del objects[r["oid"]]
    return drop


def _reject_faulty(objects, min_pts=140):
    """Route physically-implausible boxes to the RED review layer (they are
    returned like the structure drops, so they render red + hidden and stay
    relabel-able — NOT hard-deleted). Two geometric tests, no semantics beyond
    a size sanity-check keyed on the class the detector itself assigned:

      • noise fragment — too few points AND tiny volume. The rendered nodes
        split cleanly (p25≈190 pts vs p50≈1200): the sub-min_pts tail is
        broken-off shards and open-vocab false hits on clutter.
      • class-impossible size — e.g. a 'person' 30 cm tall / 0.02 m^3. YOLO-
        World fires 'person' on random clutter (the user flagged this); a box
        far too small for its own class is that false positive, not the object.
    """
    MIN_H  = {"person": 0.9}     # a standing person is ≥~0.9 m tall
    MIN_VOL = {"person": 0.12}
    drop = []
    for oid, o in list(objects.items()):
        if o.get("_ann"):
            continue
        s = o.get("bbox_size"); npt = int(o.get("n_world_pts", 10 ** 9))
        if not s:
            continue
        lbl = o.get("label", ""); vol = float(s[0]) * float(s[1]) * float(s[2])
        reason = None
        if npt < min_pts and vol < 0.05:
            reason = "noise-fragment"
        elif lbl in MIN_H and (float(s[1]) < MIN_H[lbl] or vol < MIN_VOL[lbl]):
            reason = f"impossible-{lbl}"
        if reason:
            c = o.get("box_center") or o.get("centroid")
            drop.append({"oid": oid, "label": lbl, "reason": reason,
                         "box_center": list(c), "bbox_size": list(s),
                         "pts": o.get("bbox_pts")})
    for r in drop:
        del objects[r["oid"]]
    return drop


# Detector classes that are structural rather than objects (ScanNet's own
# vocabulary — see pipeline4/detr3d/detector.py:SCANNET_CLASSES). When a
# learned detector's OWN raw class (det_class/det_prob — only present on
# detector-sourced nodes, e.g. pipeline4) is one of these with reasonable
# confidence, but CLIP relabeled the crop as an ordinary object, that
# disagreement is a signal the box MIGHT be wall/window/door structure CLIP
# misread from its 2D crop. Caught a real false positive while validating
# this: a genuine 1.87x1.74x0.33m wall-mounted shelf (the exact object
# geo_label_clip.py's OWN "wall-level" heuristic was previously found to
# wrongly kill, see the CLIP-labeling section of pipeline4/README.md) got
# raw-classified "window" at 0.42 confidence too — a flat, thin,
# wall-adjacent rectangle reads as "window" to 3DETR regardless of whether
# it's a real shelf or an actual window/door opening, so det_class alone
# isn't enough corroboration.
#
# Reuses the SAME depth test geo_label_clip.py's structure-veto already
# relies on for exactly this ambiguity (WALL_MIN_DEPTH_M there): a real
# wall/window/door is a thin sheet, a shelf/cabinet/rack standing against
# a wall has actual depth. Verified this cleanly separates the confirmed
# cases: the false-positive shelf has depth 0.33-0.41m (>= the threshold,
# now correctly spared); the two genuinely-bad "cabinet" boxes have depth
# 0.25-0.28m (< the threshold, still correctly caught).
#
# Depth alone still misses one real case: a genuine floor-to-ceiling glass
# partition/window can have real, substantial depth in its crop (checked
# directly — several confirmed-bad "window"-mismatched boxes in a
# different space had 0.4-0.9m depth, comfortably past the threshold
# above) while still obviously being structural, because it spans nearly
# the room's ENTIRE height — no real piece of furniture does that. So a
# SECOND, independent path: flag regardless of depth if the box's height
# is an implausibly large fraction of the room's own floor-to-ceiling
# span (a room parameter, not a fixed absolute height, so it generalizes
# across rooms of very different scale). Checked directly: the confirmed
# real false-positive shelf is 41% of its room's height, comfortably under
# the threshold below; the confirmed-bad full-height windows are 76-95%.
_STRUCTURAL_DET_CLASSES = frozenset({"wall", "floor", "ceiling", "door", "window"})
_STRUCT_MISMATCH_MIN_DEPTH_M = 0.30
_STRUCT_MISMATCH_MAX_HEIGHT_FRAC = 0.65


def _reject_structural_mismatch(objects: dict, floor_y=None, ceil_y=None,
                                min_det_prob: float = 0.15,
                                min_depth_m: float = _STRUCT_MISMATCH_MIN_DEPTH_M,
                                max_height_frac: float = _STRUCT_MISMATCH_MAX_HEIGHT_FRAC) -> list:
    room_h = (ceil_y - floor_y) if (floor_y is not None and ceil_y is not None) else None
    drop = []
    for oid, o in list(objects.items()):
        dc = o.get("det_class")
        if not (dc in _STRUCTURAL_DET_CLASSES
                and o.get("det_prob", 0) >= min_det_prob
                and o.get("label") not in _STRUCTURAL_DET_CLASSES):
            continue
        sz = o.get("bbox_size")
        if not sz:
            continue
        is_thin = min(sz[0], sz[2]) < min_depth_m
        spans_room = room_h and (sz[1] / room_h) > max_height_frac
        if not (is_thin or spans_room):
            continue   # real depth AND not full-height — spare it
        c = o.get("box_center") or o.get("centroid")
        drop.append({"oid": oid, "label": o.get("label"),
                     "reason": f"raw-det-{dc}",
                     "box_center": list(c), "bbox_size": list(sz),
                     "pts": o.get("bbox_pts")})
    for r in drop:
        del objects[r["oid"]]
    return drop


# CLIP itself (geo_label_clip.py's classify_node) already has a wall/
# partition corroboration veto: when its own top-ranked guess for a crop is
# "wall"/"partition panel", it only accepts that if the box's geometry
# actually agrees (thin enough — WALL_MIN_DEPTH_M) — otherwise it walks
# DOWN to the next-ranked guess instead, so the object underneath isn't
# erased by an uncorroborated structure label. That's the right design,
# but its DEPTH-based corroboration specifically turned out to have the
# same reliability problem chased through this whole module: verified
# directly on a real case — CLIP's top pick was "wall" at 0.71-0.79 fused
# (a decisive margin over the 0.12-0.13 runner-up that became the final
# label), overridden purely because the box's OWN measured depth (itself
# sensitive to the same trim-aggressiveness issue documented on
# apply_building_yaw) read as "thick enough" not to be a wall.
#
# Deliberately NOT floor/ceiling/ceiling_light here, even though CLIP's
# classify_node vetoes those too — verified those corroborations are
# NOT the same unreliable mechanism: they're elevation/height-based
# (bbox_min vs floor_y, bbox_size[1] vs a fixed minimum), not the
# trim-sensitive depth measurement. Tried including them first and it
# was a real regression, caught directly: legitimate table/cardboard_box/
# trash_bin/pallet objects with real height (0.22-0.4m) sitting a few cm
# above floor_y were correctly identified as NOT floor by that elevation
# check, then wrongly reverted to "clip-top1-floor" removal by this check
# second-guessing a veto that was actually working correctly. Only the
# wall/partition_panel depth-veto is corroborated as unreliable — this
# check is scoped to exactly that.
_CLIP_STRUCTURAL_LABELS = frozenset({"wall", "partition_panel"})


def _reject_clip_structural_override(objects: dict, min_fused: float = 0.4) -> list:
    drop = []
    for oid, o in list(objects.items()):
        topk = o.get("clip_topk")
        if not topk:
            continue
        top = topk[0]
        top_lbl = str(top.get("label", "")).replace(" ", "_")
        if (top_lbl in _CLIP_STRUCTURAL_LABELS
                and top.get("fused", 0) >= min_fused
                and o.get("label") not in _CLIP_STRUCTURAL_LABELS):
            c = o.get("box_center") or o.get("centroid")
            drop.append({"oid": oid, "label": o.get("label"),
                         "reason": f"clip-top1-{top_lbl}",
                         "box_center": list(c), "bbox_size": list(o["bbox_size"]),
                         "pts": o.get("bbox_pts")})
    for r in drop:
        del objects[r["oid"]]
    return drop


def _reject_hollow_or_skewed(objects: dict, inner_frac_max: float = 0.05,
                             y_offset_min: float = 0.35, inner_cube: float = 0.4,
                             min_pts: int = 20) -> list:
    """Drop objects whose own points don't actually fill their own detected
    box — a box can have plenty of points and still not correspond to one
    real, coherent object:

      * near-zero mass anywhere close to the box's own center (points only
        in a thin outer shell) — the box spans mostly empty space, picking
        up edge/background points from whatever real structure it happens
        to border, not a solid object filling the middle;
      * the point mass is heavily skewed to one vertical extreme of the box
        (mean point height far from the box's own vertical center) — this
        happens when a detector's box height is set by a sparse handful of
        stray/background points reaching far above (or below) a dense,
        much shorter real cluster, inflating the reported box height
        well past the real object's actual extent.

    Requires BOTH conditions together: verified directly that either alone
    is far too broad (a hollow-center-only rule flagged ~24% of all nodes
    in a real scene, since open-frame furniture — chairs, racks — routinely
    has low center density too; verified against a random sample). This is
    a best-effort statistical filter on genuinely noisy detector output,
    not a precise classifier — it will still miss some bad boxes and is
    intentionally conservative to avoid dropping real ones. Non-destructive
    like the other precision passes: routed to the red review layer, never
    silently deleted for good."""
    drop = []
    for oid, o in list(objects.items()):
        pts = o.get("bbox_pts")
        if pts is None or len(pts) < min_pts:
            continue
        bmin = np.asarray(o["bbox_min"]); bmax = np.asarray(o["bbox_max"])
        ctr = (bmin + bmax) / 2
        half = np.maximum((bmax - bmin) / 2, 1e-6)
        rel = (pts - ctr) / half
        cheb = np.max(np.abs(rel), axis=1)
        inner_frac = float((cheb < inner_cube).mean())
        y_offset = float(abs(pts[:, 1].mean() - ctr[1]) / half[1])
        if inner_frac < inner_frac_max and y_offset > y_offset_min:
            drop.append({"oid": oid, "label": o.get("label"),
                         "reason": "hollow-or-skewed",
                         "box_center": list(o.get("box_center") or o.get("centroid")),
                         "bbox_size": list(o["bbox_size"]), "pts": pts})
    for r in drop:
        del objects[r["oid"]]
    return drop


# Labels whose OWN shape prior (geo_label_clip.py SHAPE_PRIORS) has no upper
# height bound — genuinely tall industrial equipment is expected/normal for
# these, so a room-height cap would wrongly punish a real tall rack.
_NO_HEIGHT_CAP_LABELS = frozenset({"storage_rack", "machine", "ladder", "pillar",
                                   "wall", "door", "curtain", "air_duct",
                                   "partition_panel"})


def _reject_implausible_height_fraction(objects: dict, floor_y, ceil_y,
                                        max_frac: float = 0.78) -> list:
    """Drop a box spanning an implausible fraction of the room's own
    floor-to-ceiling height, regardless of label or detector metadata —
    catches a real case neither structural check above fires on: a box
    labeled shelf/cabinet by CLIP with reasonable confidence, raw detector
    class NOT structural, CLIP's own top-1 NOT structural either, yet
    spanning 84-90% of the room's total height with points densely and
    fairly evenly distributed throughout (checked the histogram directly —
    not a sparse noise tail, which rules out the trim/skew explanation
    behind the other checks in this module; this genuinely looks like the
    detector's query swept up a large vertical slice of real structure,
    not one coherent piece of furniture). No real shelf/cabinet/table
    reasonably spans that much of a room. Excludes labels whose own shape
    prior has no upper height bound (_NO_HEIGHT_CAP_LABELS) since a
    genuinely tall industrial rack is expected there, not suspicious."""
    if floor_y is None or ceil_y is None:
        return []
    room_h = ceil_y - floor_y
    if room_h <= 0:
        return []
    drop = []
    for oid, o in list(objects.items()):
        if o.get("label") in _NO_HEIGHT_CAP_LABELS:
            continue
        sz = o.get("bbox_size")
        if not sz or sz[1] / room_h <= max_frac:
            continue
        c = o.get("box_center") or o.get("centroid")
        drop.append({"oid": oid, "label": o.get("label"),
                     "reason": "spans-room-height",
                     "box_center": list(c), "bbox_size": list(sz),
                     "pts": o.get("bbox_pts")})
    for r in drop:
        del objects[r["oid"]]
    return drop


def _reject_floor_hugging(objects: dict, floor_y, ceil_y=None,
                          y_up_sign: float = 1.0,
                          max_height_m: float = 0.32,
                          max_gap_m: float = 0.30, max_det_prob: float = 0.40) -> list:
    """Drop detector-sourced boxes that are almost certainly floor-level
    scan artifacts (texture/residue, not a real free-standing object):
    sitting essentially AT floor level (room parameter: floor_y) with an
    implausibly small height AND low detector confidence. Requires ALL
    THREE together, since any one alone is too common among real objects
    (a genuinely short real object, or a confident-but-oddly-shaped
    detection, shouldn't be dropped on its own). No-op for objects without
    a det_prob field — this only ever applies to detector-sourced
    pipelines. Non-destructive, same as the other precision passes here.

    Base thresholds (0.18/0.22/0.22) left as originally set — calibrated
    on a different scan, still applied to every detector-sourced object
    regardless of its raw class. A *wider* second band
    (`max_gap_m`/`max_height_m`/`max_det_prob`) additionally applies, but
    ONLY to `det_class in {"chair", "table"}`: real, user-reported
    floor-level boxes in shinhan_space (labeled pallet/table/
    cardboard_box) were rendered as camera crops and turned out to be
    genuine floor carpet texture and a scan operator's own head — but at
    gap/height/confidence values (up to 0.26m/0.30m/0.35) the base
    thresholds were too tight to catch. All 11 verified-junk boxes there
    share raw `det_class` "chair" or "table" at a height nowhere near a
    real chair/table's actual size — chairs and tables simply don't come
    that short, so a "chair"/"table" guess this thin is far more likely
    scan noise than a genuinely short piece of furniture.

    Deliberately NOT widened for every det_class: checked directly against
    factory_space_14/15's own near-floor short objects at these same wider
    values and found real false positives — two genuine objects (a
    wrapped stack of crates, a stack of cardboard boxes) that would have
    been wrongly caught, both with raw `det_class="garbagebin"`. Bins
    genuinely span a huge real height range (a squat 0.2m bin is normal),
    so "garbagebin" at low confidence isn't the same implausibility
    signal "chair"/"table" is. Restricting the wider band to
    chair/table avoids both false positives while still catching them at
    the base thresholds if they separately qualify. Non-destructive
    either way — anything caught stays recoverable via Show Removed.

    y_up_sign: same correction as geo_label_clip.shape_prior() — for a
    y_invert space (raw Y increases downward), the physical bottom is
    bbox_max (not bbox_min) and the true floor reference is ceil_y (not
    floor_y). Requires ceil_y in that case; no-op if it's missing."""
    if floor_y is None:
        return []
    if y_up_sign < 0 and ceil_y is None:
        return []
    BASE_HEIGHT_M, BASE_GAP_M, BASE_DET_PROB = 0.22, 0.18, 0.22
    WIDE_DET_CLASSES = frozenset({"chair", "table"})
    drop = []
    for oid, o in list(objects.items()):
        if "det_prob" not in o or o["det_prob"] is None:
            continue
        bmin = o.get("bbox_min"); bmax = o.get("bbox_max"); size = o.get("bbox_size")
        if not bmin or not size:
            continue
        if o.get("det_class") in WIDE_DET_CLASSES:
            gap_lim, h_lim, prob_lim = max_gap_m, max_height_m, max_det_prob
        else:
            gap_lim, h_lim, prob_lim = BASE_GAP_M, BASE_HEIGHT_M, BASE_DET_PROB
        if y_up_sign > 0:
            phys_bot_y, true_floor_ref = bmin[1], floor_y
        else:
            phys_bot_y, true_floor_ref = bmax[1], ceil_y
        elev_bot = (phys_bot_y - true_floor_ref) * y_up_sign
        if (elev_bot <= gap_lim and size[1] <= h_lim
                and o["det_prob"] <= prob_lim):
            c = o.get("box_center") or o.get("centroid")
            drop.append({"oid": oid, "label": o.get("label"),
                         "reason": "floor-hugging-low-conf",
                         "box_center": list(c), "bbox_size": list(size),
                         "pts": o.get("bbox_pts")})
    for r in drop:
        del objects[r["oid"]]
    return drop


def _reject_floating_ceiling(objects: dict, floor_y, ceil_y):
    """Drop boxes confined to the top ~1 m below the ceiling — residual ceiling
    clutter (ducts/beams/fixtures/panels) that survived bg_removal's
    point-level stripping. Object-SAFE: every floor-standing object (incl. a
    high wall shelf) has its box bottom well below ceil-CEIL_CONFINE_M, so this
    can only remove things hanging up at the ceiling. Geometric background
    removal, NOT semantic CLIP deletion."""
    if ceil_y is None:
        return []
    drop = []
    for oid, o in objects.items():
        if o.get("_ann"):
            continue
        sz = o.get("bbox_size")
        centre = o.get("box_center") or o.get("centroid")
        if sz is None or centre is None:
            continue
        # keep real ceiling lights (labelled) hanging in the light band
        if o.get("label") == "ceiling_light" and \
                (float(centre[1]) + float(sz[1]) / 2.0 - (floor_y or 0)) > 2.8:
            continue
        bottom = float(centre[1]) - float(sz[1]) / 2.0
        if bottom > ceil_y - CEIL_CONFINE_M:
            drop.append({"oid": oid, "label": o.get("label"), "reason": "ceiling",
                         "box_center": list(centre), "bbox_size": list(sz),
                         "pts": o.get("bbox_pts")})
    for r in drop:
        del objects[r["oid"]]
    return drop


def _load_cloud_rgb(space_name, sp_paths):
    """Full cloud xyz+rgb for the color-guided table pass (the sidecar npz
    probe carries xyz only)."""
    from plyfile import PlyData
    p = REPO / "ui" / "_spaces" / space_name / "Data_" / "downsampled_web.ply"
    if not p.exists():
        p = Path(sp_paths["pointcloud"])
    ply = PlyData.read(str(p))["vertex"]
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float64)
    rgb = np.stack([ply["red"], ply["green"], ply["blue"]], 1).astype(np.float64)
    return xyz, rgb


TABLE_UNDER_RATIO = 0.85   # reject a candidate whose column BELOW the top is
                           # this full relative to the top — a table is empty
                           # under its top (just legs); a rack/cabinet is full,
                           # so this geometric test rejects same-coloured racks
                           # WITHOUT relying on the colour to tell them apart.
                           # Kept loose (0.85) so tables with some clutter
                           # underneath still pass — only clearly full-below
                           # racks/cabinets are cut.
TABLE_MIN_SHORT_M = 0.40   # a real tabletop is at least this deep and this big
TABLE_MIN_AREA_M2 = 0.45   # in footprint — drops tiny teal patches/offcuts.


def _valley_split_long(uv, cell=0.10, min_seg=0.6, split_above=2.0):
    """Split a cluster's aligned points along its LONGER axis at genuine
    density valleys only (not uniform tiling — the user rejected that). A
    continuous surface with no valley stays one piece; a row of tables with
    gaps between them splits at the gaps. Returns a list of boolean masks."""
    ext = uv.ptp(0)
    ax = int(np.argmax(ext))
    if ext[ax] < split_above:
        return [np.ones(len(uv), bool)]
    a = uv[:, ax]
    lo, hi = a.min(), a.max()
    nb = max(int((hi - lo) / cell), 4)
    hist, edges = np.histogram(a, bins=nb)
    med = np.median(hist[hist > 0]) if (hist > 0).any() else 0
    cuts = []
    for i in range(1, len(hist) - 1):
        if (hist[i] < 0.30 * med and hist[i] <= hist[i - 1] and hist[i] <= hist[i + 1]):
            pos = float((edges[i] + edges[i + 1]) / 2)
            if pos - lo > min_seg and hi - pos > min_seg and (not cuts or pos - cuts[-1] > min_seg):
                cuts.append(pos)
    if not cuts:
        return [np.ones(len(uv), bool)]
    bounds = [lo - 1e-6] + cuts + [hi + 1e-6]
    return [(a >= bounds[k]) & (a < bounds[k + 1]) for k in range(len(bounds) - 1)]


# Only the tall, clearly-not-a-table classes exclude a teal candidate. "shelf"
# is deliberately excluded from this set: shelves are large and sit next to
# tables, so using them would reject real tables.
RACK_LABELS = {"storage_rack", "cabinet"}


def _aligned_rack_rects(geo_nodes, node_xyz, yaw_deg):
    """Wall-aligned footprint rects of the already-detected rack/shelf/cabinet
    nodes — used to reject teal tabletop candidates that are actually a rack's
    (same-coloured) shelf, WITHOUT an empty-below heuristic that also killed
    real tables with storage under them."""
    rects = []
    for n in geo_nodes:
        if n.get("label") not in RACK_LABELS or n.get("is_structure"):
            continue
        if n.get("sam_lifted"):     # (geo3) SAM-lifted racks are lower-confidence
            continue                # and must not suppress real table detections
        p = node_xyz.get(n["id"])
        if p is None or len(p) < 10:
            continue
        uv = _rot_xz(np.asarray(p)[:, [0, 2]], -yaw_deg)
        rects.append((uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()))
    return rects


def _rack_overlap_frac(rect, rack_rects):
    u0, v0, u1, v1 = rect
    area = max((u1 - u0) * (v1 - v0), 1e-6)
    inter = 0.0
    for r0, s0, r1, s1 in rack_rects:
        iu = max(0.0, min(u1, r1) - max(u0, r0))
        iv = max(0.0, min(v1, s1) - max(v0, s0))
        inter += iu * iv
    return inter / area


def _rect_from_world(center, size, yaw_deg):
    """World-axis annotation box footprint -> wall-aligned (u0,v0,u1,v1)."""
    cx, cy, cz = center
    sx, sy, sz = size
    corners = np.array([[cx - sx / 2, cz - sz / 2], [cx + sx / 2, cz - sz / 2],
                        [cx + sx / 2, cz + sz / 2], [cx - sx / 2, cz + sz / 2]])
    al = _rot_xz(corners, -yaw_deg)
    return (float(al[:, 0].min()), float(al[:, 1].min()),
            float(al[:, 0].max()), float(al[:, 1].max()))


def operator_structure_centroids(out_space):
    """XZ centroids of nodes the operator RELABELLED to a structure class
    (edit/add → wall/floor/ceiling/partition/…, tolerant of typos like
    'walll'). These are authoritative 'this is structure' marks: matching
    object nodes are dropped, so an operator can fix a full-height wall that
    CLIP called 'shelf' and the geometric rules missed."""
    p = REPO / "ui" / "_spaces" / out_space / "annotations.json"
    if not p.exists():
        return []
    try:
        anns = json.loads(p.read_text()).get("annotations", [])
    except Exception:
        return []
    STRUCT = ("wall", "floor", "ceil", "partition", "window", "door",
              "curtain", "duct", "pillar", "beam", "panel")
    pts = []
    for a in anns:
        lbl = str(a.get("label", "")).lower().strip()
        c = a.get("box_center")
        if a.get("op") in ("add", "edit") and c and any(lbl.startswith(s) for s in STRUCT):
            pts.append((float(c[0]), float(c[2])))
    return pts


def load_annotation_tables(out_space, yaw_deg, floor_y):
    """Consume the viewer's Annotate output for tables (train-a-cut path). The
    operator draws/edits the correct table boxes and deletes wrong ones; those
    corrections are AUTHORITATIVE. Returns (add_boxes, delete_rects):
    add_boxes are consolidate-style {col,rect,level,...} built from the drawn
    box so they flow through the same wall-aligned box pipeline; delete_rects
    are aligned footprints to remove (any label)."""
    p = REPO / "ui" / "_spaces" / out_space / "annotations.json"
    if not p.exists() or floor_y is None:
        return [], []
    try:
        anns = json.loads(p.read_text()).get("annotations", [])
    except Exception:
        return [], []
    adds, dels = [], []
    for a in anns:
        op, bc, sz, lb = a.get("op"), a.get("box_center"), a.get("bbox_size"), a.get("label")
        if op == "delete" and bc and sz:
            dels.append(_rect_from_world(bc, sz, yaw_deg))
        elif op in ("add", "edit") and bc and sz and (lb in ("table", "desk") or not lb):
            rect = _rect_from_world(bc, sz, yaw_deg)
            u0, v0, u1, v1 = rect
            level = float(bc[1] + sz[1] / 2)
            corners = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
            world = _rot_xz(corners, yaw_deg)
            col = np.vstack([
                np.column_stack([world[:, 0], np.full(4, floor_y), world[:, 1]]),
                np.column_stack([world[:, 0], np.full(4, level), world[:, 1]]),
            ])
            adds.append({"col": col, "n_points": 400, "ann": True,
                         "mean_rgb": [61, 135, 123], "rect": rect, "level": level})
    return adds, dels


def color_guided_tables(xyz, rgb, floor_y, yaw_deg, target, tol, rack_rects=(),
                        band_lo=0.60, band_hi=1.05):
    """Recall-first tabletop detection (user: "at least detect them merged if
    you can't detect them finely"). Colour is a per-space cue the operator
    supplies via --table-rgb (reference/config, not a baked-in assumption).
    Pipeline: keep teal points at table height -> 0.08m occupancy grid ->
    morphological OPEN (severs thin blur bridges + specks, keeps table blobs
    incl. sparse corner ones) -> connected components (a real aisle = empty
    cells = separate components; a continuous plane = one region, per the
    user's merge rule) -> valley-split long rows at genuine density gaps.
    Racks (same colour) are rejected by overlap with the detected rack nodes,
    NOT by an empty-below test (that killed corner tables with storage under
    them). Returns [{col, n_points, mean_rgb, rect, level}]."""
    from scipy.ndimage import label as ndlabel, binary_opening
    if floor_y is None:
        return []
    y = xyz[:, 1]
    al_all = _rot_xz(xyz[:, [0, 2]], -yaw_deg)
    top = (y > floor_y + band_lo) & (y < floor_y + band_hi)
    teal = top & (np.linalg.norm(rgb - np.asarray(target), axis=1) < tol)
    if int(teal.sum()) < 50:
        return []
    ti = np.where(teal)[0]
    cell = 0.08
    gu0, gv0 = al_all[ti, 0].min(), al_all[ti, 1].min()
    gi = np.floor((al_all[ti, 0] - gu0) / cell).astype(np.int64)
    gj = np.floor((al_all[ti, 1] - gv0) / cell).astype(np.int64)
    nu, nv = int(gi.max()) + 1, int(gj.max()) + 1
    cnt = np.zeros((nu, nv)); np.add.at(cnt, (gi, gj), 1)
    grid = binary_opening(cnt >= 1, iterations=1)
    lab, n = ndlabel(grid, structure=np.ones((3, 3)))
    inreg = lab[gi, gj]                        # region id per teal point
    boxes = []
    for c in range(1, n + 1):
        pidx = ti[inreg == c]
        if len(pidx) < 20:
            continue
        # two-level valley split: pass 1 cuts the longer axis at gaps, pass 2
        # re-cuts each segment's (new) longer axis, so a 2-D bridged block
        # (e.g. a 4x6m blob of several tables) breaks up on both axes.
        segs2 = []
        for seg in _valley_split_long(al_all[pidx]):
            sp = pidx[seg]
            for seg2 in _valley_split_long(al_all[sp]):
                segs2.append(sp[seg2])
        for si in segs2:
            cu, cv = al_all[si, 0], al_all[si, 1]
            ext = sorted((float(np.ptp(cu)), float(np.ptp(cv))))
            if ext[0] < TABLE_MIN_SHORT_M or ext[0] * ext[1] < TABLE_MIN_AREA_M2:
                continue
            u0, u1, v0, v1 = cu.min(), cu.max(), cv.min(), cv.max()
            if _rack_overlap_frac((u0, v0, u1, v1), rack_rects) > 0.6:
                continue                        # it's a rack shelf, not a table
            level = float(np.median(y[si]))
            world = _rot_xz(np.column_stack([cu, cv]), yaw_deg)
            col = np.vstack([
                np.column_stack([world[:, 0], np.full(len(world), floor_y), world[:, 1]]),
                np.column_stack([world[:, 0], np.full(len(world), level + 0.05), world[:, 1]]),
            ])
            boxes.append({"col": col, "n_points": int(len(si)),
                          "mean_rgb": np.asarray(target).round().astype(int).tolist(),
                          "rect": (float(u0), float(v0), float(u1), float(v1)),
                          "level": level})
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, choices=space_choices())
    ap.add_argument("--geo-json", default=None,
                     help="Default: pipeline2/out/<space>_geo.json")
    ap.add_argument("--out-space", default=None,
                     help="Registered output space name (default: <space>_geo)")
    ap.add_argument("--area-min-gap-m", type=float, default=sg.AREA_MIN_GAP_M,
                     help="Min real-world gap (m) between object clusters to "
                          "count as a genuine area boundary — raise this to "
                          "merge more objects into fewer, larger areas")
    ap.add_argument("--area-max-objects", type=int, default=sg.AREA_MAX_OBJECTS,
                     help="Safety valve: force an area split above this "
                          "object count even with no qualifying gap")
    ap.add_argument("--area-max-size-m", type=float, default=sg.AREA_MAX_SIZE_M,
                     help="Safety valve: force an area split above this "
                          "footprint (m) even with no qualifying gap")
    ap.add_argument("--area-max-room-frac", type=float, default=sg.AREA_MAX_ROOM_FRAC,
                     help="An area may not span more than this fraction of "
                          "its own room's footprint (either axis)")
    ap.add_argument("--no-area-split", action="store_true",
                     help="Skip area subdivision entirely — each room "
                          "becomes exactly one area containing all its "
                          "objects")
    ap.add_argument("--no-structure-filters", action="store_true",
                     help="Skip the geometric floor/wall/slab rejectors (geo5): "
                          "bg_removal.py already separated foreground/background "
                          "up front, so these would only re-cull real objects.")
    ap.add_argument("--rooms-paper", action="store_true",
                     help="Segment rooms/areas with the Tang et al. AdAS+Hc "
                          "method (room_segment.py) and tag each object with its "
                          "room, instead of scene_graph.detect_rooms.")
    ap.add_argument("--no-audit-prune", action="store_true",
                     help="Skip the automatic geometric audit passes that run "
                          "even under --no-structure-filters (_reject_floating_"
                          "ceiling, _reject_structure_2nd_pass 'wall-level', "
                          "_reject_faulty 'impossible-person'/noise-fragment, "
                          "and strip-shaped table rejection) — for a detector-"
                          "based pipeline (pipeline4) these heuristics were "
                          "tuned against the OLD DBSCAN-cluster geometry and "
                          "can false-positive on real, plausible objects (a "
                          "normal wall-mounted shelf killed as 'wall-level', a "
                          "small real object killed as 'impossible-person'). "
                          "CLIP's own is_structure drop (door/window/wall/"
                          "floor/ceiling labels) and annotation deletes still "
                          "run regardless — this only turns off the extra "
                          "automatic geometric second-guessing.")
    ap.add_argument("--reject-bad-geometry", action="store_true",
                     help="Route two new classes of bad detector-sourced boxes "
                          "to the red review layer (non-destructive — Show "
                          "Removed to inspect/relabel): (1) a box whose raw "
                          "detector class (det_class) was structural (wall/"
                          "floor/ceiling/door/window) with reasonable "
                          "confidence but CLIP relabeled it as an ordinary "
                          "object; (2) a box whose own points don't actually "
                          "fill it — near-zero point mass at the box's own "
                          "center, or heavily skewed to one vertical extreme "
                          "— both signs the box spans mostly empty/background "
                          "space rather than one coherent object. See "
                          "_reject_structural_mismatch / _reject_hollow_or_"
                          "skewed. Best-effort statistical filter, not exact "
                          "— verified conservative (won't flag most real "
                          "objects) but will still miss some bad boxes.")
    ap.add_argument("--merge-fragments", action="store_true",
                     help="Detect same-family, touching/overlapping-footprint "
                          "objects (table/desk/counter, shelf/rack/bookshelf, "
                          "cabinet, machine, whiteboard) that are really one "
                          "physical object split into several adjacent boxes "
                          "by the detector, and fold them into a single "
                          "coarse-group unit the viewer shows collapsed by "
                          "default (see scene_graph.find_fragment_groups). "
                          "OFF by default: existing geo9-chain outputs already "
                          "resolve this via geo9_split/box_cleanup, so this is "
                          "meant for detector-based pipelines (pipeline4) that "
                          "don't.")
    ap.add_argument("--structure-debug", action="store_true",
                     help="Include the structure-debug layer (removed "
                          "floor/wall/ceiling/mezzanine segments + CLIP-"
                          "dropped structure nodes as color-coded, id-tagged "
                          "boxes for auditing). OFF by default: structure "
                          "elements are excluded from the scene graph and "
                          "viewer.")
    ap.add_argument("--table-rgb", default=None,
                     help="R,G,B (0-255) of THIS space's table tops — enables "
                          "color-guided table detection (replaces the chaotic "
                          "generic table/desk nodes with clean tabletop boxes). "
                          "Per-space; factory tables are teal '61,135,123'.")
    ap.add_argument("--table-tol", type=float, default=55.0,
                     help="RGB euclidean tolerance for the table-color match")
    ap.add_argument("--table-band", default="0.55,1.0",
                     help="table-height band above the floor: 'lo,hi' in metres")
    ap.add_argument("--table-model", default=None,
                     help="Path to a learned table-model JSON (from another "
                          "space that shares this space's table type, e.g. "
                          "reuse factory_space_13's model for _14). Ignored if "
                          "this space has its own annotations.")
    args = ap.parse_args()

    geo_path = Path(args.geo_json) if args.geo_json else \
        REPO / "pipeline2" / "out" / f"{args.space}_geo.json"
    geo = json.loads(geo_path.read_text())
    print(f"[geo→sg] {len(geo['nodes'])} geometric clusters from {geo_path}")

    # Per-node point statistics for the 3DSSG-style attributes, from
    # geo_cluster's points sidecar: mean RGB (color), voxel-occupancy
    # volume + PCA shape (see sg._point_stats — material volume and true
    # shape, which the bounding box alone can't measure).
    pts_path = geo_path.with_name(geo_path.stem + "_points.npz")
    node_rgb: dict[int, list[int]] = {}
    node_stats: dict[int, tuple] = {}
    node_xyz: dict[int, np.ndarray] = {}
    probe_xyz: np.ndarray | None = None
    if pts_path.exists():
        with np.load(pts_path) as node_pts:
            if "_probe_xyz" in node_pts:
                probe_xyz = node_pts["_probe_xyz"].astype(np.float64)
            for n in geo["nodes"]:
                rkey, xkey = f"rgb_{n['id']}", f"xyz_{n['id']}"
                if rkey in node_pts and len(node_pts[rkey]):
                    node_rgb[n["id"]] = \
                        node_pts[rkey].astype(np.float64).mean(0).round().astype(int).tolist()
                if xkey in node_pts:
                    xyz = node_pts[xkey].astype(np.float64)
                    node_xyz[n["id"]] = xyz
                    st = sg._point_stats(xyz)
                    if st is not None:
                        node_stats[n["id"]] = st
        print(f"[geo→sg] point-cloud colors for {len(node_rgb)} nodes, "
              f"shape/volume stats for {len(node_stats)} from {pts_path.name}")

    sp_paths = space(args.space)

    # Rooms/yaw first: the table-footprint consolidation cuts benches at
    # deep density valleys in the WALL-ALIGNED frame, so it needs the yaw.
    room_meta = sg.detect_rooms(
        sp_paths, eps_m=sg.ROOM_EPS_M, min_pts=sg.ROOM_MIN_PTS,
        slice_lo=sg.ROOM_SLICE_LO, slice_hi=sg.ROOM_SLICE_HI,
        subsample=sg.ROOM_SUBSAMPLE, n_rooms_hint=0,
        wall_cell_m=sg.WALL_CELL_M, wall_min_bands=sg.WALL_MIN_BANDS)
    yaw_deg = room_meta.get("_yaw_deg", 0.0)

    # ── (pipeline2b) LEARNED table detection (train-a-cut) ────────────────
    # A per-space table model (colour + height band + size prototype) is FIT
    # from the viewer annotations — nothing hardcoded — then used to detect
    # every matching table. The same procedure fits any space from its own
    # annotations (factory teal, shinhan's tables, ...). Priority: this
    # space's annotations -> a supplied --table-model -> manual --table-rgb.
    import table_model as tm
    out_space = args.out_space or f"{args.space}_geo"
    table_boxes, table_drop = [], set()
    fy = geo.get("floor_y")
    ann_path = REPO / "ui" / "_spaces" / out_space / "annotations.json"
    anns = []
    if ann_path.exists():
        try:
            anns = json.loads(ann_path.read_text()).get("annotations", [])
        except Exception:
            anns = []
    model = None
    if anns:
        cxyz, crgb = _load_cloud_rgb(args.space, sp_paths)
        model = tm.learn_table_model(cxyz, crgb, fy, yaw_deg, anns)
        if model:
            mpath = REPO / "pipeline2b" / "out" / f"{args.space}_table_model.json"
            mpath.write_text(json.dumps(model, indent=2))
            print(f"[geo→sg] learned table model from {len(anns)} annotations: "
                  f"rgb={model['rgb']} tol={model['tol']} band=[{model['band_lo']},"
                  f"{model['band_hi']}] proto={model['proto_short']}x{model['proto_long']}"
                  f" → {mpath.name}")
    elif args.table_model:
        model = json.loads(Path(args.table_model).read_text())
        cxyz, crgb = _load_cloud_rgb(args.space, sp_paths)
        print(f"[geo→sg] using supplied table model {args.table_model}")
    elif args.table_rgb:
        rgbv = [float(v) for v in args.table_rgb.split(",")]
        band = [float(v) for v in args.table_band.split(",")]
        model = {"rgb": rgbv, "tol": args.table_tol, "band_lo": band[0],
                 "band_hi": band[1], "proto_short": 0.8, "proto_long": 1.5}
        cxyz, crgb = _load_cloud_rgb(args.space, sp_paths)
        print(f"[geo→sg] manual table model rgb={rgbv}")

    if model:
        rack_rects = _aligned_rack_rects(geo["nodes"], node_xyz, yaw_deg)
        table_boxes = tm.detect_tables(cxyz, crgb, fy, yaw_deg, model, rack_rects)
        table_drop = {n["id"] for n in geo["nodes"]
                      if n.get("label") in ("table", "desk")
                      and not n.get("is_structure")}
        print(f"[geo→sg] learned-model tables: {len(table_boxes)} detected; "
              f"replace {len(table_drop)} NCut table/desk nodes")

    # Hybrid: the operator's annotated boxes are AUTHORITATIVE (exact ground
    # truth for what they corrected); the learned model generalizes to the
    # rest. Detected boxes overlapping an annotated/deleted region are dropped
    # in favour of the drawn box.
    ann_adds, ann_dels = load_annotation_tables(out_space, yaw_deg, fy)
    if ann_adds or ann_dels:
        supersede = [a["rect"] for a in ann_adds] + list(ann_dels)
        table_boxes = [b for b in table_boxes
                       if _rack_overlap_frac(b["rect"], supersede) < 0.25]
        table_boxes += ann_adds
        if anns and table_boxes:
            matched, total = tm.match_report(table_boxes, anns, yaw_deg)
            print(f"[geo→sg] tables now match {matched}/{total} annotated targets")
        print(f"[geo→sg] +{len(ann_adds)} authoritative annotated tables, "
              f"{len(ann_dels)} deletes")

    objects: dict = {}
    n_structure = 0
    for n in geo["nodes"]:
        # Semantic backstop (geo_label_clip.py): a node whose CLIP top-1
        # label is wall/floor/ceiling/door/window/... escaped the geometric
        # structure filters — drop it here rather than showing it as an
        # object in the viewer.
        if n.get("is_structure"):
            n_structure += 1
            continue
        oid = n["id"]
        if oid in table_drop:
            continue
        # bbox_pts drives apply_building_yaw's tight, wall-aligned box: every
        # node uses its own cluster points, so ALL boxes become wall-aligned
        # and percentile-trimmed instead of loose world-AABBs around a
        # building that sits at a yaw angle.
        bbox_pts = node_xyz.get(oid)
        # (pipeline2b) footprint tightening and floor-anchoring are DISABLED
        # per user request — "leave the boxes as they are initially defined".
        # tighten_footprint reshaped shelf/rack boxes, and the FLOOR_STANDING
        # anchor appended a floor-level layer so boxes reached the ground;
        # both changed the raw box. Keep the detector's own points only.
        # if bbox_pts is not None and n.get("label") in FOOTPRINT_TIGHTEN_LABELS:
        #     bbox_pts = tighten_footprint(bbox_pts, yaw_deg)
        # fl = geo.get("floor_y")
        # if (bbox_pts is not None and len(bbox_pts) >= 6
        #         and n.get("label") in FLOOR_STANDING_LABELS and fl is not None
        #         and bbox_pts[:, 1].min() > fl + 0.05):
        #     base = bbox_pts.copy(); base[:, 1] = fl
        #     bbox_pts = np.vstack([bbox_pts, base])
        if bbox_pts is not None and len(bbox_pts) >= 6:
            # Detector-sourced nodes (det_class present) get a much gentler
            # trim than the default — see sg.apply_building_yaw's docstring
            # for why BBOX_TRIM_Q collapses a real dimension on pipeline4's
            # denser, more uniform point crops (verified concretely: a real
            # ~0.85m-deep rack came out at 0.2m). This is the SAME final
            # bbox apply_building_yaw recomputes anyway once yaw is known,
            # but room/area assignment below runs on this one first.
            q = 0.01 if n.get("det_class") is not None else sg.BBOX_TRIM_Q
            bmin, bmax = sg._trimmed_bbox(bbox_pts, q=q)
        else:
            bmin = np.array(n["bbox_min"])
            bmax = np.array(n["bbox_max"])
        size = bmax - bmin
        vol = float(max(np.prod(np.maximum(size, 1e-6)), 1e-6))
        objects[oid] = {
            "centroid":    ((bmin + bmax) / 2).tolist(),
            "bbox_pts":    bbox_pts,
            "bbox_min":    bmin.tolist(),
            "bbox_max":    bmax.tolist(),
            "bbox_size":   size.tolist(),
            "sigma":       (size / 3.46).tolist(),  # rough std proxy (uniform-box approx)
            "volume":      vol,
            "max_side":    float(size.max()),
            "n_proposals": int(n["n_points"]),
            "n_world_pts": int(n["n_points"]),
            "label":       n.get("label", f"obj_{oid}"),
            "clip_topk":   n.get("clip_topk", []),
            "mean_rgb":    n.get("mean_rgb") or node_rgb.get(oid),
            "material":    n.get("material"),
            "state":       n.get("state"),
            "occ_volume":  node_stats.get(oid, (None, None))[0],
            "point_shape": node_stats.get(oid, (None, None))[1],
            # Only present for detector-sourced nodes (pipeline4) — the
            # raw learned-detector class/confidence, BEFORE CLIP relabeled
            # it. Used by _reject_structural_mismatch to catch a wall/
            # window/door CLIP misread as an ordinary object; absent
            # (None) for pipeline2b's own geometric-clustering nodes, so
            # that check is naturally a no-op there.
            "det_class":   n.get("det_class"),
            "det_prob":    n.get("det_prob"),
        }

    # Consolidated table boxes as fresh object nodes.
    next_id = (max((n["id"] for n in geo["nodes"]), default=0) + 1)
    for tb in table_boxes:
        col = tb["col"]
        bmin, bmax = sg._trimmed_bbox(col)
        size = bmax - bmin
        objects[next_id] = {
            "centroid":    ((bmin + bmax) / 2).tolist(),
            "bbox_pts":    col,
            "bbox_min":    bmin.tolist(),
            "bbox_max":    bmax.tolist(),
            "bbox_size":   size.tolist(),
            "sigma":       (size / 3.46).tolist(),
            "volume":      float(max(np.prod(np.maximum(size, 1e-6)), 1e-6)),
            "max_side":    float(size.max()),
            "n_proposals": tb["n_points"],
            "n_world_pts": tb["n_points"],
            "label":       "table",
            "clip_topk":   [{"label": "table", "score": 1.0}],
            "mean_rgb":    tb["mean_rgb"],
            "material":    None, "state": None,
            "occ_volume":  None, "point_shape": None,
            "_ann":        bool(tb.get("ann")),   # authoritative user box
        }
        next_id += 1
    print(f"[geo→sg] {len(objects)} objects carried through "
          f"({n_structure} dropped by CLIP structure label; "
          f"{len(table_drop)} table nodes → {len(table_boxes)} tabletop boxes)")

    objects = sg.assign_rooms(objects, room_meta)
    wall_blocker = build_wall_blocker(geo.get("wall_grid"))
    objects, areas = sg.build_areas(objects, min_gap_m=args.area_min_gap_m,
                                     max_objects_per_area=args.area_max_objects,
                                     max_area_size_m=args.area_max_size_m,
                                     max_room_frac=args.area_max_room_frac,
                                     wall_blocker=wall_blocker,
                                     split=not args.no_area_split)
    objects = sg.build_attributes(objects)

    objects = sg.apply_building_yaw(objects, yaw_deg)

    # ── (pipeline2b) reject floor-slab / wall-panel / empty boxes ─────────
    # User request: remove boxes drawn flat on the floor (floor residue),
    # boxes that are thin tall wall panels (e.g. a wall fragment labeled
    # "shelf"), and boxes that don't actually enclose points (empty space —
    # judged by a point fill ratio inside the box). Uses each object's own
    # points (bbox_pts, world) and its wall-aligned bbox_size.
    removed_structure = []   # boxes the geo5 passes dropped (shown red for audit)
    if args.no_structure_filters:
        # geo5: foreground/background separation already done up front by
        # bg_removal.py, so the geometric wall/floor/slab rejectors here would
        # only re-cull real objects (they cost geo4 ~12 shelves). Skip them,
        # but do sweep residual ceiling-confined clutter (ducts/beams/fixtures
        # that leaked through bg_removal's point-level pass) with an
        # object-safe geometric test.
        from collections import Counter
        if not args.no_audit_prune:
            float_dropped = _reject_floating_ceiling(objects, geo.get("floor_y"),
                                                     geo.get("ceil_y"))
            removed_structure += float_dropped
            if float_dropped:
                print(f"[geo→sg] dropped {len(float_dropped)} floating ceiling-"
                      f"clutter boxes: {dict(Counter(r['reason'] for r in float_dropped))}")
            # Second geometric pass: drop residual structure boxes (thin panels
            # on detected wall lines, floor slabs, paper-thin planes) whatever
            # CLIP called them. Uses the detected wall_grid so free-standing
            # shelves are spared.
            s2 = _reject_structure_2nd_pass(objects, geo.get("floor_y"),
                                            geo.get("ceil_y"), geo.get("wall_grid"))
            removed_structure += s2
            if s2:
                _c = Counter(r["label"] + ":" + r["reason"] for r in s2)
                print(f"[geo→sg] 2nd-pass dropped {len(s2)} structure boxes: {dict(_c)}")
        else:
            print("[geo→sg] --no-audit-prune: skipping floating-ceiling + "
                  "2nd-pass structure rejection")
        # Honor operator structure-relabels: any node the operator relabelled to
        # wall/floor/ceiling/etc. is dropped (fixes full-height walls that CLIP
        # called 'shelf' and geometry missed — normals too noisy to decide here).
        struct_pts = operator_structure_centroids(out_space)
        if struct_pts:
            rm = [oid for oid, o in objects.items()
                  if (lambda c: any((c[0] - px) ** 2 + (c[2] - pz) ** 2 < 0.7 ** 2
                                    for px, pz in struct_pts))(o.get("box_center") or o["centroid"])]
            for oid in rm:
                o = objects[oid]
                removed_structure.append({"oid": oid, "label": o.get("label"),
                    "reason": "operator-wall", "box_center": list(o.get("box_center") or o["centroid"]),
                    "bbox_size": list(o["bbox_size"]), "pts": o.get("bbox_pts")})
                del objects[oid]
            if rm:
                print(f"[geo→sg] operator structure-relabels dropped {len(rm)} nodes")
        # Precision pass: route noise fragments + class-impossible boxes (open-
        # vocab false 'person' hits, tiny geometric shards) to the red review
        # layer. Non-destructive — reviewable via Show Removed, relabel to keep.
        if not args.no_audit_prune:
            faulty = _reject_faulty(objects)
            removed_structure += faulty
            if faulty:
                print(f"[geo→sg] faulty prune → {len(faulty)} boxes to red: "
                      f"{dict(Counter(r['reason'] for r in faulty))}")
        if args.reject_bad_geometry:
            co = _reject_clip_structural_override(objects)
            removed_structure += co
            if co:
                print(f"[geo→sg] CLIP-structural-override prune → {len(co)} "
                      f"boxes to red: {dict(Counter(r['reason'] for r in co))}")
            sm = _reject_structural_mismatch(objects, floor_y=geo.get("floor_y"),
                                             ceil_y=geo.get("ceil_y"))
            removed_structure += sm
            if sm:
                print(f"[geo→sg] structural-mismatch prune → {len(sm)} boxes "
                      f"to red: {dict(Counter(r['reason'] for r in sm))}")
            # hollow-or-skewed is deliberately NOT called anywhere in this
            # pipeline (see the disabled call and comment right after
            # fragment-merge below, for why): verified it was deleting real,
            # legitimate table fragments that simply never found a merge
            # partner (e.g. because their true partner got swept into an
            # oversized rejected component elsewhere in find_fragment_groups)
            # — an ungrouped fragment looking geometrically "incomplete" on
            # its own is not evidence it's a bad detection.
            fh = _reject_floor_hugging(objects, geo.get("floor_y"),
                                       ceil_y=geo.get("ceil_y"),
                                       y_up_sign=float(geo.get("y_up_sign", 1.0)))
            removed_structure += fh
            if fh:
                print(f"[geo→sg] floor-hugging prune → {len(fh)} boxes to red")
            hf = _reject_implausible_height_fraction(objects, geo.get("floor_y"),
                                                     geo.get("ceil_y"))
            removed_structure += hf
            if hf:
                print(f"[geo→sg] spans-room-height prune → {len(hf)} boxes to red")
        print("[geo→sg] --no-structure-filters: skipping geometric "
              "floor/wall/slab rejection (bg_removal handled structure)")
    else:
        dropped = _reject_floor_wall_empty(objects, geo.get("floor_y"))
        if dropped:
            from collections import Counter
            print(f"[geo→sg] rejected {len(dropped)} boxes: "
                  f"{dict(Counter(r for _, r in dropped))}")

        # Tall vertical planar sheets = wall / partition fragments (user
        # request: remove wall elements boxed as shelf/etc). Judged from each
        # object's own points; reports every removal by id+label so specific
        # false positives can be vetoed.
        wall_dropped = _reject_wall_slabs(objects, geo.get("floor_y"))
        if wall_dropped:
            print(f"[geo→sg] rejected {len(wall_dropped)} wall/partition slabs: "
                  + ", ".join(f"#{oid}({lbl})" for oid, lbl in wall_dropped))

    # Annotation deletes remove ANY object whose aligned centroid falls in a
    # deleted region (lets the operator delete wrong non-table boxes too).
    if ann_dels:
        rm = []
        for oid, o in objects.items():
            c = o.get("box_center") or o["centroid"]
            cu, cv = _rot_xz(np.array([[c[0], c[2]]]), -yaw_deg)[0]
            if any(u0 <= cu <= u1 and v0 <= cv <= v1 for u0, v0, u1, v1 in ann_dels):
                rm.append(oid)
        for oid in rm:
            del objects[oid]
        if rm:
            print(f"[geo→sg] annotation deletes removed {len(rm)} objects")

    # Drop table boxes shaped like a strip, not a tabletop (a wall-top or
    # shelf-edge line caught at tabletop height): too shallow, or a plank-
    # like aspect ratio no real table has.
    if not args.no_audit_prune:
        sliver = []
        for oid, o in objects.items():
            if o.get("label") != "table" or o.get("_ann"):
                continue
            short, long = sorted((o["bbox_size"][0], o["bbox_size"][2]))
            if short < 0.35 or (long / max(short, 1e-6)) > 5.0:
                sliver.append(oid)
        for oid in sliver:
            del objects[oid]
        if sliver:
            print(f"[geo→sg] dropped {len(sliver)} strip-shaped table boxes "
                  f"(not tabletop-shaped)")

    # --merge-fragments (opt-in): fold same-family, touching/overlapping-
    # footprint objects that are really ONE physical object split into
    # several adjacent boxes by the detector into ONE real graph node (see
    # sg.find_fragment_groups / sg.build_coarse_objects) — not an overlay on
    # top of the fine-grained fragments, an actual replacement. Runs on the
    # FINAL objects — post CLIP labeling, post structure-drop/table-
    # consolidation/sliver-rejection — so it uses the real semantic label,
    # not a detector's raw class guess. Edges, rooms and every downstream
    # computation below operate on the merged (coarse) objects: relations
    # between two fragments now inside the same merged object simply don't
    # exist anymore, and a relation between fragments in two DIFFERENT
    # merged objects is recomputed fresh on the merged geometry rather than
    # inherited from the fragment level — a merged object's box is not the
    # same shape as any one of its fragments, so a stale fragment-level
    # relation could easily be wrong (e.g. "close_by" a neighbour that the
    # merged box no longer comes near, or missing one it now touches).
    frag_groups = sg.find_fragment_groups(objects) if args.merge_fragments else []
    if args.merge_fragments:
        objects, fine_fragments = sg.build_coarse_objects(objects, frag_groups,
                                                           yaw_deg=yaw_deg)
    else:
        fine_fragments = {}

    # hollow-or-skewed is disabled (not just reordered — genuinely not
    # called): it checks whether an object's own points fill its own box,
    # which is a bad test for a fragment that never found a merge partner.
    # Verified directly (factory_space_13): table fragment #118 (reported
    # by the user as a wrongly-removed real table, id #200014 in the red-
    # audit renumbering) never merges because its true partner (#170) got
    # swept into a 47-member component find_fragment_groups discards
    # wholesale for being too large — #118 is a completely real table, its
    # only "problem" is being an ungrouped fragment, which hollow-or-skewed
    # can't distinguish from an actual bad detection. Per explicit
    # instruction: leave the wholesale-rejection behavior in
    # find_fragment_groups alone, just stop deleting ungrouped fragments —
    # they now surface as ordinary (if numerous and small) standalone
    # table boxes instead of disappearing into the red audit layer.

    edges = sg.build_edges(
        objects,
        above_delta=sg.ABOVE_DELTA_M, hanging_delta=sg.HANGING_DELTA_M,
        max_direct_gap=sg.MAX_DIRECT_GAP_M, max_hang_gap=sg.MAX_HANG_GAP_M,
        footprint_iou_thr=sg.FOOTPRINT_IOU_THR, on_floor_m=sg.ON_FLOOR_M,
        wall_blocker=wall_blocker, yaw_deg=yaw_deg)

    out_space = args.out_space or f"{args.space}_geo"
    graph = sg.serialise(objects, edges, room_meta, areas, out_space,
                         yaw_deg=yaw_deg, fine_fragments=fine_fragments)

    # ── (geo5) Paper room/area segmentation (Tang et al. AdAS+Hc) ──────────
    # Divide the floor plan into rooms with room_segment.py and re-tag each
    # object by the room it stands in, replacing scene_graph.detect_rooms.
    # Contained + guarded: on any failure we keep the detect_rooms result.
    if args.rooms_paper and probe_xyz is not None:
        try:
            import room_segment as rs
            seg_kind = {s["id"]: s["kind"]
                        for s in geo.get("structure_segments", [])}
            wall_pts = []
            if pts_path.exists():
                with np.load(pts_path) as _z:
                    for sid, k in seg_kind.items():
                        key = f"sxyz_{sid}"
                        if k in ("wall", "partition") and key in _z:
                            wall_pts.append(_z[key].astype(np.float64))
            wall_xyz = (np.concatenate(wall_pts) if wall_pts
                        else np.zeros((0, 3)))
            rmap = rs.segment_rooms(probe_xyz, wall_xyz)
            na = 0
            for nd in graph["nodes"]:
                c = nd.get("box_center") or nd.get("centroid")
                if c:
                    rid = rmap.room_of(float(c[0]), float(c[2]))
                    if rid > 0:
                        nd["room_id"] = int(rid)
                        na += 1
            if isinstance(graph.get("stats"), dict):
                graph["stats"]["n_rooms"] = rmap.n_rooms
            print(f"[geo→sg] rooms-paper (AdAS+Hc): {rmap.n_rooms} rooms, "
                  f"{na} objects tagged")
        except Exception as e:
            print(f"[geo→sg] rooms-paper failed ({e!r}); kept detect_rooms")

    # (pipeline2b) Drop the coarse "unit" layer (desk unit / rack unit Gn) per
    # user request — show only the individual object boxes, no enclosing
    # units. --merge-fragments is the one case that keeps it: without the
    # coarse layer there's no way to collapse a detector's over-segmented
    # fragments back into one box, which is the whole point of that flag.
    if not args.merge_fragments:
        graph["coarse_groups"] = []
        if isinstance(graph.get("meta"), dict):
            graph["meta"]["n_coarse_groups"] = 0

    # ── Removed-structure RED audit layer ────────────────────────────────
    # Every box the geo5 wall/floor passes dropped, appended as an inert node
    # (structure_debug + removed_box=True → rendered RED in the viewer) so the
    # operator can see exactly what was removed and catch any real object that
    # was wrongly cut. Boxes are wall-aligned via the same yaw as the objects.
    if removed_structure:
        for i, r in enumerate(removed_structure):
            c = r.get("box_center"); s = r.get("bbox_size")
            if not c or not s:
                continue
            graph["nodes"].append({
                "id": 200000 + i,
                "uid": f"removed-{r.get('reason','?')}-{i}",
                "group_id": -1,
                "label": f"removed_{r.get('reason','?')} ({r.get('label','?')})",
                "caption": "", "label_score": 0.0, "label_entropy": 0.0,
                "centroid": [round(float(v), 4) for v in c],
                "box_center": [round(float(v), 4) for v in c],
                "bbox_size": [round(float(v), 4) for v in s],
                "room_id": -1, "area_id": -1,
                "n_proposals": 0, "n_world_pts": 0, "on_floor": False,
                "n_absorbed": 0, "absorbed_ids": [], "class_hierarchy": [],
                "attributes": {}, "affordances": [],
                "structure_debug": True, "removed_box": True,
            })
        print(f"[geo→sg] RED audit layer: {len(removed_structure)} removed "
              f"structure boxes appended (removed_box=True)")

    # ── Structure-debug layer ─────────────────────────────────────────────
    # Everything the pipeline REMOVED, appended as inert viewer nodes so it
    # can be audited by eye: (a) SE segments the paper-style structure stage
    # removed (floor/ceiling/wall/mezzanine, from se_discovery.py), and
    # (b) nodes the CLIP structure backstop dropped. Each renders as a
    # color-coded box with its id — report an id that is actually a real
    # object (a rack face flagged wall, a table flagged ceiling) and the
    # corresponding rule can be tuned against exactly that segment.
    if args.structure_debug:
        def _yaw_bbox(pts: np.ndarray):
            """(bmin, bmax, world box center) — percentile-trimmed in the
            building-aligned frame so a handful of stray points can't
            inflate or shift a debug box off its segment."""
            ptsr = pts
            if abs(yaw_deg) >= 0.05:
                xz = sg._rotate_xz_deg(pts[:, [0, 2]], -yaw_deg)
                ptsr = np.column_stack([xz[:, 0], pts[:, 1], xz[:, 1]])
            bmin = np.percentile(ptsr, 2.0, axis=0)
            bmax = np.percentile(ptsr, 98.0, axis=0)
            mid = (bmin + bmax) / 2.0
            if abs(yaw_deg) >= 0.05:
                mxz = sg._rotate_xz_deg(np.array([[mid[0], mid[2]]]), yaw_deg)[0]
                center = [float(mxz[0]), float(mid[1]), float(mxz[1])]
            else:
                center = [float(v) for v in mid]
            return bmin, bmax, center

        n_dbg = 0
        if pts_path.exists():
            with np.load(pts_path) as npz:
                for s in geo.get("structure_segments", []):
                    key = f"sxyz_{s['id']}"
                    if key not in npz:
                        continue
                    pts = npz[key].astype(np.float64)
                    bmin, bmax, bctr = _yaw_bbox(pts)
                    graph["nodes"].append({
                        "id": 100000 + int(s["id"]),
                        "uid": f"struct-{s['kind']}-{s['id']}",
                        "group_id": -1, "label": s["kind"], "caption": "",
                        "label_score": 1.0, "label_entropy": 0.0,
                        "centroid": [round(float(v), 4) for v in pts.mean(0)],
                        "box_center": [round(float(v), 4) for v in bctr],
                        "bbox_size": [round(float(v), 4) for v in (bmax - bmin)],
                        "room_id": -1, "area_id": -1,
                        "n_proposals": int(s["n_points"]),
                        "n_world_pts": int(s["n_points"]),
                        "on_floor": False, "n_absorbed": 0, "absorbed_ids": [],
                        "class_hierarchy": [], "attributes": {},
                        "affordances": [], "structure_debug": True,
                    })
                    n_dbg += 1
                for n in geo["nodes"]:
                    if not n.get("is_structure"):
                        continue
                    key = f"xyz_{n['id']}"
                    if key not in npz:
                        continue
                    pts = npz[key].astype(np.float64)
                    bmin, bmax, bctr = _yaw_bbox(pts)
                    graph["nodes"].append({
                        "id": int(n["id"]),
                        "uid": f"dropped-{n['id']}",
                        "group_id": -1,
                        "label": n.get("label", f"obj_{n['id']}"),
                        "caption": "", "label_score": 0.0, "label_entropy": 0.0,
                        "centroid": [round(float(v), 4) for v in pts.mean(0)],
                        "box_center": [round(float(v), 4) for v in bctr],
                        "bbox_size": [round(float(v), 4) for v in (bmax - bmin)],
                        "room_id": -1, "area_id": -1,
                        "n_proposals": int(n["n_points"]),
                        "n_world_pts": int(n["n_points"]),
                        "on_floor": False, "n_absorbed": 0, "absorbed_ids": [],
                        "class_hierarchy": [], "attributes": {},
                        "affordances": [], "structure_debug": True,
                    })
                    n_dbg += 1
        print(f"[geo→sg] structure-debug layer: {n_dbg} removed-structure "
              f"boxes appended (color-coded in the viewer; use their ids "
              f"to report misclassified structure)")

    out_dir = REPO / "out" / f"geo_{out_space}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene_graph.json").write_text(json.dumps(graph, indent=2))
    print(f"[geo→sg] → {out_dir / 'scene_graph.json'}")

    # Auto-scaffold the viewer space (Data_ symlinks, viewer symlink,
    # index.html, topdown assets) — copies instantly from any registered
    # sibling that shares this space's data_root, so a freshly-registered
    # space becomes viewable without manual setup. Uses out_space's own
    # registered title for the index page when out_space is registered;
    # falls back to the base space's paths otherwise.
    try:
        scaffold_paths = space(out_space) if out_space in space_choices() else sp_paths
    except Exception:
        scaffold_paths = sp_paths
    sg._ensure_ui_space(out_space, scaffold_paths)

    web_dir = REPO / "ui" / "_spaces" / out_space
    if (web_dir / "Data_" / "downsampled_web.ply").exists():
        dest = web_dir / "scene_graph.json"
        dest.write_text(json.dumps(graph, indent=2))
        print(f"[geo→sg] viewer copy → {dest}")
        sg.render_overlay(graph, out_space)
        sg.render_hydra_diagram(graph, out_space)
    else:
        print(f"[geo→sg] {web_dir} not scaffolded yet — skipping viewer overlay render")

    sg.print_summary(graph)
    print(f"[geo→sg] Done — {graph['stats']['n_nodes']} nodes · "
          f"{graph['stats']['n_edges']} edges · {graph['stats']['n_rooms']} rooms")


if __name__ == "__main__":
    main()
