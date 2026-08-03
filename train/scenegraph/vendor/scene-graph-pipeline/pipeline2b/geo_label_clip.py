#!/usr/bin/env python3
"""
Pipeline 2, step 2: CLIP semantic labeling of the geometric nodes produced
by geo_cluster.py.

For each node in pipeline2/out/<space>_geo.json:
  1. Project the node's 3D bbox into every perspective view (cameras.json
     pose + shared intrinsics; p_cam = R^T (p_world - pos), +Y down OpenCV
     convention — same as pipeline/03_backproject.py's inverse).
  2. Pick the best few views (bbox well inside the frame, not too far, seen
     from different scanpoints for occlusion robustness).
  3. Crop the projected bbox region out of each view, CLIP-encode the crops,
     average into one embedding per node (per-instance CLIP feature
     aggregation in the style of OpenMask3D — Takmaz et al., NeurIPS 2023 —
     which classifies class-agnostic 3D instance masks by pooling CLIP
     features of their best 2D views; CLIP itself: Radford et al., ICML
     2021, via open_clip ViT-H-14/dfn5b).
  4. Zero-shot classify against a small indoor/factory vocabulary and write
     `label`, `clip_topk`, and `is_structure` back into the geo JSON.
     Nodes whose best label is structural (wall/floor/ceiling/...) are
     flagged so geo_to_scenegraph.py drops them — a semantic backstop for
     the geometric structure filters.

With --refine, adds an experimental recursive container-refinement pass
(see the "Recursive container refinement" comment in main()) that can
subdivide desks/shelves/racks/cabinets further when CLIP confirms the
sub-pieces are distinct objects. Off by default: it produces richer graphs
but less stable node identity across re-runs, so flat one-label-per-node
labeling is the production default.

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
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))
from _paths import space, space_choices  # noqa: E402
import clip_utils  # noqa: E402

# Vocabulary: the object types actually present in these office/factory
# scans, plus the structural classes used as a semantic rejection backstop.
VOCAB = [
    "table", "desk", "office chair", "person", "shelf", "storage rack",
    "cabinet", "cardboard box", "pallet", "machine", "computer monitor",
    "plant", "trash bin", "partition panel", "whiteboard", "sofa", "bench",
    "ladder", "cart", "ceiling light", "air conditioner",
    "fire extinguisher", "workbench",
    # structural classes (nodes labeled with these get dropped):
    "door", "window", "wall", "floor", "ceiling", "pillar", "curtain",
    "air duct",
]
# "bench" added after shinhan_space's table 8 turned out to be a real, dense
# bench that CLIP still called "table" from real photos - not noise, but a
# vocabulary gap: "bench" was never an option for it to compete for. Any
# label a real object could plausibly be needs to be IN VOCAB before a
# CLIP-based check can ever catch a mislabel for it - a closed-set
# classifier can only ever answer with a name it's given.
# Functional groups: labels a real object could legitimately be read as
# interchangeably by CLIP without that being evidence of a false positive
# (a desk photographed at a bad angle scoring "workbench" first isn't
# wrong, it's the same kind of thing). This is vocabulary-level config -
# reusable for ANY space using this VOCAB, not a per-space or per-object
# override. Tried deriving these automatically from CLIP's own text-
# embedding similarity between VOCAB entries first (no manual list at
# all) - confirmed directly it doesn't work: "desk" comes back nearly as
# similar to "office chair" (0.86) and "cabinet" (0.85) as to "workbench"
# (0.87), so a similarity threshold can't separate genuine near-synonyms
# from merely contextually-associated office objects. An explicit,
# reviewable list is more honest than a threshold tuned to one example.
FUNCTIONAL_GROUPS = [
    {"table", "desk", "workbench", "bench"},
    {"office chair", "sofa"},
    {"shelf", "storage rack", "cabinet"},
    {"cart", "pallet"},
    {"window", "air conditioner"},   # both wall-mounted rectangular fixtures with a visible grid/pane pattern - confirmed genuinely confusable, not just nearby in embedding space
]


# Detection-label -> VOCAB-label aliases: OWLv2 detection prompts and VOCAB
# use different canonical wording for the same real class (OWLv2 prompt
# says "chair", VOCAB's own class is "office chair"). This is a permanent,
# vocabulary-level fact, not a per-run setting - resolved automatically
# here so callers never need to remember to pass a --clip-label override.
LABEL_ALIASES = {
    "chair": "office chair",
    "light": "ceiling light",
}
# Reverse direction: when relabeling a box INTO one of these VOCAB names,
# use the detection-side convention instead (matches what the rest of this
# project's viewer/scene-graph/legend code already expects as the label
# string) - without this a relabel would silently introduce a second,
# differently-spelled label for a class that already exists under its own
# detection-side name (e.g. ending up with both "chair" and "office chair"
# as separate categories for the same real thing).
REVERSE_LABEL_ALIASES = {v: k for k, v in LABEL_ALIASES.items()}


def label_group(label: str) -> set[str]:
    """The set of VOCAB labels acceptable as a match for a detection `label`
    (including its own VOCAB-resolved name)."""
    resolved = LABEL_ALIASES.get(label, label)
    for g in FUNCTIONAL_GROUPS:
        if resolved in g:
            return g
    return {resolved}


# "partition panel" is deliberately a structure label: visually a bare wall
# patch and a real partition are the same white plane, and no geometric
# discrimination (wall-grid adjacency, one-side emptiness) reliably tells
# them apart — an interior wall has a room, hence points, on both sides
# just like a real partition does. Keeping the label in VOCAB lets wall
# patches match it instead of stealing a real object's label.
STRUCTURE_LABELS = {"door", "window", "wall", "floor", "ceiling", "pillar",
                    "curtain", "air duct", "partition panel"}

# Objects that are, BY PHYSICAL NATURE (true in any space, not a per-space
# fact), thin flat panels mounted flush to a wall - real world width/height
# should each be several times the object's own depth. Used to sanity-check
# a detected box's geometry independent of its label/CLIP score: a "whiteboard"
# box that's roughly cube-shaped almost certainly mixed depth layers during
# backprojection (e.g. the real wall panel plus foreground clutter caught in
# the same 2D detection box) and needs a PCA-based planar refit, not a label
# change - see pipeline9/refit_planar_objects.py.
PLANAR_LABELS = {"whiteboard", "window"}

# Custom prompt ensembles for classes CLIP confuses in these scans — a tube
# light photographed from below IS mostly ceiling pixels, so the generic
# "a photo of a ceiling light" prompt loses to "ceiling" without these; a
# bare white wall patch similarly scores high on "whiteboard" without a
# prompt that requires a frame/tray/writing.
PROMPTS = {
    "ceiling light": [
        "a fluorescent tube light fixture mounted on a ceiling",
        "a bright LED tube lamp on the ceiling of an industrial building",
        "a glowing light fixture seen from below",
        "a rectangular LED panel light on a ceiling",
    ],
    "ceiling": [
        "a plain ceiling surface with no light fixture",
        "an empty stretch of ceiling in an industrial building",
    ],
    "whiteboard": [
        "a whiteboard on a rolling stand with writing on it",
        "a framed whiteboard with markers and an eraser tray",
    ],
    "machine": [
        "a large industrial machine standing on a factory floor",
        "industrial manufacturing equipment with panels and cables",
    ],
    "table": [
        "a table with a flat top standing on legs",
        "a work table in a workshop",
    ],
    "bench": [
        "a long narrow bench meant for sitting, with no backrest",
        "a wooden or metal bench seat, longer than it is deep",
        "a low bench in a workshop or waiting area, not a worktable",
    ],
    "air conditioner": [
        "a wall-mounted or ceiling-mounted air conditioner unit with vents",
        "a white rectangular AC unit box, not a window and not a duct",
        "an indoor split air conditioning unit mounted high on a wall",
    ],
    "window": [
        "a window with glass panes in a wall, showing outside light",
        "a glass window with a visible frame",
    ],
    "fire extinguisher": [
        "a red fire extinguisher cylinder mounted on a wall",
        "a small red pressurized canister with a hose and gauge",
    ],
    "workbench": [
        "a sturdy workbench with tools and equipment on top",
        "a work table used for building or repairing things",
    ],
    "storage rack": [
        "a tall metal storage rack with multiple shelf levels",
        "warehouse pallet racking holding boxes and goods",
    ],
    "wall": [
        "a plain white wall surface of a room",
        "a bare empty wall",
    ],
}

# ── 3DSSG-style attribute probing (Wald et al. CVPR 2020, §3.2) ──────────
# Material and state are zero-shot CLIP text probes evaluated on the SAME
# averaged crop embedding classify_node already computes for the label —
# an extra text-bank dot product, no additional image encoding. 3DSSG's
# material/texture attributes come from manual annotation; CLIP probing is
# the automated best-effort equivalent.
MATERIALS = ["wood", "metal", "plastic", "fabric", "leather", "glass",
             "cardboard", "concrete", "ceramic"]
MATERIAL_PROMPTS = {m: [f"a photo of an object made of {m}",
                        f"a {m} surface"] for m in MATERIALS}

# Dynamic state (3DSSG "dynamic properties"): probed only for classes where
# a state is visually decidable from a crop. Keys are VOCAB labels (before
# underscore normalization); each maps to the candidate state phrases.
STATE_PROBES: dict[str, dict[str, list[str]]] = {
    "door":    {"open":   ["an open door with the doorway visible"],
                "closed": ["a closed door"]},
    "window":  {"open":   ["an open window"],
                "closed": ["a closed window"]},
    "cabinet": {"open":   ["a cabinet with its doors open"],
                "closed": ["a cabinet with its doors closed"]},
    "computer monitor": {"on":  ["a computer monitor that is turned on, showing content"],
                         "off": ["a computer monitor with a black powered-off screen"]},
    "machine": {"on":  ["an industrial machine that is running, lights on"],
                "off": ["an industrial machine that is switched off"]},
    "ceiling light": {"on":  ["a glowing ceiling light that is switched on"],
                      "off": ["a ceiling light fixture that is switched off"]},
}

# ── Geometric shape priors ───────────────────────────────────────────────
# CLIP top-k scores on small crops are frequently near-ties (a table seen
# obliquely scores 0.236 "machine" / 0.235 "cart" / 0.235 "desk"), and the
# argmax is then effectively random. The node's 3D box knows better: a
# 0.67m-tall, 1.1x2.6m flat box is table-shaped, not machine-shaped.
# Each entry gives plausible bands per feature; outside a band the prior
# falls off linearly to PRIOR_MIN (never 0 — decisive CLIP evidence can
# still override a soft shape violation). Features:
#   h        bbox height (m)
#   long     larger horizontal dimension
#   short    smaller horizontal dimension
#   top/bot  top/bottom of the box above the floor
#   ceil_gap distance from the box top up to the ceiling
SHAPE_PRIORS: dict[str, dict[str, tuple[float | None, float | None]]] = {
    "table":            {"h": (0.25, 1.15), "top": (0.5, 1.25), "long": (0.7, 4.0)},
    "desk":             {"h": (0.25, 1.25), "top": (0.55, 1.3), "long": (0.7, 3.5)},
    "office chair":     {"h": (0.45, 1.5), "long": (0.3, 1.1)},
    "person":           {"h": (0.8, 2.0), "long": (0.25, 1.2)},
    "shelf":            {"h": (0.7, 2.8), "top": (0.9, None)},
    "storage rack":     {"h": (1.2, None), "top": (1.5, None)},
    "cabinet":          {"h": (0.5, 2.4), "long": (0.35, 2.5)},
    "cardboard box":    {"h": (0.08, 1.4), "long": (0.15, 1.6)},
    "pallet":           {"h": (None, 0.65), "top": (None, 0.9), "long": (0.5, 3.0)},
    "machine":          {"h": (0.75, None), "long": (0.5, None),
                         "bot": (None, 0.3), "top": (0.9, None)},
    "computer monitor": {"h": (0.15, 0.75), "long": (0.25, 1.2), "top": (0.55, 2.2)},
    "plant":            {"h": (0.2, 2.8)},
    "trash bin":        {"h": (0.25, 1.3), "long": (0.15, 1.0)},
    "whiteboard":       {"h": (0.5, 2.1), "short": (None, 0.45), "top": (0.9, 2.3)},
    "sofa":             {"h": (0.45, 1.2), "long": (1.2, 3.2)},
    "ladder":           {"h": (1.0, None), "short": (None, 1.0)},
    "cart":             {"h": (0.45, 1.4), "long": (0.4, 1.8),
                         "bot": (None, 0.3)},
    "ceiling light":    {"bot": (1.8, None)},
    "air conditioner":  {"h": (0.15, 0.65), "bot": (1.4, None), "short": (None, 0.55)},
    "door":             {"h": (1.4, None), "top": (1.7, None), "short": (None, 0.5)},
    "window":           {"bot": (0.4, None)},
    "wall":             {"h": (1.0, None)},
    "ceiling":          {"h": (None, 0.7), "ceil_gap": (None, 0.7)},
    "floor":            {"h": (None, 0.3), "bot": (None, 0.15),
                         "top": (None, 0.35)},
    "pillar":           {"h": (1.6, None), "long": (None, 1.3)},
    "curtain":          {"h": (0.9, None)},
    "air duct":         {"bot": (1.6, None)},
    "partition panel":  {"h": (0.9, None), "short": (None, 0.4)},
    # "workbench", "bench", "fire extinguisher" were all in VOCAB with no
    # SHAPE_PRIORS entry - the same gap that let "air conditioner" sweep
    # shinhan's tables (see that fix above): shape_prior() returns a neutral
    # 1.0 for any label missing here, so on a close/ambiguous crop it wins
    # by default even against a wildly-wrong-sized box, while every real
    # competing label (table/shelf/storage rack) pays a shape discount.
    # Confirmed directly on factory14: a 3.72m-tall and a 2.47m-tall box
    # (both real storage racks) and a 0.09m-tall sliver (a shelf-top band)
    # were all called "workbench" with fused scores 0.5-0.99, while their
    # raw CLIP scores were barely distinguishable from "table"/"shelf"/
    # "storage rack" (~0.25-0.30 for all candidates - essentially noise).
    "workbench":        {"h": (0.5, 1.3), "top": (0.5, 1.35), "long": (0.4, 4.0)},
    "bench":            {"h": (0.3, 1.0), "top": (0.35, 1.05), "long": (0.5, 3.0)},
    "fire extinguisher": {"h": (0.3, 0.75), "long": (0.06, 0.35)},
}
PRIOR_MIN = 0.15

# A bare wall / partition and a rack/shelf/cabinet standing against it are
# BOTH tall thin vertical planes to CLIP, so CLIP frequently picks the
# structural "wall"/"partition panel" (a delete-label) by a hair over the
# real object — silently erasing the object. The one geometric feature that
# separates them: a wall/partition is a thin sheet, a rack/shelf/cabinet has
# real DEPTH. Nodes at/above this short-dimension depth are never deleted as
# wall/partition; they fall through to the object CLIP ranked just beneath.
# (Symmetric to the floor/ceiling elevation vetoes below — a structure label
# only sticks when the geometry actually agrees with it.)
WALL_MIN_DEPTH_M = 0.30


def _band_score(v: float, lo: float | None, hi: float | None) -> float:
    if lo is not None and v < lo:
        return max(0.0, 1.0 - (lo - v) / max(0.12, 0.3 * lo))
    if hi is not None and v > hi:
        return max(0.0, 1.0 - (v - hi) / max(0.12, 0.3 * hi))
    return 1.0


def shape_prior(label: str, node: dict,
                floor_y: float | None, ceil_y: float | None,
                y_up_sign: float = 1.0) -> float:
    """floor_y/ceil_y are always plain P1/P99 percentiles of the raw Y
    column (so floor_y < ceil_y numerically, always) — but for a pipeline4
    y_invert space the raw Y axis itself increases downward, so floor_y
    (P1, numerically low) actually sits near the true CEILING and ceil_y
    (P99) near the true FLOOR. y_up_sign (+1 normal, -1 y_invert) corrects
    which raw-Y endpoint of the node is physically "up" and which
    percentile is the true floor reference, without ever touching the
    coordinates themselves (see pipeline4/p4_detect.py's world_to_model()
    docstring for why: camera-projection consistency for view selection
    below). Reduces to the exact original formulas when y_up_sign=1.0, so
    every space that predates this parameter is completely unaffected."""
    spec = SHAPE_PRIORS.get(label)
    if not spec:
        return 1.0
    sx, sy, sz = node["bbox_size"]
    feats = {"h": sy, "long": max(sx, sz), "short": min(sx, sz)}
    if floor_y is not None:
        # The physically-higher raw-Y endpoint is bbox_max for a normal
        # space, but bbox_min for a y_invert one (raw Y decreases upward).
        top_y, bot_y = ((node["bbox_max"][1], node["bbox_min"][1])
                       if y_up_sign > 0 else
                       (node["bbox_min"][1], node["bbox_max"][1]))
        true_floor_ref = floor_y if y_up_sign > 0 else ceil_y
        feats["top"] = (top_y - true_floor_ref) * y_up_sign
        feats["bot"] = (bot_y - true_floor_ref) * y_up_sign
        if ceil_y is not None:
            feats["ceil_gap"] = (ceil_y - floor_y) - feats["top"]
    s = 1.0
    for k, (lo, hi) in spec.items():
        if k in feats:
            s *= _band_score(feats[k], lo, hi)
    if s <= 0.0:
        # A band scored exactly 0 — not "a bit outside the band" but a CLEAR
        # shape violation (e.g. a 1.3m-tall box called "pallet", whose height
        # band caps at 0.65m; _band_score only clamps to 0 once a violation
        # is well past its tolerance). PRIOR_MIN exists so a genuinely
        # borderline object isn't hard-vetoed by CLIP being a little outside
        # a band, but it also let CLIP's softmax (scale 100) amplification
        # push a shape-IMPOSSIBLE label to the top anyway — verified
        # concretely on pipeline4 output (see pipeline4/README.md). A hard
        # 0 here removes the label from candidacy entirely instead of just
        # discounting it 85%.
        return 0.0
    return PRIOR_MIN + (1.0 - PRIOR_MIN) * s


# Recursive container refinement (see main(), --refine) — which labels are
# worth trying to subdivide, and the acceptance thresholds for a split.
CONTAINER_LABELS = {"desk", "table", "shelf", "storage_rack", "cabinet"}
REFINE_MAX_DEPTH = 2       # recursion depth per container
REFINE_MIN_PTS   = 60      # each sub-piece must keep at least this many pts
REFINE_COLOR_SEP = 0.12    # min mean-RGB distance between the two halves

MAX_VIEWS_PER_NODE = 7     # crops averaged per node — more viewpoint
                           # diversity reduces mislabels from a single
                           # occluded/ambiguous crop
MIN_DIST_M         = 0.5   # camera must be at least this far from the node
MAX_DIST_M         = 7.0   # ... and at most this far
MIN_CROP_PX        = 40    # projected box must be at least this big
CENTER_MARGIN_FRAC = 0.10  # node center must project this far inside the frame


def _bbox_corners(bmin, bmax) -> np.ndarray:
    xs, ys, zs = zip(bmin, bmax)
    return np.array([[x, y, z] for x in xs for y in ys for z in zs])


def select_views(node: dict, R: np.ndarray, pos: np.ndarray,
                  K: dict, panos: list[str],
                  pts: np.ndarray | None = None) -> list[tuple[int, tuple]]:
    """Return up to MAX_VIEWS_PER_NODE (camera_index, crop_bbox) picks,
    scored by projected size / distance, at most one per scanpoint.

    When the node's own points are available (`pts`), the crop is the
    robust 2D bbox (p2/p98) of the PROJECTED POINTS rather than the
    projected 3D box corners — a 3D box's corner hull covers up to ~2x the
    object's screen area from an oblique view, so corner crops feed CLIP
    mostly background; point crops hug the object itself."""
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
    W, H = K["width"], K["height"]
    c = np.array(node["centroid"])

    # Center visibility for all cameras at once.
    pc = np.einsum("nji,nj->ni", R, c - pos)    # R^T (c - pos), (N, 3)
    z = pc[:, 2]
    dist = np.linalg.norm(pc, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * pc[:, 0] / z + cx
        v = fy * pc[:, 1] / z + cy
    mx, my = W * CENTER_MARGIN_FRAC, H * CENTER_MARGIN_FRAC
    vis = ((z > 0.3) & (dist > MIN_DIST_M) & (dist < MAX_DIST_M)
           & (u > mx) & (u < W - mx) & (v > my) & (v < H - my))
    cand = np.where(vis)[0]
    if len(cand) == 0:
        return []

    sub = None
    if pts is not None and len(pts) >= 30:
        sub = pts if len(pts) <= 400 else \
            pts[np.random.default_rng(0).choice(len(pts), 400, replace=False)]

    corners = _bbox_corners(node["bbox_min"], node["bbox_max"])
    scored = []
    for ci in cand:
        if sub is not None:
            cc = (sub - pos[ci]) @ R[ci]        # (n, 3) camera frame
            zz = cc[:, 2]
            front = zz > 0.1
            if front.mean() < 0.5:
                continue
            uu = fx * cc[front, 0] / zz[front] + cx
            vv = fy * cc[front, 1] / zz[front] + cy
            inf = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            if inf.sum() < max(25, 0.4 * len(sub)):
                continue
            x1, x2 = np.percentile(uu[inf], [2.0, 98.0])
            y1, y2 = np.percentile(vv[inf], [2.0, 98.0])
        else:
            cc = (corners - pos[ci]) @ R[ci]    # (8, 3) camera frame
            zz = np.maximum(cc[:, 2], 0.05)
            uu = fx * cc[:, 0] / zz + cx
            vv = fy * cc[:, 1] / zz + cy
            x1, x2 = float(np.clip(uu.min(), 0, W)), float(np.clip(uu.max(), 0, W))
            y1, y2 = float(np.clip(vv.min(), 0, H)), float(np.clip(vv.max(), 0, H))
        if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
            continue
        area_frac = (x2 - x1) * (y2 - y1) / (W * H)
        # prefer a box that fills a good part of the frame, seen up close
        score = min(area_frac, 0.8) / float(dist[ci])
        scored.append((score, int(ci), (float(x1), float(y1), float(x2), float(y2))))
    scored.sort(reverse=True)

    picks, seen_panos = [], set()
    for score, ci, bbox in scored:
        if panos[ci] in seen_panos:
            continue
        seen_panos.add(panos[ci])
        picks.append((ci, bbox))
        if len(picks) >= MAX_VIEWS_PER_NODE:
            break
    return picks


def _children_are_one_object(children: list[dict],
                              min_overlap: float = 0.30) -> bool:
    """True when the (same-labeled) children's boxes substantially
    INTERLEAVE — the signature of a color split cutting through one object
    (its two color modes occupy the same space). Children that merely touch
    side-by-side are two real objects standing next to each other (e.g. two
    desks in a row) and must be kept separate."""
    for i in range(len(children)):
        for j in range(i + 1, len(children)):
            a, b = children[i], children[j]
            inter = 1.0
            vol_a = vol_b = 1.0
            for ax in range(3):
                ov = min(a["bbox_max"][ax], b["bbox_max"][ax]) - \
                     max(a["bbox_min"][ax], b["bbox_min"][ax])
                if ov <= 0:
                    return False
                inter *= ov
                vol_a *= max(a["bbox_max"][ax] - a["bbox_min"][ax], 1e-6)
                vol_b *= max(b["bbox_max"][ax] - b["bbox_min"][ax], 1e-6)
            if inter / min(vol_a, vol_b) < min_overlap:
                return False
    return True


def dedup_same_label_nested(nodes: list[dict],
                             containment: float = 0.65) -> list[dict]:
    """Drop duplicate boxes drawn around the same physical object: when two
    nodes share a label and one's box mostly overlaps the other's, keep the
    DENSER one — an inflated duplicate box is sparse, the tight one isn't."""
    def vol(n):
        s = n["bbox_size"]
        return max(s[0] * s[1] * s[2], 1e-9)

    drop: set[int] = set()
    for i in range(len(nodes)):
        if i in drop:
            continue
        for j in range(i + 1, len(nodes)):
            if j in drop:
                continue
            a, b = nodes[i], nodes[j]
            if a.get("label") != b.get("label"):
                continue
            inter = 1.0
            for ax in range(3):
                ov = min(a["bbox_max"][ax], b["bbox_max"][ax]) - \
                     max(a["bbox_min"][ax], b["bbox_min"][ax])
                if ov <= 0:
                    inter = 0.0
                    break
                inter *= ov
            if inter / min(vol(a), vol(b)) < containment:
                continue
            dens_a = a["n_points"] / vol(a)
            dens_b = b["n_points"] / vol(b)
            drop.add(i if dens_a < dens_b else j)
    kept = [n for k, n in enumerate(nodes) if k not in drop]
    print(f"[geo-clip] same-label nested dedup: {len(drop)} duplicate boxes "
          f"dropped (kept the denser box of each pair)")
    return kept


# ── Desk/table label unification ─────────────────────────────────────────
# "table" and "desk" are the same thing in these scans (user directive), so
# they share one label. The actual box GEOMETRY for tables is rebuilt in
# geo_to_scenegraph.consolidate_table_footprints (it needs the building yaw,
# known only there) — one box per CONTINUOUS tabletop region, gaps break it.
TABLE_LABELS = {"table", "desk"}
TABLE_UNIFIED = "table"


def unify_table_labels(nodes: list[dict]) -> list[dict]:
    n = 0
    for nd in nodes:
        if nd.get("label") in TABLE_LABELS and nd["label"] != TABLE_UNIFIED:
            nd["label"] = TABLE_UNIFIED
            n += 1
    if n:
        print(f"[geo-clip] unified {n} desk labels -> '{TABLE_UNIFIED}'")
    return nodes



def apply_annotations(nodes: list[dict], ann_path: Path) -> list[dict]:
    """User ground truth from the viewer's Annotate mode: forced labels and
    deletions override the pipeline (labels only — the pipeline computes
    boxes itself; annotation geometry is used offline for calibration)."""
    if not ann_path.exists():
        return nodes
    try:
        anns = json.loads(ann_path.read_text()).get("annotations", [])
    except Exception as e:
        print(f"[geo-clip] annotations unreadable ({e}) — ignored")
        return nodes
    label_over: dict[int, str] = {}
    deletes: set[int] = set()
    for a in anns:              # later ops win
        nid = a.get("node_id")
        if nid is None:
            continue
        if a.get("op") == "delete":
            deletes.add(int(nid))
            label_over.pop(int(nid), None)
        elif a.get("op") == "edit" and a.get("label"):
            label_over[int(nid)] = a["label"]
            deletes.discard(int(nid))
    out = []
    n_lbl = 0
    for n in nodes:
        if n["id"] in deletes:
            continue
        if n["id"] in label_over:
            lb = label_over[n["id"]]
            if n.get("label") != lb:
                n_lbl += 1
            n["label"] = lb
            n["label_source"] = "annotation"
            n["is_structure"] = lb in STRUCTURE_LABELS
        out.append(n)
    print(f"[geo-clip] annotations applied: {n_lbl} labels forced, "
          f"{len(nodes) - len(out)} nodes deleted "
          f"({ann_path.name})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, choices=space_choices())
    ap.add_argument("--geo-json", default=None,
                     help="Default: pipeline2/out/<space>_geo.json")
    ap.add_argument("--refine", action="store_true",
                     help="Enable the EXPERIMENTAL recursive container "
                          "refinement (\"Version B\" in the A/B review). "
                          "Default is flat one-label-per-node labeling — "
                          "the production choice after the A/B comparison.")
    ap.add_argument("--no-structure-drop", action="store_true",
                     help="Never flag a node as structure for deletion (geo5): "
                          "foreground/background separation is done up front by "
                          "bg_removal.py, so structure must never be removed "
                          "semantically at the CLIP stage.")
    ap.add_argument("--no-annotations", action="store_true",
                     help="Ignore the viewer annotations.json (pipeline2b node "
                          "ids differ from those annotations, so applying them "
                          "by id would mislabel; annotations are verification-"
                          "only here).")
    ap.add_argument("--skip-clip", action="store_true",
                     help="Reuse the labels already in the geo JSON and run "
                          "only the post passes (annotation overrides, "
                          "desk/table consolidation, dedupe) — seconds "
                          "instead of a full CLIP pass.")
    args = ap.parse_args()

    geo_path = Path(args.geo_json) if args.geo_json else \
        REPO / "pipeline2" / "out" / f"{args.space}_geo.json"
    geo = json.loads(geo_path.read_text())
    nodes = geo["nodes"]
    floor_y = geo.get("floor_y")   # for structure-label elevation
    ceil_y = geo.get("ceil_y")     # corroboration in classify_node
    y_up_sign = float(geo.get("y_up_sign", 1.0))   # -1 for a y_invert space
    print(f"[geo-clip] {len(nodes)} nodes from {geo_path}")
    ann_path = (Path("/nonexistent/annotations.json") if args.no_annotations
                else REPO / "ui" / "_spaces" / f"{args.space}_geo" / "annotations.json")

    if args.skip_clip:
        print("[geo-clip] --skip-clip: reusing existing labels")
        pts_path = geo_path.with_name(geo_path.stem + "_points.npz")
        node_pts = np.load(pts_path) if pts_path.exists() else None
        nodes = apply_annotations(nodes, ann_path)
        nodes = unify_table_labels(nodes)
        nodes = dedup_same_label_nested(nodes)
        geo["nodes"] = nodes
        geo_path.write_text(json.dumps(geo, indent=2))
        from collections import Counter
        counts = Counter(n["label"] for n in nodes if not n.get("is_structure"))
        print("[geo-clip] label counts: " +
              ", ".join(f"{l}={c}" for l, c in counts.most_common()))
        print(f"[geo-clip] → {geo_path}")
        return

    sp = space(args.space)
    cams = json.loads(sp["cameras"].read_text())
    K = json.loads(sp["intrinsics"].read_text())
    R = np.array([c["R"] for c in cams], dtype=np.float64)
    pos = np.array([c["pos"] for c in cams], dtype=np.float64)
    names = [Path(c["pano"]).name for c in cams]
    panos = [n.split("_")[0] for n in names]   # scanpoint id prefix
    views_dir = sp["views"]

    model, preprocess, tokenizer, device = clip_utils.load_clip_model()
    text_emb = clip_utils.build_label_text_embeddings(
        model, tokenizer, device, VOCAB, PROMPTS)
    mat_emb = clip_utils.build_label_text_embeddings(
        model, tokenizer, device, MATERIALS, MATERIAL_PROMPTS)
    state_banks: dict[str, tuple[list[str], np.ndarray]] = {}
    for cls, probes in STATE_PROBES.items():
        state_names = list(probes.keys())
        state_banks[cls] = (state_names, clip_utils.build_label_text_embeddings(
            model, tokenizer, device, state_names,
            {n: probes[n] for n in state_names}))
    print(f"[geo-clip] CLIP ready on {device}, {len(VOCAB)} labels, "
          f"{len(MATERIALS)} materials, {len(STATE_PROBES)} state classes")

    img_cache: dict[int, Image.Image] = {}
    stats = {"labeled": 0, "structure": 0, "noview": 0}

    def classify_node(node: dict, pts: np.ndarray | None = None) -> bool:
        """Crop the node's best views, CLIP-encode, zero-shot label fused
        with the geometric shape prior. Sets clip_topk / label /
        is_structure on the node. False if no view."""
        picks = select_views(node, R, pos, K, panos, pts=pts)
        crops = []
        for ci, bbox in picks:
            if ci not in img_cache:
                p = views_dir / names[ci]
                if not p.exists():
                    continue
                img_cache[ci] = Image.open(p).convert("RGB")
                if len(img_cache) > 400:
                    img_cache.pop(next(iter(img_cache)))
            # Multi-scale crops per view (OpenMask3D §3.3 does the same):
            # a tight crop isolates the object, a padded one keeps enough
            # context for CLIP to read scale/support — averaging both is
            # more robust than betting on either.
            for pad in (0.05, 0.30):
                crop = clip_utils.crop_bbox_padded(img_cache[ci], bbox,
                                                   pad_frac=pad)
                if crop is not None:
                    crops.append(crop)
        if not crops:
            node["clip_topk"] = []
            node["is_structure"] = False
            stats["noview"] += 1
            return False
        feats = clip_utils.encode_images(model, preprocess, device, crops)
        emb = clip_utils.average_normalize(feats)
        sims = emb @ text_emb.T
        # Fuse CLIP with the geometric shape prior: softmax at CLIP's
        # standard logit scale (100), multiplied by each label's shape
        # plausibility for this node's actual 3D box. Near-tie CLIP scores
        # (deltas of ~0.001 cosine) are decided by geometry; a decisive
        # CLIP margin still overrides a soft prior violation.
        logits = 100.0 * (sims - sims.max())
        probs = np.exp(logits)
        probs /= probs.sum()
        prior = np.array([shape_prior(l, node, floor_y, ceil_y, y_up_sign)
                          for l in VOCAB])
        fused = probs * prior
        fused /= max(fused.sum(), 1e-12)
        order = np.argsort(-fused)
        node["clip_topk"] = [
            {"label": VOCAB[i], "score": round(float(sims[i]), 4),
             "fused": round(float(fused[i]), 4)}
            for i in order[:3]
        ]
        # Geometric corroboration: an elevation-bound structure label must
        # match the node's actual elevation — a "ceiling" at desk height or
        # a "floor" a metre up is a CLIP mislabel, and since structure
        # labels DELETE the node, an uncorroborated mislabel would silently
        # erase a real object. Walk down the ranking until a label
        # consistent with the geometry.
        top = VOCAB[order[0]]
        if floor_y is not None and ceil_y is not None:
            # Same y_up_sign correction as shape_prior(): pick whichever
            # raw-Y endpoint is physically top/bottom and the matching
            # floor reference, then measure elevation-from-floor and
            # gap-from-ceiling generically (reduces to the plain
            # bbox_max/bbox_min - floor_y/ceil_y formulas below when
            # y_up_sign=1.0).
            if y_up_sign > 0:
                phys_top_y, phys_bot_y = node["bbox_max"][1], node["bbox_min"][1]
                true_floor_ref = floor_y
            else:
                phys_top_y, phys_bot_y = node["bbox_min"][1], node["bbox_max"][1]
                true_floor_ref = ceil_y
            elev_bot = (phys_bot_y - true_floor_ref) * y_up_sign
            elev_top = (phys_top_y - true_floor_ref) * y_up_sign
            ceil_gap = (ceil_y - floor_y) - elev_top
            for i in order:
                cand = VOCAB[i]
                if cand == "floor" and (elev_bot > 0.10
                                        or node["bbox_size"][1] > 0.20):
                    # a floor residual is a THIN SHEET AT floor level — a
                    # low object standing on the floor (pallet stack, box,
                    # platform) starts a little higher and/or has real
                    # height; CLIP calls it "floor" because the crop is
                    # dominated by floor texture around it.
                    continue
                if cand == "ceiling" and (ceil_gap > 0.60
                                          or node["bbox_size"][1] > 1.0):
                    # a ceiling residual is a thin sheet AT the ceiling — a
                    # tall box that merely REACHES up there (rack + its top
                    # load) is an object photographed against the ceiling,
                    # and deleting it as "ceiling" erases the rack.
                    continue
                if cand == "ceiling light" and ceil_gap > 1.20:
                    continue
                if (cand in ("wall", "partition panel")
                        and min(node["bbox_size"][0], node["bbox_size"][2])
                        >= WALL_MIN_DEPTH_M):
                    # object-depth node — not a thin wall/partition sheet.
                    # Skip the structure label; keep the object underneath.
                    continue
                top = VOCAB[i]
                break
        node["label"] = top.replace(" ", "_")
        node["is_structure"] = (top in STRUCTURE_LABELS) and not args.no_structure_drop

        # 3DSSG-style attributes from the same crop embedding: material for
        # every object (except people), state only where visually decidable.
        if top != "person":
            msims = emb @ mat_emb.T
            mi = int(np.argmax(msims))
            node["material"] = MATERIALS[mi]
            node["material_score"] = round(float(msims[mi]), 4)
        if top in state_banks:
            state_names, semb = state_banks[top]
            ssims = emb @ semb.T
            node["state"] = state_names[int(np.argmax(ssims))]

        stats["labeled"] += 1
        if node["is_structure"]:
            stats["structure"] += 1
        return True

    # Node points sidecar — used for point-projected crops here and by the
    # recursive container refinement below.
    pts_path = geo_path.with_name(geo_path.stem + "_points.npz")
    node_pts = np.load(pts_path) if pts_path.exists() else None

    def _pts_for(node_id: int) -> np.ndarray | None:
        if node_pts is None:
            return None
        key = f"xyz_{node_id}"
        return node_pts[key].astype(np.float64) if key in node_pts else None

    for node in nodes:
        classify_node(node, pts=_pts_for(node["id"]))

    # ── Recursive container refinement (--refine only) ───────────────────
    # CLIP can't detect multiple objects inside one crop, but it CAN verify
    # a geometric subdivision. For every node labeled as a container class,
    # attempt a color-based split of its own points and CLIP-label the
    # sub-pieces:
    #   - if any sub-piece carries the SAME label as the parent (another
    #     "desk" inside the desk box), the parent box was really several
    #     objects — REMOVE the parent, keep the children, recurse into the
    #     same-labeled ones;
    #   - if no sub-piece matches the parent (a storage rack holding only
    #     boxes), the parent is a genuine container — KEEP the parent box
    #     AND the content boxes (children get container_id = parent id).
    if not args.refine:
        print("[geo-clip] flat labeling (default; --refine enables the "
              "experimental recursive container refinement)")
        refined = nodes
    elif node_pts is None:
        print(f"[geo-clip] {pts_path} missing — skipping recursive refinement "
              f"(rerun geo_cluster.py to produce it)")
        refined = nodes
    else:
        next_id = max(n["id"] for n in nodes) + 1
        ref_stats = {"replaced": 0, "containers": 0, "children": 0}

        def make_node(p: np.ndarray, r: np.ndarray | None) -> dict:
            nonlocal next_id
            # percentile-trimmed bounds, same as geo_cluster's final boxes —
            # stray tail points must not inflate a child's box either
            bmin = np.percentile(p, 1.5, axis=0)
            bmax = np.percentile(p, 98.5, axis=0)
            mean_rgb = (r.mean(0) * 255).round().astype(int).tolist() \
                if r is not None else None
            node = {"id": next_id, "label": f"obj_{next_id}",
                    "centroid": p.mean(0).tolist(),
                    "bbox_min": bmin.tolist(), "bbox_max": bmax.tolist(),
                    "bbox_size": (bmax - bmin).tolist(),
                    "n_points": int(len(p)), "mean_rgb": mean_rgb}
            next_id += 1
            return node

        def try_split(p: np.ndarray, r: np.ndarray | None):
            """One color-guided 2-means split; None if not clearly two
            differently-colored regions. No box-overlap cap here — unlike
            geo_cluster's blind color split, CLIP verifies the result."""
            if r is None or len(p) < 2 * REFINE_MIN_PTS:
                return None
            extent = np.maximum(p.max(0) - p.min(0), 1e-6)
            feat = np.concatenate([(p - p.min(0)) / extent, r * 1.5], axis=1)
            try:
                from sklearn.cluster import KMeans
                lbl = KMeans(n_clusters=2, n_init=3, random_state=0).fit_predict(feat)
            except Exception:
                return None
            m = lbl == 0
            a, b = p[m], p[~m]
            if len(a) < REFINE_MIN_PTS or len(b) < REFINE_MIN_PTS:
                return None
            if float(np.linalg.norm(r[m].mean(0) - r[~m].mean(0))) < REFINE_COLOR_SEP:
                return None
            for half in (a, b):   # don't create slab pieces (dead weight)
                sz = np.sort(np.maximum(half.max(0) - half.min(0), 1e-6))
                if sz[0] < 0.10 and sz[1] * sz[2] > 0.25:
                    return None
            return (a, r[m]), (b, r[~m])

        def refine(node: dict, p: np.ndarray | None, r: np.ndarray | None,
                   depth: int) -> list[dict]:
            if (node.get("is_structure") or p is None
                    or depth >= REFINE_MAX_DEPTH
                    or node.get("label") not in CONTAINER_LABELS):
                return [node]
            halves = try_split(p, r)
            if halves is None:
                return [node]
            children = []
            for cp, cr in halves:
                child = make_node(cp, cr)
                if not classify_node(child, pts=cp) or child["is_structure"]:
                    continue
                children.append((child, cp, cr))
            if len(children) < 2:
                return [node]
            same_label = [t for t in children if t[0]["label"] == node["label"]]
            # One-object guard: when ALL children carry the parent's own
            # label AND their boxes substantially interleave, the color
            # split just cut ONE desk/rack in half — the halves are parts,
            # not separate objects. Keep the single parent box instead of
            # several overlapping same-label boxes with edges between them.
            if len(same_label) == len(children) and _children_are_one_object(
                    [c for c, _, _ in children]):
                return [node]
            if same_label:
                ref_stats["replaced"] += 1
                out = []
                for c, cp, cr in children:
                    if c["label"] == node["label"]:
                        out.extend(refine(c, cp, cr, depth + 1))
                    else:
                        out.append(c)
                ref_stats["children"] += len(out)
                return out
            ref_stats["containers"] += 1
            ref_stats["children"] += len(children)
            for c, _, _ in children:
                c["container_id"] = node["id"]
            return [node] + [c for c, _, _ in children]

        refined = []
        for node in nodes:
            key = f"xyz_{node['id']}"
            p = node_pts[key].astype(np.float64) if key in node_pts else None
            rkey = f"rgb_{node['id']}"
            r = (node_pts[rkey].astype(np.float32) / 255.0) \
                if p is not None and rkey in node_pts else None
            refined.extend(refine(node, p, r, 0))
        print(f"[geo-clip] recursive refinement: {ref_stats['replaced']} "
              f"multi-object boxes replaced by their parts, "
              f"{ref_stats['containers']} genuine containers kept WITH their "
              f"contents, {ref_stats['children']} sub-nodes added "
              f"-> {len(refined)} nodes")

    # Post-passes (both modes): user annotation overrides, tabletop-plane
    # box consolidation, then duplicate same-label boxes around one
    # physical object collapse to the densest box.
    refined = apply_annotations(refined, ann_path)
    refined = unify_table_labels(refined)
    refined = dedup_same_label_nested(refined)

    geo["nodes"] = refined
    geo_path.write_text(json.dumps(geo, indent=2))
    from collections import Counter
    counts = Counter(n["label"] for n in refined if n.get("clip_topk"))
    print(f"[geo-clip] labeled {stats['labeled']} classifications "
          f"({stats['noview']} had no usable view), "
          f"{sum(1 for n in refined if n.get('is_structure'))} structure-flagged "
          f"in final set")
    print("[geo-clip] label counts: " +
          ", ".join(f"{l}={c}" for l, c in counts.most_common()))
    print(f"[geo-clip] → {geo_path}")


if __name__ == "__main__":
    main()
