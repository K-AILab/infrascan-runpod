#!/usr/bin/env python3
"""
scene_graph.py  —  Scene Graph builder (v5)

Builds a rich DSG-style scene graph inspired by:
  • Rosinol et al. 2020 — Dynamic Scene Graphs (layers, places, rooms)
  • Wald et al. 2020 (3DSSG) — semantic relationships
  • Gu et al. 2023 (ConceptGraphs) — open-vocab labelling via LLM/LVLM;
    multi-view feature fusion; geometric + semantic duplicate removal
  • Heo et al. 2025 (DCRL-3DSSG) — object-centric representation learning
    for enhanced 3D semantic scene graph prediction

Key changes from v4
────────────────────
Object classification (Heo et al. §3.1 "Object Feature Learning")
  • 11-dim geometric descriptor per object (bbox, sigma, volume, max_side)
    computed in build_objects; used for geometry-adjusted CLIP scoring
  • Geometry prior multiplied into CLIP cosine scores before argmax:
    height band × aspect ratio × size heuristics → lower label entropy
  • label_entropy stored per node (Heo et al. §2 observation: high entropy
    correlates with predicate errors — so uncertain labels produce fewer
    edges downstream)

Edges — Heo et al. §3.2 "Relationship Feature Encoder"
  • Geometric descriptor g_ij = CAT(μ_i−μ_j, σ_i−σ_j, b_i−b_j,
    log(v_i/v_j), log(l_i/l_j)) ∈ R^11 (paper Eq. 5) drives edge type
    classification; all edges are directed (bidirectional where applicable)
  • "similar" edges removed entirely
  • Support & Attachment: standing_on, lying_on, hanging_on, connected_to,
    support (structural element supports object)
  • Spatial & Directional: above, below, left, right, front, behind,
    inside, surrounding, close_by

Room detection
  • Wall-segment occupancy grid + flood fill (unchanged from v4)

Usage:
    python pipeline2/scene_graph.py --space factory_space

    # tune room detection for open-plan spaces:
    python pipeline2/scene_graph.py --space factory_space --room-eps 2.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))   # _paths.py lives there
from _paths import space, space_choices  # noqa: E402
import clip_utils  # noqa: E402

# ── Tunable defaults ──────────────────────────────────────────────────────
ABOVE_DELTA_M     = 0.25   # min vertical gap for standing_on
HANGING_DELTA_M   = 1.00   # min vertical gap for hanging_on (lamp just above shelf)
MAX_DIRECT_GAP_M  = 0.60   # max vertical gap for standing_on/supported_by
                            # prevents floor↔ceiling phantom edges
MAX_HANG_GAP_M    = 1.80   # max gap for hanging_on (lamp above desk, not lamp above floor)
FOOTPRINT_IOU_THR = 0.15   # footprint overlap threshold
ON_FLOOR_M        = 0.30
REST_INTERPEN_M   = 0.15   # standing_on/lying_on rest test: how far the
                            # higher box's BOTTOM face may dip below the
                            # lower box's TOP face (boxes interpenetrate a
                            # little when an object overhangs a desk edge)
REST_GAP_MAX_M    = 0.25   # ... and how far above it it may float. Tighter
                            # than MAX_DIRECT_GAP_M: with real box faces
                            # (rather than centroid heights) an object 0.5m
                            # above a desk is clearly NOT standing on it.

MIN_WORLD_PTS     = 10     # objects need ≥10 valid 3D backprojections
MIN_PROPOSALS     = 5      # objects must appear in ≥5 panoramic views
MAX_NODES         = 20000  # safety ceiling only — not a target. Raise further
                            # with --max-nodes if your space has more objects.
DEDUP_M           = 1.00   # raised from 0.25: collapse duplicates after Y-snap
DEDUP_COS         = 0.65   # lowered from 0.70: merge same-object cross-view views

# ── Point-cloud centroid snapping ─────────────────────────────────────────
# World_pos Y from panoramic depth estimation is unreliable (vertical
# component is most sensitive to depth errors in equirectangular projections).
# We load the scanned PLY point cloud, build a 2D KD-tree on (X,Z), and:
#   1. Clip each proposal's world_pos Y to the building's actual Y range.
#   2. Snap each object centroid to the nearest point-cloud surface cluster,
#      choosing the cluster closest to the (clipped) centroid Y estimate.
#   3. Discard objects with fewer than PLY_SNAP_MIN_PTS nearby points
#      in a PLY_SNAP_XZ_M horizontal radius (hallucinated objects in empty
#      space have no nearby real surface).
PLY_SNAP_XZ_M     = 1.50   # XZ search radius for point-cloud snap (m)
PLY_SNAP_MIN_PTS  = 8      # min nearby points to accept an object
PLY_SNAP_MAX_CORRECTION_M = 0.6  # max |snap - original Y| to actually apply
PLY_Y_SLACK_M     = 0.60   # slack added to point-cloud Y bounds before clipping

SIM_COS_THR       = 0.85   # kept for CLI back-compat (not used — similar edges removed)
SIM_K             = 8      # kept for CLI back-compat (not used)

# ── Spatial / directional edge thresholds (Heo et al. §3.2) ──────────────
CLOSE_BY_M        = 1.20   # max 3D distance for spatial/directional edges —
                            # edges should only connect objects that are
                            # really close and plausibly related; combined
                            # with the same-working-area rule below.
NO_EDGE_LABELS    = {"ceiling_light", "ceiling light"}
                            # Labels that never participate in edges: a
                            # ceiling light doesn't physically interact with
                            # the furniture below it — the relation would
                            # carry no useful information.
DIR_DOMINANCE     = 0.60   # horizontal axis must be ≥ this fraction of horiz dist
INSIDE_SCALE      = 0.55   # fraction of bbox half-extent for the
                            # standing_in/lying_in containment check
MAX_EDGES_PER_NODE = 15    # per-node edge cap applied after all edges are built;
                            # prevents graph explosion in object-dense spaces
MAX_EDGE_Y_GAP_M  = 1.80  # skip ALL edge types when objects are more than this
                            # far apart vertically — prevents ceiling lights from
                            # generating edges to floor objects

# Objects in different rooms are never linked by an edge (a graph edge
# implies spatial relation, and rooms are separated by walls); edges are
# likewise restricted to objects sharing the same working area, since a
# room→area→object hierarchy makes "same area" the natural unit of
# "related" — see the room/area checks in build_edges.

# Labels whose presence as the lower object triggers "support" relation
STRUCT_LABELS = frozenset([
    "pillar", "column", "support_beam", "railing", "wall",
])

# Labels that may legitimately "hang" from an overhead structure.
# Only these as the higher object produce a hanging_on edge.
_HANGING_LABELS = frozenset([
    "ceiling_light", "pendant_lamp", "projector", "smoke_detector",
    "sprinkler", "duct", "exit_sign", "wall_light",
])

ROOM_EPS_M        = 1.50   # raised for large factory floors
ROOM_MIN_PTS      = 15
ROOM_SLICE_LO     = 0.05
ROOM_SLICE_HI     = 1.00   # wider band to capture wall bases
ROOM_SUBSAMPLE    = 60_000

# ── Expanded label vocabulary (industrial + office) ───────────────────────
# Used by heuristic fallback AND as the candidate set for LLM captioning.
LABEL_VOCAB = [
    # Seating
    "chair", "armchair", "stool", "bench", "sofa",
    # Tables / work surfaces
    "desk", "table", "workbench", "counter",
    # Storage
    "cabinet", "shelf", "bookshelf", "rack", "locker", "drawer",
    # Lighting
    "ceiling_light", "pendant_lamp", "wall_light", "desk_lamp", "floor_lamp",
    # Screens / electronics
    "monitor", "laptop", "keyboard", "mouse", "projector", "tv_screen",
    "printer", "phone", "tablet",
    # Industrial / factory
    "machine", "conveyor", "robot_arm", "workstation", "control_panel",
    "electrical_panel", "pipe", "duct", "column", "support_beam",
    "pallet", "forklift", "cart", "trolley",
    # Safety
    "fire_extinguisher", "exit_sign", "safety_barrier", "first_aid",
    # Plants / decor
    "plant", "painting", "whiteboard", "blackboard",
    # Containers
    "trash_bin", "box", "crate", "barrel", "bag",
    # Architecture
    "door", "window", "wall", "pillar", "stairs", "railing",
    # Misc
    "clock", "bottle", "cup",
]

# ── Per-class descriptive prompt ensembles ───────────────────────────────
# Inspired by CuPL (Pratt et al. 2023, "What does a platypus look like?")
# and the CLIP paper (Radford et al. 2021) which shows that averaging
# multiple context-varied prompts into one embedding improves zero-shot
# accuracy over a single generic template.  We write the prompts manually
# (no GPT-3 needed) — 3-5 descriptions per class, emphasising the visual
# and spatial cues that distinguish frequently confused pairs.
LABEL_PROMPTS: dict[str, list[str]] = {
    # ── Most frequently confused pairs ──────────────────────────────────
    "whiteboard": [
        "a large white rectangular board mounted on a wall for writing",
        "a dry-erase whiteboard with markers and writing on it",
        "a magnetic whiteboard surface hanging on a wall",
        "a wide flat white surface used for presentations on a wall",
    ],
    "blackboard": [
        "a dark green or black chalkboard mounted on a wall",
        "a blackboard with white chalk writing on it",
        "a large dark writing board fixed to a wall",
    ],
    "monitor": [
        "a flat-panel computer monitor on a desk",
        "a small rectangular display screen on a workstation desk",
        "a PC monitor showing a desktop interface",
    ],
    "laptop": [
        "an open laptop computer on a desk",
        "a portable laptop with a screen and keyboard on a table",
        "a clamshell laptop computer",
    ],
    "keyboard": [
        "a computer keyboard with individual keys on a desk",
        "a flat rectangular keyboard for typing on a desk",
        "a QWERTY keyboard next to a monitor",
    ],
    "tv_screen": [
        "a large flat-screen television mounted on a wall",
        "a wide TV display screen",
        "a large monitor or screen on the wall",
    ],
    # ── Lighting ─────────────────────────────────────────────────────────
    "ceiling_light": [
        "a rectangular fluorescent ceiling light panel",
        "a flat light fixture mounted flush to the ceiling",
        "an overhead ceiling light",
    ],
    "pendant_lamp": [
        "a pendant lamp hanging down from the ceiling",
        "a suspended light fixture hanging from the ceiling",
        "an overhead hanging light",
    ],
    "floor_lamp": [
        "a tall floor lamp standing on the floor",
        "a standing lamp on the floor",
    ],
    "desk_lamp": [
        "a small desk lamp on a table or workbench",
        "a task lamp next to a monitor on a desk",
    ],
    # ── Industrial / factory ─────────────────────────────────────────────
    "machine": [
        "a large industrial machine in a factory",
        "a manufacturing machine or equipment in a workshop",
        "a heavy industrial machine with metal body",
    ],
    "conveyor": [
        "a conveyor belt used to move items in a factory",
        "an industrial conveyor system with a moving belt",
        "a flat belt conveyor in a warehouse",
    ],
    "workbench": [
        "a sturdy workbench in a workshop or factory",
        "a work table with tools on it in a factory",
        "a heavy-duty workshop bench",
    ],
    "control_panel": [
        "an industrial control panel with buttons and switches",
        "a machine control panel with dials and buttons",
        "an electrical control panel in a factory",
    ],
    "electrical_panel": [
        "a grey electrical panel box mounted on a wall",
        "a breaker box or distribution panel on a wall",
        "an electrical switchboard on a factory wall",
    ],
    "pallet": [
        "a wooden pallet on the floor",
        "a flat wooden pallet used in a warehouse",
        "a shipping pallet on the ground",
    ],
    # ── Storage ───────────────────────────────────────────────────────────
    "shelf": [
        "a shelf unit with items on multiple levels",
        "metal shelving in a warehouse or office",
        "a storage shelf with goods on it",
    ],
    "cabinet": [
        "a storage cabinet with doors",
        "a metal filing cabinet in an office",
        "a tall storage cabinet with shelves inside",
    ],
    "rack": [
        "a metal storage rack with shelves in a warehouse",
        "an industrial shelving rack",
        "a tall metal rack for storing materials",
    ],
    # ── Safety ────────────────────────────────────────────────────────────
    "fire_extinguisher": [
        "a red fire extinguisher mounted on a wall",
        "a cylindrical fire extinguisher",
    ],
    "safety_barrier": [
        "a yellow safety barrier or bollard in a factory",
        "an industrial safety barrier or guardrail",
        "a floor-mounted safety fence",
    ],
    # ── Architecture ──────────────────────────────────────────────────────
    "pillar": [
        "a structural pillar or column in a building",
        "a vertical load-bearing column in a factory",
        "a concrete or metal support pillar",
    ],
    "door": [
        "a door in a wall",
        "an open or closed door in a building",
    ],
    "window": [
        "a window in a wall or facade",
        "a glass window letting in natural light",
    ],
}


# ── DINOv2 ImageNet-1k class index → label (expanded for factory) ─────────
def _imagenet_to_label() -> dict[int, str]:
    return {
        # Chairs / seating
        423: "chair",   559: "chair",   765: "chair",   846: "chair",
        497: "bench",   832: "sofa",
        # Tables
        532: "table",   880: "table",
        # Monitors / screens
        664: "monitor", 782: "monitor", 849: "tv_screen",
        # Computers
        620: "laptop",  508: "keyboard", 673: "mouse",
        # Lighting — ceiling vs floor vs desk
        604: "desk_lamp", 719: "floor_lamp",
        # Storage
        731: "cabinet", 578: "bookshelf",
        # Plants
        984: "plant",   985: "plant",   992: "plant",
        # Trash
        413: "trash_bin",
        # Boxes / containers
        469: "box",     542: "crate",
        # Clocks
        409: "clock",   530: "clock",
        # Bags
        414: "bag",     791: "bag",
        # Bottles
        440: "bottle",  737: "bottle",  968: "cup",
        # Phones
        528: "phone",   722: "phone",
        # Printers
        742: "printer",
        # Projector
        791: "projector",
        # Pillars / columns (approximated by vault, arch)
        980: "pillar",
        # Industrial machinery (approximated by lathe, drill)
        808: "machine", 722: "control_panel",
        # Barrels
        414: "barrel",
        # Railing / fence
        756: "railing",
        # Stairs
        887: "stairs",
    }


# ── Height+geometry heuristics ────────────────────────────────────────────
def _heuristic_label(centroid: list, n_world_pts: int,
                     floor_y: float, ceiling_y: float) -> str:
    """Improved heuristic using height band, point density, and ceiling proximity."""
    y = float(centroid[1])
    h = y - floor_y
    ceil_dist = ceiling_y - y

    # Very close to ceiling → ceiling-mounted light
    if ceil_dist < 0.40 and h > 1.5:
        return "ceiling_light"
    # High up but not ceiling-mounted → pendant lamp or wall light
    if h > 1.8:
        return "pendant_lamp" if n_world_pts > 5 else "exit_sign"
    # Upper mid-height range
    if h > 1.3:
        if n_world_pts > 15: return "cabinet"
        if n_world_pts > 8:  return "monitor"
        return "wall_light"
    if h > 0.8:
        if n_world_pts > 20: return "workbench"
        if n_world_pts > 10: return "desk"
        return "chair"
    if h > 0.3:
        if n_world_pts > 15: return "table"
        if n_world_pts > 6:  return "stool"
        return "trash_bin"
    # Floor level
    if n_world_pts > 20: return "pallet"
    if n_world_pts > 8:  return "box"
    return "crate"


# ══ helpers ══════════════════════════════════════════════════════════════

def _segment_crosses_wall(c_a, c_b, wall_blocker: dict) -> bool:
    """True if the XZ segment between two object centroids passes THROUGH a
    confirmed wall. wall_blocker: {"cells": set[(gx,gz)], "x0", "z0",
    "cell_m"} — an occupancy grid of confirmed wall points (the geo
    pipeline exports one; other pipelines pass None and skip this).

    Only the interior 12-88% of the segment is sampled, so an object
    standing against a wall keeps edges to neighbors along the same wall;
    >= 2 samples must land on wall cells so one noisy cell can't kill a
    real edge."""
    dx = float(c_b[0] - c_a[0])
    dz = float(c_b[2] - c_a[2])
    cell_m = wall_blocker["cell_m"]
    n = max(int((dx * dx + dz * dz) ** 0.5 / (cell_m * 0.5)), 2)
    cells, x0, z0 = wall_blocker["cells"], wall_blocker["x0"], wall_blocker["z0"]
    hits = 0
    for t in np.linspace(0.12, 0.88, n):
        gx = int((c_a[0] + dx * t - x0) / cell_m)
        gz = int((c_a[2] + dz * t - z0) / cell_m)
        if (gx, gz) in cells:
            hits += 1
            if hits >= 2:
                return True
    return False


def _bbox_footprint_containment(obj_a: dict, obj_b: dict) -> float | None:
    """XZ footprint intersection area / SMALLER footprint area, from the
    objects' actual bounding boxes. Returns None when either object lacks a
    bbox (callers then fall back to the centroid-circle _xz_iou below).

    Containment-over-smaller (not IoU-over-union) is deliberate: a cup on
    the corner of a 2m desk covers ~100% of its OWN footprint but a tiny
    fraction of the union — a circle-IoU version of this test scores such
    pairs near 0 and the support relation never fires, so on-desk objects
    end up labeled 'left of the desk' instead of 'standing on' it."""
    a_min, a_max = obj_a.get("bbox_min"), obj_a.get("bbox_max")
    b_min, b_max = obj_b.get("bbox_min"), obj_b.get("bbox_max")
    if a_min is None or a_max is None or b_min is None or b_max is None:
        return None
    ix = min(a_max[0], b_max[0]) - max(a_min[0], b_min[0])
    iz = min(a_max[2], b_max[2]) - max(a_min[2], b_min[2])
    if ix <= 0 or iz <= 0:
        return 0.0
    area_a = max((a_max[0] - a_min[0]) * (a_max[2] - a_min[2]), 1e-9)
    area_b = max((b_max[0] - b_min[0]) * (b_max[2] - b_min[2]), 1e-9)
    return float(ix * iz / min(area_a, area_b))


def _xz_iou(c_a, c_b, r: float = 0.35) -> float:
    dx = float(c_a[0] - c_b[0])
    dz = float(c_a[2] - c_b[2])
    d  = (dx**2 + dz**2) ** 0.5
    if d >= 2 * r: return 0.0
    if d < 1e-6:   return 1.0
    alpha = 2 * np.arccos(np.clip(d / (2 * r), -1, 1))
    inter = r * r * (alpha - np.sin(alpha))
    union = 2 * np.pi * r * r - inter
    return float(inter / union)


def _geo_descriptor(obj_a: dict, obj_b: dict) -> np.ndarray:
    """
    11-dim geometric descriptor g_ij (Heo et al. paper Eq. 5):
        g = CAT(μ_i − μ_j,  σ_i − σ_j,  b_i − b_j,
                log(v_i/v_j),  log(l_i/l_j))
    where μ=centroid, σ=std-dev of pts, b=bbox size,
    v=volume, l=max side length.
    """
    mu_a  = np.array(obj_a["centroid"],              dtype=np.float32)
    mu_b  = np.array(obj_b["centroid"],              dtype=np.float32)
    sig_a = np.array(obj_a.get("sigma",    [0,0,0]), dtype=np.float32)
    sig_b = np.array(obj_b.get("sigma",    [0,0,0]), dtype=np.float32)
    b_a   = np.array(obj_a.get("bbox_size",[1,1,1]), dtype=np.float32)
    b_b   = np.array(obj_b.get("bbox_size",[1,1,1]), dtype=np.float32)
    v_a   = max(float(obj_a.get("volume",   1.0)),   1e-6)
    v_b   = max(float(obj_b.get("volume",   1.0)),   1e-6)
    l_a   = max(float(obj_a.get("max_side", 1.0)),   1e-6)
    l_b   = max(float(obj_b.get("max_side", 1.0)),   1e-6)
    return np.concatenate([
        mu_a - mu_b,              # 3 dims: centroid offset
        sig_a - sig_b,            # 3 dims: spread offset
        b_a - b_b,                # 3 dims: bbox size offset
        [np.log(v_a / v_b)],      # 1 dim : volume ratio
        [np.log(l_a / l_b)],      # 1 dim : max-side ratio
    ]).astype(np.float32)         # total: 11 dims


def _geometry_prior(label: str, h: float, bbox_size: list,
                    ceil_dist: float, n_world_pts: int = 0) -> float:
    """
    Multiplicative geometry prior P(label | geometry) for CLIP score
    adjustment (Heo et al. §3.1: combining visual + geometric cues).
    Returns > 1.0 when geometry supports the label, < 1.0 when inconsistent.

    Key improvements over v5:
    - Whiteboard/blackboard get a strong boost for large wall-mounted surfaces
      and a hard penalty when the object is too small to be a board.
    - Monitor/laptop penalty is applied when the object is unusually large
      (resolves whiteboard → monitor misclassification).
    - Keyboard gets strict size AND flatness constraints to eliminate false
      positives on grid-pattern factory surfaces.
    - n_world_pts used as a soft size gate: labels that describe tiny objects
      (keyboard, mouse, clock, bottle) are suppressed on high-point objects.
    """
    bx = float(bbox_size[0]) if len(bbox_size) > 0 else 1.0
    by = float(bbox_size[1]) if len(bbox_size) > 1 else 1.0
    bz = float(bbox_size[2]) if len(bbox_size) > 2 else 1.0
    horiz    = max(bx, bz)
    area_xz  = bx * bz              # footprint area
    aspect_h = horiz / max(by, 1e-3)  # horiz / vert

    # ── Lighting ────────────────────────────────────────────────────────
    if label == "ceiling_light":
        return 3.0 if ceil_dist < 0.4 else (0.1 if ceil_dist > 1.0 else 1.0)
    if label == "pendant_lamp":
        return 2.0 if 1.5 < h < 3.2 else 0.3
    if label in ("wall_light", "exit_sign"):
        return 1.8 if 1.3 < h < 2.5 else 0.4
    if label == "floor_lamp":
        return 1.5 if 0.5 < h < 2.2 and horiz < 0.5 else 0.4
    if label == "desk_lamp":
        return 1.5 if 0.7 < h < 1.3 and horiz < 0.4 else 0.3

    # ── Seating ────────────────────────────────────────────────────────
    if label in ("chair", "stool", "bench"):
        return 1.8 if 0.2 < h < 0.95 else 0.4
    if label == "sofa":
        return 1.8 if 0.2 < h < 0.8 and horiz > 0.8 else 0.4

    # ── Horizontal surfaces (desks, tables) ───────────────────────────
    if label in ("desk", "table", "counter"):
        return 2.0 if 0.6 < h < 1.05 and horiz > 0.4 else 0.3
    if label == "workbench":
        return 2.0 if 0.7 < h < 1.1 and horiz > 0.5 else 0.3

    # ── Storage ────────────────────────────────────────────────────────
    if label in ("cabinet", "locker"):
        return 1.8 if h > 0.5 and by > 0.5 else 0.4
    if label in ("shelf", "rack", "bookshelf"):
        return 1.8 if h > 0.4 and (by > 0.5 or horiz > 0.6) else 0.4

    # ── SCREENS vs BOARDS — the critical discriminator ─────────────────
    # Whiteboards and blackboards: wide wall-mounted surfaces.
    # Rule of thumb: width > 0.8 m AND height (by) > 0.4 m.
    # Penalty applies to anything too small to plausibly be a board.
    if label in ("whiteboard", "blackboard"):
        wide_enough  = horiz > 0.75          # at least 75 cm wide
        tall_enough  = by > 0.35             # at least 35 cm tall
        wall_height  = 0.5 < h < 3.0        # mounted in wall range
        large_pts    = n_world_pts > 15      # enough coverage for a large surface
        if wide_enough and tall_enough and wall_height and large_pts:
            return 3.5
        if not wide_enough or not tall_enough:
            return 0.10   # definitely not a board if too small

    # Monitors and laptops: small desktop devices, typically < 70 cm wide.
    # Hard penalty when the bounding box is monitor-sized or larger — this
    # prevents a 1.5 m whiteboard from being labelled "monitor".
    if label in ("monitor", "tv_screen"):
        if horiz > 1.0:          # > 1 m wide → almost certainly not a monitor
            return 0.05
        if horiz > 0.7:          # borderline — apply moderate penalty
            return 0.5
        return 1.8 if 0.7 < h < 1.7 else 0.3

    if label == "laptop":
        if horiz > 0.6:
            return 0.05          # laptops are never 60 cm+ wide
        return 1.8 if 0.7 < h < 1.5 else 0.3

    # ── Keyboards — strict size + flatness gate ────────────────────────
    # A keyboard is flat (by < 0.08 m) and small (horiz 0.3–0.5 m).
    # Any object taller than 15 cm or wider than 60 cm is not a keyboard.
    if label == "keyboard":
        flat_enough  = by < 0.15
        small_enough = 0.20 < horiz < 0.60
        desk_h       = 0.65 < h < 1.25
        few_pts      = n_world_pts < 150     # keyboards are small
        if flat_enough and small_enough and desk_h and few_pts:
            return 1.5
        return 0.05   # very strong penalty — keyboards are easily hallucinated

    if label == "mouse":
        return 1.2 if by < 0.08 and horiz < 0.20 and n_world_pts < 50 else 0.05

    # ── Structural / architectural ────────────────────────────────────
    if label in ("pillar", "column", "support_beam"):
        return 2.5 if by > 1.5 and horiz < 0.6 else 0.2
    if label == "duct":
        return 2.0 if h > 1.8 and horiz > 0.2 else 0.4
    if label == "railing":
        return 1.8 if 0.7 < h < 1.3 and horiz > 0.3 else 0.4

    # ── Floor-level items ─────────────────────────────────────────────
    if label in ("pallet",):
        return 2.0 if h < 0.25 and area_xz > 0.4 else 0.2
    if label in ("box", "crate", "barrel"):
        return 1.8 if h < 0.8 else (0.3 if h > 1.5 else 1.0)
    if label == "trash_bin":
        return 1.5 if 0.05 < h < 0.9 and horiz < 0.7 else 0.5

    # ── Industrial ────────────────────────────────────────────────────
    if label in ("machine", "workstation"):
        return 1.8 if area_xz > 0.3 and h > 0.4 else 0.5
    if label == "conveyor":
        return 2.0 if area_xz > 0.5 and h < 1.5 else 0.4
    if label in ("control_panel", "electrical_panel"):
        return 1.8 if 0.8 < h < 2.2 and horiz < 1.0 else 0.4
    if label in ("fire_extinguisher",):
        return 1.8 if 0.3 < h < 1.5 and horiz < 0.25 and n_world_pts < 80 else 0.4

    # ── Small items — penalise when the object is large ───────────────
    SMALL_LABELS = {"keyboard", "mouse", "clock", "bottle", "cup", "phone",
                    "tablet", "bag", "first_aid"}
    if label in SMALL_LABELS and n_world_pts > 200:
        return 0.1   # large objects shouldn't be classified as small items

    return 1.0


# ══ 1. Load ═══════════════════════════════════════════════════════════════

def load_pipeline(sp_paths: dict, use_refined_ids: bool = True) -> dict:
    out_dir = sp_paths["out_dir"]
    print(f"[sg] loading pipeline outputs from {out_dir} …")
    meta    = json.loads((out_dir / "metadata.json").read_text())
    embs    = np.load(str(out_dir / "embeddings.npy"))
    refined_f = out_dir / "object_ids_refined.npy"
    if use_refined_ids and refined_f.exists():
        obj_ids_path = refined_f
        print(f"[sg] using {refined_f.name} (from 03c_refine_objects.py)")
    else:
        obj_ids_path = out_dir / "object_ids.npy"
    obj_ids = np.load(str(obj_ids_path))
    n_unique = int(obj_ids.max()) + 1 if len(obj_ids) else 0
    print(f"[sg] {len(meta)} proposals · {n_unique} raw object_ids")
    return {"meta": meta, "embs": embs, "obj_ids": obj_ids}


def _trimmed_centroid(pts: np.ndarray, q: float = 0.12) -> np.ndarray:
    """
    Robust centroid: remove the outer q-fraction of points in each axis
    before computing the median.  Panoramic reprojections include off-axis
    noise from wide crops; trimming reduces centroid drift by ~10–30 cm on
    flat objects (whiteboards, screens) observed from many viewpoints.

    Inspired by the robust-localisation step in LERF (Kerr et al. 2023) which
    identifies object centres via the density peak of multi-view language-field
    queries rather than a raw mean of all activated voxels.
    """
    if len(pts) < 6:
        return np.median(pts, axis=0)
    lo = np.percentile(pts, q * 100,         axis=0)
    hi = np.percentile(pts, (1.0 - q) * 100, axis=0)
    mask = np.all((pts >= lo) & (pts <= hi), axis=1)
    core = pts[mask]
    return np.median(core if len(core) >= 3 else pts, axis=0)


BBOX_TRIM_Q = 0.10   # per-axis percentile trim for bounding-box extent


def _trimmed_bbox(pts: np.ndarray, q: float = BBOX_TRIM_Q) -> tuple:
    """
    Robust per-axis bounding box: percentile-trimmed extent instead of raw
    min/max. Each object's "points" here are single noisy backprojected
    world_pos estimates from many different views/frames, not a dense
    surface scan — one bad-depth view is enough to blow the raw AABB out
    to several times the object's real size. Trimming the outer q-fraction
    per axis keeps the box close to the true footprint while still growing
    with genuine multi-view spread.
    """
    if len(pts) < 6:
        return pts.min(axis=0), pts.max(axis=0)
    lo = np.percentile(pts, q * 100,         axis=0)
    hi = np.percentile(pts, (1.0 - q) * 100, axis=0)
    return lo, hi


# ── Point-cloud surface snap ───────────────────────────────────────────────

def load_ply_for_snap(sp_paths: dict):
    """
    Load the downsampled scan PLY and build a 2D KD-tree on (X, Z) for fast
    horizontal-plane nearest-neighbour lookups.

    Returns (xyz_np, xz_tree, y_lo, y_hi) or None if the PLY is unavailable.
    """
    from plyfile import PlyData
    from scipy.spatial import cKDTree

    ply_path = sp_paths.get("pointcloud")
    if not ply_path or not Path(ply_path).exists():
        print("[sg] no PLY for snap — centroid correction disabled")
        return None
    print(f"[sg] loading PLY for centroid snap: {ply_path} …")
    ply  = PlyData.read(str(ply_path))
    xyz  = np.column_stack([
        ply["vertex"]["x"].astype(np.float32),
        ply["vertex"]["y"].astype(np.float32),
        ply["vertex"]["z"].astype(np.float32),
    ])
    xz_tree = cKDTree(xyz[:, [0, 2]])
    y_lo, y_hi = float(xyz[:, 1].min()), float(xyz[:, 1].max())
    print(f"[sg] PLY: {len(xyz):,} pts  Y=[{y_lo:.2f}, {y_hi:.2f}]")
    return xyz, xz_tree, y_lo, y_hi


def _snap_centroid_to_ply(centroid: np.ndarray,
                           ply_xyz: np.ndarray,
                           xz_tree,
                           xz_radius: float = PLY_SNAP_XZ_M,
                           min_support: int = PLY_SNAP_MIN_PTS,
                           max_correction_m: float = PLY_SNAP_MAX_CORRECTION_M
                           ) -> np.ndarray | None:
    """
    Correct the height (Y) of an object centroid by snapping it to the nearest
    point-cloud surface cluster in the horizontal neighbourhood.

    Algorithm:
      1. Query all scanned points within xz_radius in the (X,Z) plane.
      2. If fewer than min_support → return None (object is in empty space).
      3. Cluster the found Y values by gaps > 0.5 m (floor / desk / ceiling).
      4. Pick the cluster whose median Y is closest to the current centroid Y.
      5. Replace centroid Y with that cluster's median — but only if it's
         within max_correction_m of the original; otherwise keep the
         original Y as-is.

    The (X, Z) coordinates are kept as-is; only Y is corrected.

    Near a continuously-scanned vertical surface (a wall, a pillar) the
    scanned returns span floor-to-ceiling with no 0.5 m gap anywhere, so step
    3 collapses everything into a single cluster whose median sits near the
    room's vertical middle — regardless of the object's real height. Without
    max_correction_m, step 5 would blindly apply that middle-of-the-room
    value to every object near such a surface (confirmed: a ceiling light
    whose 73 independent raw observations all agreed on Y≈1.6-1.7 m, next to
    a wall with continuous floor-to-ceiling coverage, got snapped down to
    Y≈0.07 m). The already-backprojected Y is informed by many independent
    views and is trustworthy on its own; the point-cloud snap should only
    nudge it onto a genuinely nearby surface, not relocate it wholesale.
    """
    idxs_list = xz_tree.query_ball_point([[centroid[0], centroid[2]]], r=xz_radius)
    idxs = idxs_list[0]
    if len(idxs) < min_support:
        return None

    near_y = np.sort(ply_xyz[idxs, 1])

    # Split into surface clusters separated by gaps > 0.5 m
    gaps = np.diff(near_y)
    split_pts = np.where(gaps > 0.50)[0] + 1
    clusters = np.split(near_y, split_pts)

    # Select the cluster closest to the (possibly clipped) centroid Y
    cy = float(centroid[1])
    best = min(clusters, key=lambda c: abs(float(np.median(c)) - cy))
    best_y = float(np.median(best))

    corrected = centroid.copy()
    if abs(best_y - cy) <= max_correction_m:
        corrected[1] = best_y
    return corrected


# ══ 2. Build & deduplicate objects ════════════════════════════════════════

def build_objects(data: dict, min_world_pts: int, min_proposals: int,
                  max_nodes: int, dedup_m: float, dedup_cos: float,
                  ply_snap_data=None, corrected_centroids: dict | None = None) -> dict:
    """
    Build per-object aggregates, then deduplicate using BOTH spatial proximity
    AND embedding cosine similarity (ConceptGraphs §IIA Object Association).

    Two objects are considered the same physical instance only if:
        centroid_distance < dedup_m  AND  cosine(emb_a, emb_b) > dedup_cos

    This prevents merging a chair and a trash_bin that happen to be adjacent.

    ply_snap_data: optional tuple (xyz, xz_tree, y_lo, y_hi) from
        load_ply_for_snap().  When provided:
          - world_pos Y values are clipped to [y_lo-slack, y_hi+slack] before
            centroid computation, removing clearly out-of-range depth estimates
            caused by erroneous equirectangular-to-3D backprojection.
          - The centroid is then snapped to the nearest point-cloud surface
            cluster (see _snap_centroid_to_ply).  Objects with no nearby
            surface support are discarded as hallucinations.

    corrected_centroids: optional {raw_oid: [x,y,z]} from
        pipeline/03c_refine_objects.py's BEV convex-hull correction
        (out_dir/corrected_centroids.json). When present for an object, its
        X/Z (not Y — Y still comes from the point-cloud snap below, which
        validates against real geometry that 03c's method doesn't check)
        replaces the trimmed-median estimate before that snap runs. This is a
        less occlusion-biased XZ estimate (FastSAM often only sees one face
        of an object, e.g. the top of a table) than the raw per-view median.
    """
    meta    = data["meta"]
    embs    = data["embs"]
    obj_ids = data["obj_ids"]

    ply_xyz, xz_tree, ply_y_lo, ply_y_hi = (
        ply_snap_data if ply_snap_data is not None
        else (None, None, -1e9, 1e9)
    )
    y_clip_lo = ply_y_lo - PLY_Y_SLACK_M
    y_clip_hi = ply_y_hi + PLY_Y_SLACK_M

    members: dict = defaultdict(list)
    for ri, oid in enumerate(obj_ids):
        members[int(oid)].append(ri)

    n_snap_discarded = 0
    objects: dict = {}
    for oid, rows in members.items():
        if len(rows) < min_proposals:
            continue
        valid_rows = [r for r in rows if meta[r].get("world_pos") is not None]
        if len(valid_rows) < min_world_pts:
            continue

        pts_raw = np.array([meta[r]["world_pos"] for r in valid_rows], dtype=np.float64)

        # Clip Y to the building's actual vertical range before centroid
        # computation.  Out-of-range depth estimates from panoramic
        # backprojection are clamped to the building boundary so they don't
        # pull the centroid outside the real scene.
        if ply_xyz is not None:
            pts = pts_raw.copy()
            pts[:, 1] = np.clip(pts[:, 1], y_clip_lo, y_clip_hi)
        else:
            pts = pts_raw

        centroid = _trimmed_centroid(pts)

        # Prefer 03c's BEV-corrected XZ (less biased by which face of the
        # object was actually visible) when available; Y is left for the
        # point-cloud snap below to validate against real geometry.
        if corrected_centroids is not None and oid in corrected_centroids:
            cx, _, cz = corrected_centroids[oid]
            centroid = np.array([cx, centroid[1], cz], dtype=centroid.dtype)

        # Snap centroid to the nearest point-cloud surface cluster; discard
        # objects with no nearby surface support (hallucinations in empty
        # space).
        if ply_xyz is not None:
            snapped = _snap_centroid_to_ply(centroid, ply_xyz, xz_tree)
            if snapped is None:
                n_snap_discarded += 1
                continue
            centroid = snapped
        rep_emb  = embs[rows].mean(axis=0).astype(np.float32)
        norm     = float(np.linalg.norm(rep_emb))
        rep_emb /= max(norm, 1e-8)
        scores    = [meta[r].get("score", 0.0) for r in rows]
        # Per-object geometric stats for descriptor (Heo et al. Eq. 5)
        sigma     = np.std(pts, axis=0).tolist()
        _bbox_min, _bbox_max = _trimmed_bbox(pts)
        _bbox_sz  = _bbox_max - _bbox_min
        bbox_size = _bbox_sz.tolist()
        volume    = float(max(float(np.prod(_bbox_sz)), 1e-6))
        max_side  = float(max(float(_bbox_sz.max()), 1e-6))
        objects[oid] = {
            "centroid":    centroid.tolist(),
            "world_pts":   pts,
            "rep_emb":     rep_emb,
            "n_proposals": len(rows),
            "n_world_pts": len(valid_rows),
            "view_ids":    list({meta[r]["view_id"] for r in rows}),
            "top_rows":    sorted(rows, key=lambda r: -meta[r].get("score", 0.0))[:5],
            "bbox_pts":    pts,
            "bbox_min":    _bbox_min.tolist(),
            "bbox_max":    _bbox_max.tolist(),
            "sigma":       sigma,
            "bbox_size":   bbox_size,
            "volume":      volume,
            "max_side":    max_side,
        }

    print(f"[sg] {len(objects)} objects after quality filter "
          f"(min_world_pts={min_world_pts}, min_proposals={min_proposals})")
    if n_snap_discarded:
        print(f"[sg] point-cloud snap discarded {n_snap_discarded} objects with no surface support")

    # ── Joint spatial + semantic deduplication ─────────────────────────
    if dedup_m > 0 and len(objects) > 1:
        from scipy.spatial import cKDTree
        oids_list = sorted(objects.keys())
        cents     = np.array([objects[o]["centroid"] for o in oids_list])
        emb_mat   = np.array([objects[o]["rep_emb"]  for o in oids_list], dtype=np.float32)
        tree      = cKDTree(cents)
        spatial_pairs = list(tree.query_pairs(r=dedup_m))

        parent = {o: o for o in oids_list}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb: return
            if objects[ra]["n_world_pts"] < objects[rb]["n_world_pts"]:
                ra, rb = rb, ra
            parent[rb] = ra

        n_merged = 0
        for i, j in spatial_pairs:
            cos = float(np.dot(emb_mat[i], emb_mat[j]))
            if cos > dedup_cos:
                union(oids_list[i], oids_list[j])
                n_merged += 1

        new_objects = {o: objects[o] for o in oids_list if find(o) == o}
        removed = len(objects) - len(new_objects)
        if removed:
            print(f"[sg] dedup: merged {removed} duplicates "
                  f"(dist<{dedup_m}m AND cos>{dedup_cos}) → {len(new_objects)} objects")
        objects = new_objects

    # ── Cap by support quality (generous ceiling — not a hard 2000 limit) ──
    # The old default of 2000 was an arbitrary rendering-performance number,
    # not a real constraint. With proper room/object hierarchy and overlay
    # generation we can comfortably handle far more nodes. The cap here only
    # exists as a safety valve against truly pathological inputs.
    if len(objects) > max_nodes:
        sorted_oids = sorted(objects, key=lambda o: -objects[o]["n_world_pts"])
        objects = {o: objects[o] for o in sorted_oids[:max_nodes]}
        print(f"[sg] capped to {max_nodes} nodes (safety ceiling — "
              f"raise --max-nodes if you need more)")

    return objects


# ══ 2b. Absorb small objects into containers ═════════════════════════════
#
# A box sitting on a shelf should not be its own top-level node — the
# shelf should be. This is deliberately label-agnostic (geometry only).
# Two distinct real-world cases both count as "absorb the small thing into
# the container":
#   1. Full volumetric containment (e.g. an item fully enclosed inside a
#      detected cabinet/drawer volume).
#   2. Resting on top of / footprint-nested in the container — e.g. a box
#      sitting ON a shelf barely overlaps the shelf's own detected 3D
#      points at all (a shelf's points are its physical structure, not the
#      empty air above each level where items sit), but its XZ footprint
#      sits squarely within the shelf's footprint, right at/above its top.
#      This is the more common real case and is caught via XZ-footprint
#      overlap + a vertical-nesting/resting-contact check.
# The absorbing container keeps its OWN unmodified geometry (centroid/bbox/
# volume are not inflated by what's currently on/in it) — the point is a
# stable container box for change detection regardless of its contents,
# tracked via n_absorbed/absorbed_ids for provenance only.
ABSORB_SIZE_RATIO    = 0.25   # small object's volume must be <= this fraction
                               # of the container's volume to be a candidate
ABSORB_CONTAINMENT   = 0.70   # min overlap fraction (volumetric OR XZ-footprint)
ABSORB_RADIUS_M      = 4.0    # candidate-pair search radius (centroid distance)
ABSORB_VERT_MARGIN_M = 0.15   # vertical tolerance for "resting on top of" contact


def _aabb(obj: dict) -> tuple:
    """Robust (percentile-trimmed) axis-aligned bounding box — NOT centroid
    ± bbox_size/2, which can drift after point-cloud-snap correction (snap
    only touches Y, world_pts/bbox_pts are unchanged). Uses the same trimmed
    bbox_min/bbox_max stored by build_objects() so absorption geometry
    matches what's actually rendered in the viewer."""
    if "bbox_min" in obj and "bbox_max" in obj:
        return np.array(obj["bbox_min"]), np.array(obj["bbox_max"])
    pts = obj["bbox_pts"]
    return _trimmed_bbox(pts)


def _overlap_frac(smin, smax, bmin, bmax, axes: tuple) -> float:
    """Fraction of the small object's extent (along `axes`) that overlaps
    the big object's extent along the same axes."""
    inter = 1.0
    small_extent = 1.0
    for ax in axes:
        lo = max(smin[ax], bmin[ax])
        hi = min(smax[ax], bmax[ax])
        inter *= max(hi - lo, 0.0)
        small_extent *= max(smax[ax] - smin[ax], 1e-6)
    return inter / max(small_extent, 1e-9)


def _containment_score(smin, smax, bmin, bmax, vert_margin_m: float) -> float:
    """Best of: full 3D volumetric containment, or XZ-footprint overlap
    while the small object's base sits within/just above the container's
    vertical span (nested inside OR resting on top of it)."""
    vol_overlap = _overlap_frac(smin, smax, bmin, bmax, (0, 1, 2))
    xz_overlap  = _overlap_frac(smin, smax, bmin, bmax, (0, 2))
    vertical_ok = (smin[1] >= bmin[1] - vert_margin_m and
                  smin[1] <= bmax[1] + vert_margin_m)
    return max(vol_overlap, xz_overlap if vertical_ok else 0.0)


def absorb_contained_objects(objects: dict,
                              size_ratio_thr: float = ABSORB_SIZE_RATIO,
                              containment_frac: float = ABSORB_CONTAINMENT,
                              search_radius_m: float = ABSORB_RADIUS_M,
                              vert_margin_m: float = ABSORB_VERT_MARGIN_M) -> dict:
    """Fold small objects into much larger ones they sit on/in. Returns a
    new dict containing only the surviving top-level objects."""
    oids = list(objects.keys())
    if len(oids) < 2:
        return objects

    from scipy.spatial import cKDTree
    cents = np.array([objects[o]["centroid"] for o in oids])
    tree = cKDTree(cents)
    pairs = tree.query_pairs(r=search_radius_m)

    best_parent: dict[int, tuple[int, float]] = {}  # small_oid -> (container_oid, overlap_frac)
    n_size_eligible = 0
    best_overlap_seen = 0.0
    for i, j in pairs:
        oa, ob = oids[i], oids[j]
        vol_a = float(objects[oa].get("volume", 0.0))
        vol_b = float(objects[ob].get("volume", 0.0))
        if vol_a <= 0 or vol_b <= 0:
            continue
        small, big = (oa, ob) if vol_a <= vol_b else (ob, oa)
        vol_small, vol_big = objects[small]["volume"], objects[big]["volume"]
        if vol_small / max(vol_big, 1e-9) > size_ratio_thr:
            continue
        n_size_eligible += 1

        smin, smax = _aabb(objects[small])
        bmin, bmax = _aabb(objects[big])
        overlap_frac = _containment_score(smin, smax, bmin, bmax, vert_margin_m)
        best_overlap_seen = max(best_overlap_seen, overlap_frac)
        if overlap_frac < containment_frac:
            continue

        prev = best_parent.get(small)
        if prev is None or overlap_frac > prev[1]:
            best_parent[small] = (big, overlap_frac)

    print(f"[sg] absorption candidates: {len(pairs)} pairs within {search_radius_m}m, "
          f"{n_size_eligible} size-eligible (ratio<={size_ratio_thr}), "
          f"best overlap seen={best_overlap_seen:.3f} (need >={containment_frac})")

    def _resolve(o, seen=None):
        seen = seen or set()
        if o in seen or o not in best_parent:
            return o
        seen.add(o)
        return _resolve(best_parent[o][0], seen)

    n_absorbed = 0
    for oid in oids:
        if oid not in best_parent:
            continue
        root = _resolve(oid)
        if root == oid:
            continue  # cycle guard — keep as its own node
        objects[root].setdefault("absorbed_ids", []).append(oid)
        objects[oid]["_absorbed_into"] = root
        n_absorbed += 1

    surviving = {o: v for o, v in objects.items() if "_absorbed_into" not in v}
    n_containers = 0
    for obj in surviving.values():
        obj["n_absorbed"] = len(obj.get("absorbed_ids", []))
        if obj["n_absorbed"]:
            n_containers += 1

    print(f"[sg] absorption: {n_absorbed} small objects folded into "
          f"{n_containers} containers → {len(surviving)} top-level objects "
          f"(was {len(objects)})")
    return surviving


# ══ 3. Room detection — wall-segment occupancy grid (IRS-style) ══════════
#
# Chen et al. 2025 (IRS) §III-A "Room-level Semantic Segmentation":
#   For each pair of wall segments W_{j-1}, W_j:
#     c1 = overlap(W_{j-1}, W_j)        — spatial overlap ratio
#     c2 = cos(V_{j-1}, V_j)            — normal-vector orientation similarity
#   If c1 ≥ τ1 AND c2 ≥ τ2 → merge into the same wall structure.
#   Rooms are then the regions enclosed by the merged wall boundary.
#
# We adapt this to a dense colored point cloud (no per-point semantic
# labels available) by building a 2D occupancy grid from vertical point
# density: cells with high point-count across the full floor-to-ceiling
# height band are walls (since a wall contributes points at every height),
# while open floor space has point density concentrated only near
# floor/object height.
# This directly captures the "wall card" structure visible in image 4 —
# the thin red boundary outlines are exactly these high-density strips.
#
# Pipeline:
#   1. Voxelise the full point cloud into a 2D grid (cell = WALL_CELL_M).
#   2. For each cell, count points across multiple height bands.
#      A wall cell has points in ALL bands (floor→ceiling); a non-wall
#      cell only has points in 1-2 bands (just the floor, or just an
#      object's surface).
#   3. Threshold to get a binary wall mask. Skeletonise / dilate slightly
#      to close small gaps (sensor noise, doorways count as gaps too —
#      intentionally, since open doorways should NOT separate rooms,
#      matching how the IRS algorithm treats doors as part of the wall
#      structure only when they're closed/detected as a barrier; here we
#      rely on the gap being small enough that dilation bridges it while
#      true room-separating walls are too thick/long to bridge).
#   4. Flood-fill the COMPLEMENT of the wall mask (i.e. open floor area)
#      → each connected component is one room.
#   5. Assign every object to the room whose floor-fill polygon contains
#      its XZ centroid.

WALL_CELL_M     = 0.12   # occupancy grid cell size
WALL_MIN_BANDS  = 4      # min height bands a cell must span to be "wall"
WALL_N_BANDS    = 6      # number of height bands from floor to ceiling
WALL_DILATE_PX  = 2      # morphological dilation to close small gaps
MIN_ROOM_AREA_M2 = 4.0   # discard flood-fill blobs smaller than this

# A tall shelf/rack row spans enough height bands to pass the WALL_MIN_BANDS
# test above just like a real wall does, so the flood-fill alone can carve
# a piece of furniture-adjacent floor off into its own spurious "room" —
# reproducible on some scans and not others depending on point coverage
# right at the furniture, which is exactly why the SAME physical room can
# flood-fill into a different room count between two capture sessions. The
# fix isn't a size/count threshold (a real small room and a furniture-cut
# sliver can be the same area) — it's checking what actually separates two
# flood-fill regions: real walls run continuously for a couple of meters;
# a furniture-induced gap is a short, localized pinch. So for every pair of
# candidate rooms found close enough to share a border, we look at just the
# wall cells lying in that shared border zone (not the room's whole
# boundary) and require them to form a run at least ROOM_MERGE_MIN_WALL_LEN_M
# long (via PCA major axis) before trusting the split; otherwise the two
# regions are re-merged. Confirmed on factory_space_13 vs factory_space_14
# (two scans of the same physical room): this collapses 13's spurious
# 4-room result to the correct 2, matching 14, without changing either
# space's two genuine rooms (which aren't even border-adjacent at this
# search radius — a real dividing wall has actual separation, unlike a
# furniture-occlusion pinch).
ROOM_MERGE_BORDER_PX     = WALL_DILATE_PX + 1  # search radius (cells) for
                                                # "these two regions share
                                                # a border worth checking"
ROOM_MERGE_MIN_WALL_LEN_M = 1.8  # min real-world length a border's own
                                  # wall segment must span to be trusted as
                                  # a genuine dividing wall, not clutter


# ── Building yaw (rotation offset from world X/Z axes) ────────────────────
# The captured point cloud is almost never perfectly axis-aligned to world
# X/Z, but our object boxes are computed as axis-aligned bounding boxes —
# so they visibly tilt relative to the actual (rotated) walls. This detects
# the building's true orientation once, so boxes can be computed/rendered
# in a wall-aligned frame instead.
def _detect_building_yaw_deg(wall_mask: np.ndarray, x_min: float, z_min: float,
                              wall_cell_m: float) -> float:
    """
    Fit the minimum-area bounding rectangle (cv2.minAreaRect) over all
    wall-cell world coordinates. A rectilinear building's tightest enclosing
    rectangle is aligned with its true orientation, so the rectangle's tilt
    IS the correction angle needed to make boxes hug the walls.
    Normalised to (-45, 45] since a rectangle's orientation repeats every 90°.
    """
    gxs, gzs = np.where(wall_mask)
    if len(gxs) < 10:
        return 0.0
    wx = x_min + (gxs + 0.5) * wall_cell_m
    wz = z_min + (gzs + 0.5) * wall_cell_m
    pts = np.column_stack([wx, wz]).astype(np.float32)
    try:
        import cv2
        (_, _), (_, _), angle = cv2.minAreaRect(pts)
    except Exception:
        return 0.0
    angle = float(angle) % 90.0
    if angle > 45.0:
        angle -= 90.0
    return angle


def _rotate_xz_deg(xz: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate (N,2) XZ points by theta_deg (standard 2D rotation matrix).
    Used both to project world points into the room-aligned frame
    (theta_deg = -yaw) and to place a room-aligned shape back into world
    space (theta_deg = +yaw) — same formula, opposite sign, exact inverses."""
    if theta_deg == 0.0:
        return xz
    rad = np.radians(theta_deg)
    c, s = np.cos(rad), np.sin(rad)
    x, z = xz[:, 0], xz[:, 1]
    return np.column_stack([x * c - z * s, x * s + z * c])


def _draw_area_blob(ax, member_xz: np.ndarray, radius_m: float,
                    bounds: dict, color, alpha: float) -> None:
    """Draw an area as the UNION of a radius_m disk around each member
    point — a "buffer union" blob — instead of a hull/boundary shape.
    bounds is the topdown.png bounds.json dict (u_min/u_max/v_min/v_max/
    width/height/v_flipped) — member_xz's columns are the fixed world
    (x, z) pair, same convention as every centroid elsewhere in this file.
    Converted to (u, v) internally via bounds' axis_u/axis_v, since which
    world axis is "u" vs "v" is a per-space choice (whichever gives the
    better-fitting topdown image aspect ratio) — assuming (x, z) already
    equals (u, v) transposes the whole blob whenever axis_u != 0.

    This replaces two earlier attempts (an axis-aligned bounding box, then
    a convex hull, then a Delaunay-based concave "alpha shape") that each
    had a real correctness bug: a bounding box and a convex hull both
    bridge straight across any real hole/obstacle an area's objects wrap
    around; a Delaunay alpha shape can outright DROP a point from the
    drawn shape whenever that point's nearest Delaunay-triangle edges are
    all longer than the alpha threshold — even though the point is a
    legitimate member connected via a *different*, non-Delaunay path
    (exactly the DBSCAN chain-of-hops test _kd_split_area uses to decide
    area membership in the first place, which is a different graph than
    Delaunay's "natural neighbors").

    The buffer-union has neither failure: every point's own disk always
    contains that point (nothing can ever be dropped), and two points
    farther apart than ~2*radius_m never have overlapping disks, so the
    blob naturally leaves a real hole/gap uncovered without any hull or
    triangulation logic at all. Implemented via density contouring (bin a
    grid, mark cells within radius_m of any member point, let matplotlib's
    contourf trace the resulting region, holes included) rather than an
    explicit polygon union, so it needs no extra dependency and can't
    raise on degenerate/collinear input the way ConvexHull/Delaunay can."""
    from scipy.spatial import cKDTree

    ax_u = bounds.get("axis_u", 0)
    member_xz = member_xz if ax_u == 0 else member_xz[:, ::-1]

    lo = member_xz.min(0) - radius_m * 1.5
    hi = member_xz.max(0) + radius_m * 1.5
    span = np.maximum(hi - lo, radius_m * 4)
    center = member_xz.mean(0)
    lo, hi = center - span / 2, center + span / 2

    cell = max(radius_m * 0.2, 0.05)
    nu = int(np.clip(span[0] / cell, 24, 200))
    nv = int(np.clip(span[1] / cell, 24, 200))
    u_vals = np.linspace(lo[0], hi[0], nu)
    v_vals = np.linspace(lo[1], hi[1], nv)
    gu, gv = np.meshgrid(u_vals, v_vals)

    tree = cKDTree(member_xz)
    dist, _ = tree.query(np.column_stack([gu.ravel(), gv.ravel()]), k=1)
    occ = (dist <= radius_m).reshape(gu.shape).astype(float)
    if occ.min() == occ.max():
        return   # grid entirely in/out (shouldn't happen with the padding above)

    u_min, u_max = bounds["u_min"], bounds["u_max"]
    v_min, v_max = bounds["v_min"], bounds["v_max"]
    W, H = bounds["width"], bounds["height"]
    px_x = (gu - u_min) / max(u_max - u_min, 1e-6) * W
    pf = (gv - v_min) / max(v_max - v_min, 1e-6)
    px_y = (1 - pf) * H if bounds.get("v_flipped") else pf * H

    ax.contourf(px_x, px_y, occ, levels=[0.5, 1.5], colors=[color],
                alpha=alpha, zorder=0)


def apply_building_yaw(objects: dict, yaw_deg: float) -> dict:
    """
    Recompute each object's bbox_size/bbox_min/bbox_max in the building's
    wall-aligned frame instead of raw world X/Z axes, so rendered boxes hug
    the walls instead of appearing tilted relative to a scan that isn't
    itself axis-aligned. Centroid position is unaffected — only the box's
    extent (and, in the viewer, its rendered rotation) changes. Run this
    LAST among geometry passes (after absorption/height-correction), since
    it's the authoritative final bbox and reads each object's live
    (possibly Y-shifted) bbox_pts.

    Detector-sourced objects (marked by a `det_class` field — pipeline4)
    use a much gentler trim than BBOX_TRIM_Q here. That constant was
    calibrated for pipeline2b's own points: sparse, noisy multi-view
    backprojection estimates where discarding the outer 10% per axis
    removes real backprojection noise. Pipeline4's points are dense,
    roughly-uniform 3D scan crops — the SAME 10% trim can collapse a real
    dimension there instead: verified concretely, a genuine ~0.85m-deep
    storage rack came out at bbox depth 0.2m under BBOX_TRIM_Q, reading as
    a thin wall/door-like sheet to every downstream structure check, while
    a 1%-per-axis trim recovered its true ~0.85m depth. Pipeline2b's own
    nodes (no det_class field) are completely unaffected by this.
    """
    if abs(yaw_deg) < 0.05:
        return objects
    for obj in objects.values():
        pts = obj.get("bbox_pts")
        if pts is None or len(pts) == 0:
            continue
        pts = np.asarray(pts)
        xz_local = _rotate_xz_deg(pts[:, [0, 2]], -yaw_deg)
        pts_local = np.column_stack([xz_local[:, 0], pts[:, 1], xz_local[:, 1]])
        q = 0.01 if obj.get("det_class") is not None else BBOX_TRIM_Q
        bmin, bmax = _trimmed_bbox(pts_local, q=q)
        obj["bbox_size"] = (bmax - bmin).tolist()
        obj["bbox_min"]  = bmin.tolist()   # room-aligned frame (u, y, v)
        obj["bbox_max"]  = bmax.tolist()   # room-aligned frame (u, y, v)
        # True box center in WORLD coordinates (rotate the aligned-frame
        # midpoint back) — the viewer must place the wireframe here, not at
        # the point-cloud mean: on a skewed point distribution the mean sits
        # off-center and the rendered box misses part of the object.
        mid = (bmin + bmax) / 2.0
        mid_xz = _rotate_xz_deg(np.array([[mid[0], mid[2]]]), yaw_deg)[0]
        obj["box_center"] = [float(mid_xz[0]), float(mid[1]), float(mid_xz[1])]
    return objects


def _merge_weak_room_splits(labeled: np.ndarray, n_components: int,
                            wall_mask: np.ndarray, cell_m: float,
                            border_px: int = ROOM_MERGE_BORDER_PX,
                            min_wall_len_m: float = ROOM_MERGE_MIN_WALL_LEN_M
                            ) -> np.ndarray:
    """Re-merge any two flood-fill regions whose shared border isn't backed
    by a long-enough wall run to trust as a real dividing wall (see
    ROOM_MERGE_MIN_WALL_LEN_M's comment for why this is needed at all).
    Returns a relabeled copy of `labeled` with weakly-separated regions
    unioned together; component ids are otherwise unchanged."""
    from scipy import ndimage
    from itertools import combinations

    comp_ids = [c for c in range(1, n_components + 1) if (labeled == c).any()]
    comp_masks = {c: (labeled == c) for c in comp_ids}
    comp_dil   = {c: ndimage.binary_dilation(m, iterations=border_px)
                 for c, m in comp_masks.items()}

    parent = {c: c for c in comp_ids}
    def find(c):
        while parent[c] != c:
            c = parent[c]
        return c
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in combinations(comp_ids, 2):
        if not (comp_dil[a] & comp_dil[b]).any():
            continue
        border = wall_mask & comp_dil[a] & comp_dil[b]
        n_cells = int(border.sum())
        if n_cells == 0:
            union(a, b)   # touching with no wall at all between them
            continue
        ys, xs = np.where(border)
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        pts -= pts.mean(0)
        if len(pts) >= 2:
            cov = pts.T @ pts / max(len(pts) - 1, 1)
            evals, evecs = np.linalg.eigh(cov)
            major_len = float(np.ptp(pts @ evecs[:, np.argmax(evals)])) * cell_m
        else:
            major_len = 0.0
        if major_len < min_wall_len_m:
            union(a, b)

    out = labeled.copy()
    for c in comp_ids:
        root = find(c)
        if root != c:
            out[labeled == c] = root
    return out


def detect_rooms(sp_paths: dict, eps_m: float, min_pts: int,
                 slice_lo: float, slice_hi: float, subsample: int,
                 n_rooms_hint: int = 0,
                 wall_cell_m: float = WALL_CELL_M,
                 wall_min_bands: int = WALL_MIN_BANDS) -> dict:
    """
    Detect rooms via wall-segment occupancy grid + flood fill (IRS-inspired).

    Falls back to scanner-trajectory clustering if the point cloud is
    unavailable, and to a single global room if neither works.
    """
    from plyfile import PlyData

    ply_path = sp_paths.get("pointcloud")
    if not ply_path or not Path(ply_path).exists():
        print("[sg] no point cloud — falling back to trajectory clustering")
        return _detect_rooms_trajectory(sp_paths, eps_m, min_pts, n_rooms_hint)

    print(f"[sg] reading point cloud for wall-based room detection …")
    ply = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float32)
    print(f"[sg] {len(xyz):,} points")

    vert    = xyz[:, 1]
    floor_y = float(np.percentile(vert, 1))
    ceil_y  = float(np.percentile(vert, 99))
    print(f"[sg] floor={floor_y:.2f}m  ceil={ceil_y:.2f}m  "
          f"height={ceil_y-floor_y:.2f}m")

    # ── Build 2D occupancy grid with height-band point counts ────────────
    x, z = xyz[:, 0], xyz[:, 2]
    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())
    nx = max(1, int(np.ceil((x_max - x_min) / wall_cell_m)))
    nz = max(1, int(np.ceil((z_max - z_min) / wall_cell_m)))
    print(f"[sg] occupancy grid: {nx} x {nz} cells @ {wall_cell_m}m")

    gx = np.clip(((x - x_min) / wall_cell_m).astype(np.int32), 0, nx - 1)
    gz = np.clip(((z - z_min) / wall_cell_m).astype(np.int32), 0, nz - 1)

    band_h = (ceil_y - floor_y) / WALL_N_BANDS
    band   = np.clip(((vert - floor_y) / max(band_h, 1e-6)).astype(np.int32),
                     0, WALL_N_BANDS - 1)

    # band_mask[gx,gz] = bitmask of which height bands have ANY points
    flat_idx  = gx.astype(np.int64) * nz + gz
    band_bits = (1 << band).astype(np.int64)
    band_mask = np.zeros(nx * nz, dtype=np.int64)
    np.bitwise_or.at(band_mask, flat_idx, band_bits)
    band_mask = band_mask.reshape(nx, nz)

    n_bands_per_cell = np.zeros((nx, nz), dtype=np.int32)
    for b in range(WALL_N_BANDS):
        n_bands_per_cell += ((band_mask >> b) & 1)

    wall_mask = n_bands_per_cell >= wall_min_bands
    n_wall_cells = int(wall_mask.sum())
    print(f"[sg] {n_wall_cells} wall cells "
          f"({100*n_wall_cells/(nx*nz):.1f}% of grid, "
          f"threshold={wall_min_bands}/{WALL_N_BANDS} bands)")

    if n_wall_cells == 0:
        print("[sg] no wall cells found — falling back to trajectory clustering")
        return _detect_rooms_trajectory(sp_paths, eps_m, min_pts, n_rooms_hint)

    # ── Morphological dilation to close small gaps (IRS treats doors as
    #    part of wall structure; small sensor gaps shouldn't split rooms,
    #    but a real open doorway/corridor will be too wide to bridge) ────
    try:
        from scipy import ndimage
        wall_mask_dilated = ndimage.binary_dilation(
            wall_mask, iterations=WALL_DILATE_PX)
    except ImportError:
        wall_mask_dilated = wall_mask

    # ── Flood fill the complement (open floor space) ──────────────────
    open_mask = ~wall_mask_dilated
    # Also exclude cells with literally zero points (outside building footprint)
    has_any_pts = n_bands_per_cell > 0
    open_mask = open_mask & has_any_pts

    try:
        from scipy import ndimage
        labeled, n_components = ndimage.label(
            open_mask, structure=np.ones((3,3)))  # 8-connectivity
    except ImportError:
        print("[sg] scipy.ndimage unavailable — cannot flood-fill")
        return _detect_rooms_trajectory(sp_paths, eps_m, min_pts, n_rooms_hint)

    print(f"[sg] flood-fill found {n_components} connected floor regions")

    # ── Re-merge regions only weakly separated (short/no wall run at their
    #    shared border — see ROOM_MERGE_MIN_WALL_LEN_M) before area-filtering,
    #    so a furniture-induced pinch can't masquerade as a room boundary ──
    labeled = _merge_weak_room_splits(labeled, n_components, wall_mask, wall_cell_m)

    # ── Filter tiny noise blobs, keep real rooms ──────────────────────
    cell_area = wall_cell_m * wall_cell_m
    room_meta: dict = {"_floor_y": float(floor_y), "_ceil_y": float(ceil_y),
                       "_horiz_axes": [0, 2], "_from_wallgrid": True}
    valid_rooms = []
    for comp_id in range(1, n_components + 1):
        mask = labeled == comp_id
        n_cells = int(mask.sum())
        area_m2 = n_cells * cell_area
        if area_m2 < MIN_ROOM_AREA_M2:
            continue
        valid_rooms.append((comp_id, mask, area_m2))

    if n_rooms_hint > 0 and len(valid_rooms) != n_rooms_hint:
        print(f"[sg] found {len(valid_rooms)} rooms but --n-rooms={n_rooms_hint} "
              f"requested — keeping the {n_rooms_hint} largest by area")
        valid_rooms.sort(key=lambda t: -t[2])
        valid_rooms = valid_rooms[:n_rooms_hint]

    if not valid_rooms:
        print("[sg] no rooms passed area filter — falling back to trajectory")
        return _detect_rooms_trajectory(sp_paths, eps_m, min_pts, n_rooms_hint)

    for new_id, (comp_id, mask, area_m2) in enumerate(valid_rooms):
        gxs, gzs = np.where(mask)
        # Convert grid cells back to world XZ
        wx = x_min + (gxs + 0.5) * wall_cell_m
        wz = z_min + (gzs + 0.5) * wall_cell_m
        room_meta[new_id] = {
            "centroid_xz": [float(wx.mean()), float(wz.mean())],
            "n_pts":       int(mask.sum()),
            "xz_min":      [float(wx.min()), float(wz.min())],
            "xz_max":      [float(wx.max()), float(wz.max())],
            "area_m2":     round(float(area_m2), 1),
            # Store the cell mask as a set of (gx,gz) for accurate
            # point-in-room assignment (handles L-shaped rooms correctly,
            # unlike a simple bbox/centroid-distance check)
            "_cell_set":   set(zip(gxs.tolist(), gzs.tolist())),
        }
        print(f"    Room {new_id}: {area_m2:.1f} m²  "
              f"bbox=[{room_meta[new_id]['xz_min']}, {room_meta[new_id]['xz_max']}]  "
              f"centroid={[round(v,2) for v in room_meta[new_id]['centroid_xz']]}")

    room_meta["_room_cent_ids"] = list(range(len(valid_rooms)))
    room_meta["_room_cents_xz"] = np.array(
        [room_meta[i]["centroid_xz"] for i in range(len(valid_rooms))],
        dtype=np.float64)
    room_meta["_grid_origin"]  = [x_min, z_min]
    room_meta["_grid_cell_m"]  = wall_cell_m
    room_meta["_grid_shape"]   = [nx, nz]
    room_meta["_yaw_deg"] = _detect_building_yaw_deg(wall_mask, x_min, z_min, wall_cell_m)
    print(f"[sg] {len(valid_rooms)} rooms detected via wall-segment flood-fill")
    print(f"[sg] detected building yaw: {room_meta['_yaw_deg']:.2f}° "
          f"(boxes will be rotated by this to align with walls)")
    return room_meta


def _detect_rooms_trajectory(sp_paths: dict, eps_m: float, min_pts: int,
                             n_rooms_hint: int = 0) -> dict:
    """Fallback: cluster the scanner trajectory (used when point cloud
    wall detection is unavailable or fails)."""
    cameras_path = sp_paths.get("cameras")
    if not cameras_path or not Path(cameras_path).exists():
        print("[sg] no cameras.json either — returning single global room")
        return {}

    import re
    cams = json.loads(Path(cameras_path).read_text())
    sp_pos: dict[str, list] = {}
    for c in cams:
        pano = Path(c["pano"]).name
        m = re.match(r"(\d+)_pz", pano)
        sp_id = m.group(1) if m else pano
        if sp_id not in sp_pos:
            sp_pos[sp_id] = c["pos"]

    positions = np.array(list(sp_pos.values()), dtype=np.float32)
    xz = positions[:, [0, 2]]
    print(f"[sg] trajectory fallback: {len(xz)} scanpoints")

    if n_rooms_hint > 1:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_rooms_hint, random_state=42, n_init=10)
        labels = km.fit_predict(xz)
        unique_r = list(range(n_rooms_hint))
    else:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps_m, min_samples=min_pts,
                        algorithm="ball_tree").fit_predict(xz)
        unique_r = sorted(set(labels.tolist()) - {-1})
        if not unique_r:
            unique_r = [0]; labels = np.zeros(len(xz), dtype=int)

    room_meta: dict = {"_horiz_axes": [0, 2], "_from_traj": True}
    for r in unique_r:
        pts_r = xz[labels == r]
        span  = pts_r.max(0) - pts_r.min(0) if len(pts_r) > 1 else np.array([1.,1.])
        room_meta[r] = {
            "centroid_xz": pts_r.mean(axis=0).tolist(),
            "n_pts": int((labels == r).sum()),
            "xz_min": pts_r.min(axis=0).tolist(),
            "xz_max": pts_r.max(axis=0).tolist(),
            "area_m2": round(float(span[0]*span[1]), 1),
        }
    room_meta["_room_cent_ids"] = unique_r
    room_meta["_room_cents_xz"] = np.array(
        [room_meta[r]["centroid_xz"] for r in unique_r], dtype=np.float64)
    print(f"[sg] {len(unique_r)} rooms (trajectory fallback)")
    return room_meta


def assign_rooms(objects: dict, room_meta: dict) -> dict:
    """
    Assign each object to a room.

    If room_meta came from the wall-grid flood-fill (_from_wallgrid), use
    the EXACT cell membership (handles L-shaped / irregular rooms
    correctly). Otherwise fall back to nearest-centroid (trajectory mode).
    """
    if not room_meta or "_room_cents_xz" not in room_meta:
        for obj in objects.values(): obj["room_id"] = -1
        return objects

    if room_meta.get("_from_wallgrid"):
        ox, oz   = room_meta["_grid_origin"]
        cell_m   = room_meta["_grid_cell_m"]
        nx, nz   = room_meta["_grid_shape"]
        room_ids = room_meta["_room_cent_ids"]

        # Precompute cell→room lookup for O(1) assignment
        cell_to_room: dict = {}
        for rid in room_ids:
            for cell in room_meta[rid]["_cell_set"]:
                cell_to_room[cell] = rid

        from scipy.spatial import cKDTree
        tree = cKDTree(room_meta["_room_cents_xz"])

        n_exact = n_nearest = 0
        for obj in objects.values():
            c  = obj["centroid"]
            gx = int(np.clip((c[0]-ox)/cell_m, 0, nx-1))
            gz = int(np.clip((c[2]-oz)/cell_m, 0, nz-1))
            rid = cell_to_room.get((gx, gz))
            if rid is not None:
                obj["room_id"] = int(rid)
                n_exact += 1
            else:
                # Object's exact cell is a wall/gap cell (e.g. object
                # mounted ON a wall) — fall back to nearest room centroid
                _, idx = tree.query([[c[0], c[2]]], k=1)
                obj["room_id"] = int(room_ids[int(idx[0])])
                n_nearest += 1
        print(f"[sg] room assignment: {n_exact} exact-cell, "
              f"{n_nearest} nearest-centroid fallback")
        return objects

    # Trajectory-mode fallback: nearest centroid only
    from scipy.spatial import cKDTree
    room_ids = room_meta["_room_cent_ids"]
    tree     = cKDTree(room_meta["_room_cents_xz"])
    for obj in objects.values():
        c  = np.array(obj["centroid"])
        xz = np.array([[c[0], c[2]]])
        _, idx = tree.query(xz, k=1)
        obj["room_id"] = int(room_ids[int(idx[0])])
    return objects


# ══ 3b. Room → Area → Object hierarchy (gap-based recursive split) ═══════
#
# Extends the room decomposition one level further: divide each room into
# density-packed sub-areas, then divide each area into objects.
#
# Splits ONLY at a genuine physical gap between clusters of objects — a
# walkway/aisle wide enough to plausibly separate two different working
# areas — never at an arbitrary median just because a region's object count
# or footprint crossed a threshold. A single long row of desks/racks along
# one wall is one coherent working area even though its footprint is long;
# it should stay one area, not get chopped every ~2.5m.
#
# Gap detection is true 2D connectivity (DBSCAN with eps=AREA_MIN_GAP_M,
# min_samples=1 — same "chain of close hops" logic as geo_cluster's own
# clustering): two objects are in the same natural cluster iff reachable via
# a chain of hops each closer than AREA_MIN_GAP_M. An EARLIER version of
# this checked only the largest 1D gap projected onto the X axis and onto
# the Z axis separately — which misses a gap at any other orientation, e.g.
# objects that wrap around a small obstacle (a walled nook): points on
# either side can have fully overlapping X-ranges AND overlapping Z-ranges
# even though no straight path between the two groups is actually short,
# so neither 1D projection ever showed a "gap" and the two groups stayed
# one area with a hull that visually cut straight through the obstacle
# between them. 2D density connectivity has no preferred axis, so it
# catches this. If the whole group is already one connected cluster, it's
# one working area, however large or elongated — UNLESS a generous safety
# valve (AREA_MAX_OBJECTS / AREA_MAX_SIZE_M / AREA_MAX_ROOM_FRAC) is
# exceeded, in which case we fall back to a plain median split just to keep
# area sizes bounded for pathological cases (one enormous, gap-free floor).
AREA_MIN_GAP_M   = 3.2    # min real-world gap (m, along one axis) that
                          # counts as a genuine separation between areas —
                          # tuned empirically: 1.8m still split single working
                          # clusters at ordinary in-row spacing (a walkway
                          # between desks, not a real division); 3.2m only
                          # fires at real aisle-scale gaps.
AREA_MAX_OBJECTS = 40     # safety valve: force a split above this count
                          # even with no qualifying gap
AREA_MAX_SIZE_M  = 10.0   # safety valve: force a split above this footprint
                          # (either axis) even with no qualifying gap
AREA_MAX_ROOM_FRAC = 0.65 # an area may not span more than this fraction of
                          # its OWN ROOM's footprint on either axis — a
                          # "working area" that's basically the whole room
                          # defeats the point of the room→area→object
                          # hierarchy, even if the room happens to have no
                          # single gap ≥ AREA_MIN_GAP_M anywhere. Forces at
                          # least one split per room; only applies above
                          # AREA_MIN_ROOM_DIM_M so small rooms aren't
                          # chopped into meaningless slivers.
AREA_MIN_ROOM_DIM_M = 3.0


def _kd_split_area(oid_list: list, cent_xz: np.ndarray,
                    min_gap_m: float, max_objects: int, max_size_m: float,
                    room_extent: tuple, room_frac: float,
                    area_id_gen, areas_out: list, obj_area: dict) -> None:
    xs, zs = cent_xz[:, 0], cent_xz[:, 1]
    x_lo, x_hi = float(xs.min()), float(xs.max())
    z_lo, z_hi = float(zs.min()), float(zs.max())
    extent_x, extent_z = x_hi - x_lo, z_hi - z_lo

    room_ext_x, room_ext_z = room_extent
    too_much_of_room = (
        (room_ext_x >= AREA_MIN_ROOM_DIM_M and extent_x > room_frac * room_ext_x)
        or (room_ext_z >= AREA_MIN_ROOM_DIM_M and extent_z > room_frac * room_ext_z))
    must_split_anyway = (len(oid_list) > max_objects
                          or max(extent_x, extent_z) > max_size_m
                          or too_much_of_room)

    n_components = 1
    labels = None
    if len(oid_list) > 1:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=min_gap_m, min_samples=1).fit_predict(cent_xz)
        n_components = len(set(labels.tolist()))

    if n_components <= 1 and not must_split_anyway:
        aid = next(area_id_gen)
        for oid in oid_list:
            obj_area[oid] = aid
        areas_out.append({
            "id": aid, "n_objects": len(oid_list),
            "centroid_xz": [float(xs.mean()), float(zs.mean())],
            "xz_min": [x_lo, z_lo], "xz_max": [x_hi, z_hi],
            "member_ids": list(oid_list),
        })
        return

    if n_components > 1:
        # Real 2D-connectivity gap(s) found — split directly into the
        # natural clusters (no axis/median guessing needed).
        for lbl in sorted(set(labels.tolist())):
            idx = np.where(labels == lbl)[0]
            _kd_split_area([oid_list[i] for i in idx], cent_xz[idx],
                           min_gap_m, max_objects, max_size_m,
                           room_extent, room_frac, area_id_gen, areas_out, obj_area)
        return

    # Still one connected blob, but a safety valve tripped — fall back to a
    # plain median split on the larger-spread axis, just to terminate with
    # bounded area sizes.
    axis = 0 if extent_x >= extent_z else 1
    order = np.argsort(cent_xz[:, axis])
    split_at = len(order) // 2
    left, right = order[:split_at], order[split_at:]
    _kd_split_area([oid_list[i] for i in left],  cent_xz[left],
                   min_gap_m, max_objects, max_size_m, room_extent, room_frac,
                   area_id_gen, areas_out, obj_area)
    _kd_split_area([oid_list[i] for i in right], cent_xz[right],
                   min_gap_m, max_objects, max_size_m, room_extent, room_frac,
                   area_id_gen, areas_out, obj_area)


def _wall_connected_components(oid_list: list, objects: dict,
                                wall_blocker: dict) -> list[list]:
    """Split oid_list into groups where every member is reachable from
    every other member via a chain of hops whose straight-line segment
    never crosses a confirmed wall (transitive — like the room flood-fill,
    reusing the exact same wall-crossing test build_edges uses). Returns
    [oid_list] unchanged if already fully connected or there's nothing to
    check against."""
    n = len(oid_list)
    if n <= 1 or wall_blocker is None:
        return [oid_list]
    cents = [objects[oid]["centroid"] for oid in oid_list]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if not _segment_crosses_wall(cents[i], cents[j], wall_blocker):
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb

    groups: dict = {}
    for i, oid in enumerate(oid_list):
        groups.setdefault(find(i), []).append(oid)
    return list(groups.values())


def _refine_areas_by_wall(room_areas: list, objects: dict,
                          wall_blocker: dict | None, area_id_gen,
                          obj_area: dict) -> list:
    """Any area whose members aren't all mutually reachable without
    crossing a confirmed wall gets split along that wall. Catches real
    interior partitions (a walled-off nook that's technically still "open"
    to the main floor, so room detection's flood-fill didn't register it as
    a separate room) that the gap-based split has no way to see on its
    own — it only looks at centroid spacing, never at actual wall
    geometry."""
    if wall_blocker is None:
        return room_areas
    out = []
    for area in room_areas:
        components = _wall_connected_components(area["member_ids"], objects, wall_blocker)
        if len(components) <= 1:
            out.append(area)
            continue
        for comp in components:
            cent_xz = np.array([[objects[o]["centroid"][0], objects[o]["centroid"][2]]
                                for o in comp])
            aid = next(area_id_gen)
            for oid in comp:
                obj_area[oid] = aid
            out.append({
                "id": aid, "n_objects": len(comp),
                "centroid_xz": cent_xz.mean(0).tolist(),
                "xz_min": cent_xz.min(0).tolist(), "xz_max": cent_xz.max(0).tolist(),
                "member_ids": list(comp),
            })
    return out


def build_areas(objects: dict, min_gap_m: float = AREA_MIN_GAP_M,
                max_objects_per_area: int = AREA_MAX_OBJECTS,
                max_area_size_m: float = AREA_MAX_SIZE_M,
                max_room_frac: float = AREA_MAX_ROOM_FRAC,
                wall_blocker: dict | None = None,
                split: bool = True) -> tuple[dict, list]:
    """Partition each room's objects into areas, split only at genuine
    physical gaps between clusters (and, separately, wherever a confirmed
    wall actually separates them — see _refine_areas_by_wall). Returns
    (objects, areas) — objects gain an "area_id" field.

    split=False bypasses all of the above and makes each room exactly one
    area (its full object set) — a temporary passthrough for when the area
    subdivision itself is under revision and shouldn't affect room-level
    results."""
    from itertools import count

    by_room: dict = defaultdict(list)
    for oid, obj in objects.items():
        by_room[obj.get("room_id", -1)].append(oid)

    areas: list = []
    obj_area: dict = {}
    aid_gen = count(0)
    n_wall_split = 0
    for room_id, oids in by_room.items():
        cent_xz = np.array([[objects[o]["centroid"][0], objects[o]["centroid"][2]]
                            for o in oids])
        room_areas: list = []
        if not split:
            aid = next(aid_gen)
            for o in oids:
                obj_area[o] = aid
            room_areas.append({
                "id": aid, "n_objects": len(oids),
                "centroid_xz": cent_xz.mean(0).tolist(),
                "xz_min": cent_xz.min(0).tolist(), "xz_max": cent_xz.max(0).tolist(),
                "member_ids": list(oids),
            })
        elif len(oids) == 1:
            aid = next(aid_gen)
            obj_area[oids[0]] = aid
            room_areas.append({
                "id": aid, "n_objects": 1,
                "centroid_xz": cent_xz[0].tolist(),
                "xz_min": cent_xz[0].tolist(), "xz_max": cent_xz[0].tolist(),
                "member_ids": [oids[0]],
            })
        else:
            room_extent = (float(np.ptp(cent_xz[:, 0])), float(np.ptp(cent_xz[:, 1])))
            _kd_split_area(oids, cent_xz, min_gap_m, max_objects_per_area,
                          max_area_size_m, room_extent, max_room_frac,
                          aid_gen, room_areas, obj_area)
        if split:
            n_before = len(room_areas)
            room_areas = _refine_areas_by_wall(room_areas, objects, wall_blocker,
                                               aid_gen, obj_area)
            n_wall_split += len(room_areas) - n_before
        for a in room_areas:
            a["room_id"] = room_id
        areas.extend(room_areas)

    for oid, obj in objects.items():
        obj["area_id"] = obj_area.get(oid, -1)

    if not split:
        print(f"[sg] area split disabled — {len(areas)} areas == "
              f"{len(by_room)} rooms")
    else:
        print(f"[sg] {len(areas)} areas across {len(by_room)} rooms "
              f"(min gap {min_gap_m}m; safety valve: {max_objects_per_area} "
              f"objects or {max_area_size_m}m footprint"
              f"{f'; {n_wall_split} extra splits from wall crossings' if n_wall_split else ''})")
    return objects, areas


# ══ 4. Labelling — CLIP zero-shot (no API key needed) ════════════════════
#
# Strategy (two tiers, no external API):
#
# Tier 1 — CLIP zero-shot via open_clip
#   Load a CLIP image+text encoder (ViT-B/32 openai, ~300 MB, cached after
#   first download).  For each object encode its top-3 crops with the CLIP
#   image encoder and the full LABEL_VOCAB with the CLIP text encoder.
#   Assign the label whose text embedding has the highest mean cosine
#   similarity across the crops.  This is exactly what ConceptGraphs §II-A
#   uses (CLIP image embeddings + text probing).
#
# Tier 2 — Improved height+density heuristics
#   Now distinguishes ceiling_light / pendant_lamp / wall_light /
#   desk_lamp / floor_lamp, workbench vs desk, pallet vs box, etc.

def _build_clip_classifier(vocab: list[str]):
    """
    Returns (encode_image_fn, text_features) where:
      encode_image_fn(crops) → (N_crops, D) float32 numpy, L2-normalised
      text_features           → (N_labels, D) float32 numpy, L2-normalised

    Thin wrapper over clip_utils.build_clip_classifier (shared with
    03d_clip_instances.py and server.py's text-query endpoint) so the model
    loading / prompt-ensembling logic lives in exactly one place.
    Returns None on failure.
    """
    return clip_utils.build_clip_classifier(vocab, prompts_dict=LABEL_PROMPTS)


def label_objects(objects: dict, meta: list, views_dir: Path,
                  extra_labels: list) -> dict:
    """
    Tier 1: CLIP zero-shot classification (open_clip, no API key).
    Tier 2: Height + density heuristics (improved).
    """
    all_y   = [o["centroid"][1] for o in objects.values()]
    floor_y = float(np.percentile(all_y, 5)) if all_y else 0.0
    ceil_y  = float(np.percentile(all_y, 95)) if all_y else 3.0
    vocab   = list(dict.fromkeys(LABEL_VOCAB + extra_labels))

    clip_result = _build_clip_classifier(vocab)
    PAD = 0.10
    clip_count = heur_count = 0

    for oid, obj in objects.items():
        # ── Load top crops ────────────────────────────────────────────
        crops: list[Image.Image] = []
        for row_idx in obj["top_rows"]:
            m     = meta[row_idx]
            img_p = views_dir / Path(m["pano"]).name
            if not img_p.exists():
                continue
            try:
                pil = Image.open(img_p).convert("RGB")
                x1, y1, x2, y2 = m["bbox"]
                W, H = pil.size
                pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
                crop = pil.crop((max(0, x1 - pw), max(0, y1 - ph),
                                 min(W, x2 + pw), min(H, y2 + ph)))
                if crop.width > 10 and crop.height > 10:
                    crops.append(crop)
            except Exception:
                continue
            if len(crops) >= 3:
                break

        # Default: max entropy (all labels equally likely = most uncertain)
        label, score = None, 0.0
        entropy = float(np.log(max(len(vocab), 2)))

        # ── Tier 0: pre-detected label from YOLOWorld / GroundingDINO ────
        # Only use detector labels with score >= 0.15 to suppress low-conf
        # false positives (conf=0.05 proposal threshold is very permissive).
        # Majority-vote across the top crops; fall through to CLIP if none qualify.
        det_labels = [meta[r].get("label") for r in obj["top_rows"]
                      if meta[r].get("label") and meta[r].get("score", 0) >= 0.15]
        if det_labels:
            from collections import Counter
            winner, count = Counter(det_labels).most_common(1)[0]
            # Normalise to underscore format to match LABEL_VOCAB keys
            label = winner.strip().lower().replace(" ", "_")
            score = round(count / len(det_labels), 3)
            entropy = 0.1   # very low entropy — detector was explicit
            clip_count += 1
            obj["label"]         = label
            obj["caption"]       = ""
            obj["label_score"]   = score
            obj["label_entropy"] = entropy
            continue

        # ── Tier 1: CLIP zero-shot + geometry prior ───────────────────────
        if clip_result is not None and crops:
            encode_images, text_feats = clip_result
            try:
                img_feats = encode_images(crops)          # (N_crops, D)
                mean_img  = img_feats.mean(axis=0)
                mean_img /= max(float(np.linalg.norm(mean_img)), 1e-8)
                sims      = text_feats @ mean_img          # (N_labels,)

                # Geometry-adjusted scoring (Heo et al. §3.1 — combine
                # visual embedding with geometric prior for discriminative
                # object features; reduces label entropy → fewer edge errors)
                h         = float(obj["centroid"][1]) - floor_y
                ceil_dist = ceil_y - float(obj["centroid"][1])
                bbox_sz   = obj.get("bbox_size", [1., 1., 1.])
                n_pts     = obj.get("n_world_pts", 0)
                adj_sims  = sims.copy()
                for vi, vlabel in enumerate(vocab):
                    adj_sims[vi] *= _geometry_prior(
                        vlabel, h, bbox_sz, ceil_dist, n_world_pts=n_pts)

                best_idx  = int(np.argmax(adj_sims))
                label     = vocab[best_idx]
                score     = float(adj_sims[best_idx])

                # Classification entropy H(o|z) — paper §2: higher entropy
                # correlates with predicate errors under comparable relation freq.
                shifted = adj_sims * 5.0 - float(adj_sims.max()) * 5.0
                probs   = np.exp(shifted)
                probs  /= max(float(probs.sum()), 1e-10)
                entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
                clip_count += 1
            except Exception:
                pass   # fall through to heuristic

        # ── Tier 2: heuristics ────────────────────────────────────────────
        if label is None:
            label = _heuristic_label(obj["centroid"], obj["n_world_pts"],
                                     floor_y, ceil_y)
            score = 0.0
            heur_count += 1

        obj["label"]         = label
        obj["caption"]       = ""
        obj["label_score"]   = round(score, 3)
        obj["label_entropy"] = round(entropy, 4)

    print(f"[sg] labelled: CLIP={clip_count}  heuristic={heur_count}")
    return objects


# ══ 4b. 3DSSG-style node semantics: class hierarchy + attributes +
#        affordances (Wald et al., CVPR 2020, §3.1–3.2) ═══════════════════
#
# 3DSSG defines each node not by a single category but by (a) a HIERARCHY
# of classes c = (c1, ..., cd) from WordNet hypernym chains, (b) a set of
# ATTRIBUTES — static visual/physical properties plus dynamic state — and
# (c) AFFORDANCES, class-level interaction possibilities conditioned on the
# current state ("only a closed door can be opened").
#
# 3DSSG builds its hierarchy by parsing WordNet lexical definitions with a
# manual disambiguation pass over its 534 classes; our vocabulary is small
# and fixed, so the same disambiguated chains are written out by hand —
# identical outcome, no WordNet/NLTK dependency. Chains exclude the label
# itself (class_hierarchy() prepends it) and end at 3DSSG's root "entity".

CLASS_HIERARCHY: dict[str, list[str]] = {
    # seating
    "chair":        ["seat", "furniture", "artifact", "entity"],
    "armchair":     ["chair", "seat", "furniture", "artifact", "entity"],
    "office_chair": ["chair", "seat", "furniture", "artifact", "entity"],
    "stool":        ["seat", "furniture", "artifact", "entity"],
    "bench":        ["seat", "furniture", "artifact", "entity"],
    "sofa":         ["seat", "furniture", "artifact", "entity"],
    # tables / work surfaces
    "desk":         ["table", "furniture", "artifact", "entity"],
    "table":        ["furniture", "furnishing", "artifact", "entity"],
    "workbench":    ["table", "furniture", "artifact", "entity"],
    "workstation":  ["desk", "table", "furniture", "artifact", "entity"],
    "counter":      ["table", "furniture", "artifact", "entity"],
    # storage
    "cabinet":      ["furniture", "furnishing", "artifact", "entity"],
    "locker":       ["cabinet", "furniture", "artifact", "entity"],
    "drawer":       ["storage_space", "furniture", "artifact", "entity"],
    "shelf":        ["support", "furniture", "artifact", "entity"],
    "bookshelf":    ["shelf", "support", "furniture", "artifact", "entity"],
    "rack":         ["support", "furniture", "artifact", "entity"],
    "storage_rack": ["rack", "support", "furniture", "artifact", "entity"],
    # lighting
    "ceiling_light": ["light", "source_of_illumination", "device", "artifact", "entity"],
    "pendant_lamp":  ["lamp", "source_of_illumination", "device", "artifact", "entity"],
    "wall_light":    ["light", "source_of_illumination", "device", "artifact", "entity"],
    "desk_lamp":     ["lamp", "source_of_illumination", "device", "artifact", "entity"],
    "floor_lamp":    ["lamp", "source_of_illumination", "device", "artifact", "entity"],
    # screens / electronics
    "monitor":          ["electronic_device", "device", "artifact", "entity"],
    "computer_monitor": ["monitor", "electronic_device", "device", "artifact", "entity"],
    "tv_screen":        ["electronic_device", "device", "artifact", "entity"],
    "laptop":           ["computer", "electronic_device", "device", "artifact", "entity"],
    "keyboard":         ["electronic_device", "device", "artifact", "entity"],
    "mouse":            ["electronic_device", "device", "artifact", "entity"],
    "projector":        ["optical_device", "device", "artifact", "entity"],
    "printer":          ["machine", "device", "artifact", "entity"],
    "phone":            ["electronic_device", "device", "artifact", "entity"],
    "tablet":           ["electronic_device", "device", "artifact", "entity"],
    # industrial / factory
    "machine":          ["device", "artifact", "entity"],
    "conveyor":         ["machine", "device", "artifact", "entity"],
    "robot_arm":        ["machine", "device", "artifact", "entity"],
    "control_panel":    ["panel", "device", "artifact", "entity"],
    "electrical_panel": ["panel", "device", "artifact", "entity"],
    "pipe":             ["conduit", "passage", "artifact", "entity"],
    "duct":             ["conduit", "passage", "artifact", "entity"],
    "air_duct":         ["duct", "conduit", "passage", "artifact", "entity"],
    "column":           ["structural_member", "support", "artifact", "entity"],
    "pillar":           ["structural_member", "support", "artifact", "entity"],
    "support_beam":     ["structural_member", "support", "artifact", "entity"],
    "pallet":           ["container", "instrumentality", "artifact", "entity"],
    "forklift":         ["vehicle", "conveyance", "artifact", "entity"],
    "cart":             ["wheeled_vehicle", "conveyance", "artifact", "entity"],
    "trolley":          ["wheeled_vehicle", "conveyance", "artifact", "entity"],
    "ladder":           ["framework", "artifact", "entity"],
    # safety
    "fire_extinguisher": ["device", "artifact", "entity"],
    "exit_sign":         ["sign", "communication", "artifact", "entity"],
    "safety_barrier":    ["barrier", "obstruction", "artifact", "entity"],
    "first_aid":         ["equipment", "artifact", "entity"],
    # plants / decor
    "plant":       ["organism", "living_thing", "entity"],
    "person":      ["organism", "living_thing", "entity"],
    "painting":    ["art", "creation", "artifact", "entity"],
    "whiteboard":  ["board", "sheet", "artifact", "entity"],
    "blackboard":  ["board", "sheet", "artifact", "entity"],
    # containers
    "trash_bin":     ["bin", "container", "artifact", "entity"],
    "box":           ["container", "instrumentality", "artifact", "entity"],
    "cardboard_box": ["box", "container", "artifact", "entity"],
    "crate":         ["box", "container", "artifact", "entity"],
    "barrel":        ["vessel", "container", "artifact", "entity"],
    "bag":           ["container", "instrumentality", "artifact", "entity"],
    "bottle":        ["vessel", "container", "artifact", "entity"],
    "cup":           ["vessel", "container", "artifact", "entity"],
    # architecture
    "door":            ["movable_barrier", "barrier", "artifact", "entity"],
    "window":          ["framework", "supporting_structure", "artifact", "entity"],
    "wall":            ["partition", "structure", "artifact", "entity"],
    "partition_panel": ["partition", "structure", "artifact", "entity"],
    "stairs":          ["stairway", "structure", "artifact", "entity"],
    "railing":         ["barrier", "structure", "artifact", "entity"],
    # misc
    "clock":  ["timepiece", "instrument", "device", "artifact", "entity"],
}


def class_hierarchy(label: str) -> list[str]:
    """Full hierarchy c = (c1, ..., cd) for a label — the label itself
    followed by its hypernym chain, ending at "entity" (3DSSG §3.1)."""
    if label in CLASS_HIERARCHY:
        return [label] + CLASS_HIERARCHY[label]
    return [label, "object", "entity"]


# Affordances — interaction possibilities per class (3DSSG §3.2), verbs in
# 3DSSG's own affordance vocabulary (placing, sitting, opening, ...).
# State-dependent entries are filtered by affordances_for(): only a closed
# door affords opening, only an open one affords closing.
AFFORDANCES: dict[str, list[str]] = {
    "chair": ["sitting", "moving"], "armchair": ["sitting", "moving"],
    "office_chair": ["sitting", "moving"], "stool": ["sitting", "moving"],
    "bench": ["sitting"], "sofa": ["sitting", "sleeping"],
    "desk": ["placing", "working"], "table": ["placing"],
    "workbench": ["placing", "working"], "workstation": ["placing", "working"],
    "counter": ["placing"],
    "cabinet": ["storing", "opening", "closing"],
    "locker": ["storing", "opening", "closing"],
    "drawer": ["storing", "opening", "closing"],
    "shelf": ["storing", "placing"], "bookshelf": ["storing", "placing"],
    "rack": ["storing", "placing"], "storage_rack": ["storing", "placing"],
    "ceiling_light": ["lighting", "switching"],
    "pendant_lamp": ["lighting", "switching"],
    "wall_light": ["lighting", "switching"],
    "desk_lamp": ["lighting", "switching"],
    "floor_lamp": ["lighting", "switching"],
    "monitor": ["watching", "working", "switching"],
    "computer_monitor": ["watching", "working", "switching"],
    "tv_screen": ["watching", "switching"],
    "laptop": ["working", "carrying", "opening", "closing"],
    "keyboard": ["writing"], "mouse": ["holding"],
    "projector": ["watching", "switching"], "printer": ["printing"],
    "phone": ["calling", "carrying"], "tablet": ["working", "carrying"],
    "machine": ["working", "switching"], "conveyor": ["moving", "switching"],
    "robot_arm": ["working", "switching"],
    "control_panel": ["working", "switching"],
    "electrical_panel": ["opening", "closing", "switching"],
    "pallet": ["carrying", "placing"], "forklift": ["riding", "moving"],
    "cart": ["moving", "carrying"], "trolley": ["moving", "carrying"],
    "ladder": ["climbing", "leaning", "carrying"],
    "fire_extinguisher": ["carrying", "holding"],
    "safety_barrier": ["moving"],
    "plant": ["watering", "decorating"],
    "painting": ["looking", "decorating"],
    "whiteboard": ["writing", "cleaning", "moving"],
    "blackboard": ["writing", "cleaning"],
    "trash_bin": ["throwing", "moving"],
    "box": ["carrying", "storing", "opening"],
    "cardboard_box": ["carrying", "storing", "opening"],
    "crate": ["carrying", "storing"], "barrel": ["storing"],
    "bag": ["carrying", "storing"],
    "bottle": ["drinking", "holding"], "cup": ["drinking", "holding"],
    "door": ["opening", "closing", "walking"],
    "window": ["opening", "closing", "looking"],
    "stairs": ["climbing", "walking"], "railing": ["holding", "leaning"],
    "clock": ["looking"],
}

# State-conditioned affordance pairs: (affordance, state that BLOCKS it) —
# 3DSSG: "only a closed door can be opened".
_STATE_BLOCKS = {"opening": "open", "closing": "closed",
                 "switching": None}   # switching is state-agnostic


def affordances_for(label: str, state: str | None = None) -> list[str]:
    out = []
    for aff in AFFORDANCES.get(label, []):
        blocked_by = _STATE_BLOCKS.get(aff)
        if state and blocked_by and state == blocked_by:
            continue
        out.append(aff)
    return out


# Non-rigid classes (3DSSG static physical property "rigidity").
_NONRIGID = frozenset(["bag", "plant", "person", "curtain"])

# Named color anchors for the point-cloud mean-color attribute. Matching is
# done in HSV: low saturation resolves to white/gray/black by value, low
# value to black, otherwise nearest hue.
_HUE_NAMES = [(15, "red"), (45, "orange"), (70, "yellow"), (160, "green"),
              (200, "cyan"), (260, "blue"), (300, "purple"), (335, "pink"),
              (360, "red")]


def _color_name(rgb) -> str:
    r, g, b = [float(v) / 255.0 for v in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    v = mx
    s = 0.0 if mx == 0 else (mx - mn) / mx
    if v < 0.16:
        return "black"
    if s < 0.16:
        return "white" if v > 0.72 else "gray"
    d = mx - mn
    if mx == r:
        h = (60 * ((g - b) / d) + 360) % 360
    elif mx == g:
        h = 60 * ((b - r) / d) + 120
    else:
        h = 60 * ((r - g) / d) + 240
    # brown = dark/desaturated orange
    if 10 <= h <= 50 and v < 0.55:
        return "brown"
    for hi, name in _HUE_NAMES:
        if h <= hi:
            return name
    return "red"


def _shape_name(bbox_size) -> str:
    """Bbox-aspect fallback shape, used only when the node's own points are
    unavailable — the bbox can't distinguish a round object from a boxy one
    of the same extents, so _point_stats below is always preferred."""
    w, h, d = [max(float(v), 1e-6) for v in bbox_size]
    horiz_max, horiz_min = max(w, d), min(w, d)
    if h < 0.35 * horiz_max:
        return "flat"
    if h > 2.2 * horiz_max:
        return "tall"
    if horiz_max > 2.5 * horiz_min:
        return "elongated"
    if max(w, h, d) < 1.6 * min(w, h, d):
        return "cubic"
    return "boxy"


def _point_stats(pts) -> tuple[float, str] | None:
    """(occupied_volume_m3, shape) measured from a node's OWN points.

    Volume is the count of occupied voxels × voxel volume — the MATERIAL the
    object is actually made of, not its bounding-box hull. The distinction
    matters for size comparisons: a chair's bbox is mostly air (thin legs,
    open back), so bbox volume calls a chair "bigger" than a solid cabinet
    with a slightly smaller box; voxel occupancy gets the order right. The
    voxel edge adapts to the node's sampling density (min 5 cm) so sparsely
    sampled nodes don't undercount.

    Shape comes from the PCA eigenvalue profile of the point distribution
    (the linearity/planarity/sphericity features of Weinmann et al. 2015):
    round objects score high sphericity, panels high planarity, and a
    standing person is a vertical linear cluster → "tall" — none of which a
    bounding-box aspect ratio can see (everything with similar extents
    reads "boxy"/"cubic" there).
    """
    if pts is None or len(pts) < 12:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    bbox = np.maximum(pts.max(0) - pts.min(0), 1e-6)
    vox = max(0.05, float(np.cbrt(float(np.prod(bbox)) / len(pts))))
    cells = np.unique(np.floor(pts / vox).astype(np.int64), axis=0)
    occ_vol = float(len(cells)) * vox ** 3

    centered = pts - pts.mean(0)
    cov = centered.T @ centered / len(pts)
    evals, evecs = np.linalg.eigh(cov)
    s3, s2, s1 = np.sqrt(np.maximum(evals, 1e-12))   # ascending std-devs
    linearity  = (s1 - s2) / s1
    planarity  = (s2 - s3) / s1
    sphericity = s3 / s1
    main_axis  = evecs[:, 2]
    if sphericity > 0.50:
        shape = "round"
    elif planarity > 0.55:
        shape = "flat"
    elif linearity > 0.60:
        shape = "tall" if abs(float(main_axis[1])) > 0.7 else "elongated"
    else:
        shape = "boxy"
    return occ_vol, shape


def _eff_volume(obj: dict) -> float:
    """Effective object volume for size comparisons: point-cloud occupied
    volume when measured (see _point_stats), bbox volume otherwise."""
    v = obj.get("occ_volume")
    return float(v) if v else float(obj.get("volume", 1.0))


def build_attributes(objects: dict) -> dict:
    """Attach 3DSSG-style semantics to every object:
      obj["class_hierarchy"] — label + hypernym chain (§3.1)
      obj["attributes"]      — static properties (§3.2): color (point-cloud
                               mean RGB, when available), shape and absolute
                               size class (bbox), relative size vs other
                               objects of the SAME category (3DSSG:
                               "geometric data and class labels are utilized
                               to identify the relative size of the object in
                               comparison with other objects of the same
                               category"), rigidity, plus material/state when
                               the CLIP stage provided them
      obj["affordances"]     — class-level, state-conditioned (§3.2)
    """
    # Point-derived stats first (pipeline-1 objects carry their raw points
    # as bbox_pts; pipeline 2 precomputes occ_volume/point_shape from the
    # points sidecar in geo_to_scenegraph.py before calling this).
    for obj in objects.values():
        if obj.get("occ_volume") is None and obj.get("bbox_pts") is not None:
            st = _point_stats(obj["bbox_pts"])
            if st is not None:
                obj["occ_volume"], obj["point_shape"] = st

    # volume statistics per label, for the relative-size comparison —
    # effective (material) volume, not the bounding-box hull
    vols_by_label: dict = defaultdict(list)
    for obj in objects.values():
        lbl = obj.get("label", "")
        if lbl and not lbl.startswith("obj_"):
            vols_by_label[lbl].append(_eff_volume(obj))
    median_vol = {lbl: float(np.median(v)) for lbl, v in vols_by_label.items()
                  if len(v) >= 3}

    n_color = n_material = n_ptshape = 0
    for obj in objects.values():
        lbl   = obj.get("label", "")
        vol   = _eff_volume(obj)
        bsize = obj.get("bbox_size", [1.0, 1.0, 1.0])
        attrs: dict = {}

        if obj.get("point_shape"):
            attrs["shape"] = obj["point_shape"]
            n_ptshape += 1
        else:
            attrs["shape"] = _shape_name(bsize)
        # material-volume thresholds (occupied voxels, not bbox hull)
        attrs["size"]  = ("small" if vol < 0.03 else
                          "large" if vol > 0.30 else "medium")
        if lbl in median_vol:
            m = median_vol[lbl]
            if vol <= 0.5 * m:
                attrs["relative_size"] = f"smaller_than_typical_{lbl}"
            elif vol >= 2.0 * m:
                attrs["relative_size"] = f"larger_than_typical_{lbl}"
            else:
                attrs["relative_size"] = f"typical_{lbl}_size"
        attrs["rigidity"] = "nonrigid" if lbl in _NONRIGID else "rigid"

        if obj.get("mean_rgb") is not None:
            attrs["color"] = _color_name(obj["mean_rgb"])
            n_color += 1
        if obj.get("material"):
            attrs["material"] = obj["material"]
            n_material += 1
        state = obj.get("state")
        if state:
            attrs["state"] = state

        obj["attributes"]      = attrs
        obj["class_hierarchy"] = class_hierarchy(lbl)
        obj["affordances"]     = affordances_for(lbl, state)

    print(f"[sg] attributes: {len(objects)} objects "
          f"({n_color} with point-cloud color, {n_material} with CLIP material, "
          f"{n_ptshape} with point-derived shape/volume)")
    return objects


# ══ 5. Build edges — 3DSSG-style three-tier relationships ════════════════
#
# Edge structure follows 3DSSG (Wald, Dhamo, Navab & Tombari, "Learning 3D
# Semantic Scene Graphs from 3D Indoor Reconstructions", CVPR 2020, §3.3),
# which classifies scene-graph relationships into exactly three tiers:
#
#   a) SUPPORT relationships — the scene's supporting structure: what each
#      object physically rests on / hangs from / sits inside. Every object
#      gets at most one support parent here; objects with no detected
#      parent are implicitly supported by the floor (3DSSG: "the floor is
#      the only instance that, by definition, does not have any support").
#      Relations: standing_on, lying_on, hanging_on, standing_in, lying_in,
#      supported_by, connected_to.
#
#   b) PROXIMITY relationships — spatial layout (left/right/front/behind/
#      close_by), computed ONLY between objects that share a support
#      parent (3DSSG: "we only compute proximity relationships between the
#      nodes that share a support parent. A bottle on a table therefore
#      has no proximity relationship with a chair" — the bottle's position
#      relative to the chair is derivable from its parent's). Floor-
#      supported objects in the same area are siblings of each other.
#
#   c) COMPARATIVE relationships — derived by comparing object properties:
#      bigger_than/smaller_than (volume), higher_than/lower_than
#      (elevation), same_object_type (equal semantic label). 3DSSG derives
#      these from annotated attributes; here they come from the measured
#      geometry + the CLIP label, which are the attributes we actually have.
#
# Every edge carries an "edge_type" field ("support" | "proximity" |
# "comparative") naming its tier.

COMP_SIZE_RATIO    = 2.0   # min volume ratio for bigger_than/smaller_than —
                           # 2x so ordinary size jitter between two same-ish
                           # objects doesn't read as a size relation
COMP_HEIGHT_DIFF_M = 0.40  # min centroid elevation difference for
                           # higher_than/lower_than (matches the old
                           # above/below threshold these replace)


def build_edges(objects: dict,
                above_delta: float,
                hanging_delta: float,
                max_direct_gap: float,
                max_hang_gap: float,
                footprint_iou_thr: float,
                on_floor_m: float,
                wall_blocker: dict | None = None,
                yaw_deg: float = 0.0,
                **kwargs) -> list[dict]:
    """
    Build directed relational edges in 3DSSG's three tiers (see the block
    comment above): support first (also fixing each object's support
    parent), then proximity between support-siblings only, then
    comparative. Geometric tests use the 11-dim descriptor g_ij from Heo
    et al. (Eq. 5); directional relations are decided in the building's
    wall-aligned frame (rotated by yaw_deg) so "left of" tracks the room's
    actual walls rather than the raw world axes.

    Support-tier detection per pair (first match wins):
      standing_in / lying_in : A's centroid within B's padded bbox
                               (in = smaller object; lying = flat aspect)
      standing_on / lying_on : A's bottom face rests at B's top face with
                               real footprint overlap
      supported_by           : same rest test, but the lower object is a
                               structural element (pillar etc.)
      hanging_on             : A high above B with footprint overlap and a
                               hanging-plausible label (lamp, duct, ...)
      connected_to           : very close (<0.45 m) at the same height

    Proximity tier (only between objects sharing a support parent):
      left / right / front / behind : dominant wall-aligned axis
      close_by                      : within CLOSE_BY_M, no dominant axis

    Comparative tier (between siblings; same_object_type also between any
    same-label pair that passes the shared suppression checks):
      bigger_than, higher_than, same_object_type

    Every asymmetric relation is stored as ONE canonical directed edge
    (bigger_than from the bigger object, left from the left one, ...);
    the reciprocal (smaller_than, right, ...) is derived by consumers,
    never stored — see the comment at the proximity tier.
    """
    from scipy.spatial import cKDTree

    oids    = sorted(objects.keys())
    if not oids: return []
    cents   = np.array([objects[o]["centroid"] for o in oids], dtype=np.float64)
    floor_y = float(np.percentile(cents[:, 1], 5))

    for i, oid in enumerate(oids):
        objects[oid]["on_floor"] = bool(
            abs(float(cents[i, 1]) - floor_y) < on_floor_m)

    search_r = max(max_hang_gap * 1.3, CLOSE_BY_M * 1.1)
    tree     = cKDTree(cents)
    pairs    = tree.query_pairs(r=search_r, output_type="ndarray")
    print(f"[sg] KD-tree: {len(pairs)} candidate pairs (r={search_r:.2f} m, "
          f"N={len(oids)})")

    edges: list[dict] = []
    seen: set[frozenset] = set()   # pairs already holding a support-tier edge
    support_parent: dict = {}      # child oid → its support parent oid
    n_cross_room = n_cross_area = n_wall_blocked = 0

    # ── Shared pair pre-filter (applies to every tier) ────────────────────
    # 3DSSG computes all three relationship tiers over the same candidate
    # object set; here that set is the KD-tree pairs minus the pairs no
    # relation may ever cross: too-large vertical gaps, excluded labels,
    # different rooms, a confirmed wall between them, different areas.
    admissible: list[tuple] = []
    for i, j in pairs:
        oa, ob = oids[i], oids[j]
        ca, cb = cents[i], cents[j]
        if abs(float(ca[1]) - float(cb[1])) > MAX_EDGE_Y_GAP_M:
            continue
        obj_a, obj_b = objects[oa], objects[ob]
        if (obj_a.get("label", "") in NO_EDGE_LABELS
                or obj_b.get("label", "") in NO_EDGE_LABELS):
            continue
        room_a, room_b = obj_a.get("room_id", -1), obj_b.get("room_id", -1)
        if room_a >= 0 and room_b >= 0 and room_a != room_b:
            n_cross_room += 1
            continue
        if wall_blocker is not None and _segment_crosses_wall(ca, cb, wall_blocker):
            n_wall_blocked += 1
            continue
        area_a, area_b = obj_a.get("area_id", -1), obj_b.get("area_id", -1)
        if area_a >= 0 and area_b >= 0 and area_a != area_b:
            n_cross_area += 1
            continue
        admissible.append((oa, ob, ca, cb, obj_a, obj_b))

    if n_cross_room or n_cross_area or n_wall_blocked:
        print(f"[sg] suppressed {n_cross_room} cross-room + {n_cross_area} "
              f"cross-area + {n_wall_blocked} wall-crossing candidate pairs")

    def _pair_geo(obj_a, obj_b):
        g = _geo_descriptor(obj_a, obj_b)   # shape (11,)
        dx, dy, dz = float(g[0]), float(g[1]), float(g[2])
        d3d     = float(np.linalg.norm(g[:3]))
        horiz_d = float(np.linalg.norm([dx, dz]))
        return dx, dy, dz, d3d, horiz_d

    def _is_flat(obj) -> bool:
        b = obj.get("bbox_size", [1., 1., 1.])
        return max(float(b[0]), float(b[2])) / max(float(b[1]), 1e-6) > 2.5

    # ══ Tier a: SUPPORT — one pass fixing every support parent ═══════════
    for oa, ob, ca, cb, obj_a, obj_b in admissible:
        dx, dy, dz, d3d, horiz_d = _pair_geo(obj_a, obj_b)
        gap = abs(dy)
        iou = _xz_iou(ca, cb)
        pk  = frozenset([oa, ob])

        # containment → standing_in / lying_in (3DSSG's "standing in" /
        # "lying in"): the smaller object's centroid falls within the
        # larger object's padded bbox.
        b_b = np.array(obj_b.get("bbox_size", [2., 2., 2.]), dtype=np.float32)
        b_a = np.array(obj_a.get("bbox_size", [2., 2., 2.]), dtype=np.float32)
        vol_a = float(obj_a.get("volume", 1.0))
        vol_b = float(obj_b.get("volume", 1.0))

        in_b = (abs(dx) < b_b[0] * INSIDE_SCALE and
                abs(dy) < b_b[1] * INSIDE_SCALE and
                abs(dz) < b_b[2] * INSIDE_SCALE and
                d3d > 0.10)
        in_a = (abs(dx) < b_a[0] * INSIDE_SCALE and
                abs(dy) < b_a[1] * INSIDE_SCALE and
                abs(dz) < b_a[2] * INSIDE_SCALE and
                d3d > 0.10)

        if in_b and not in_a and vol_a < vol_b:
            w = round(max(1.0 - d3d / max(float(np.linalg.norm(b_b)), 1e-6), 0.0), 4)
            rel = "lying_in" if _is_flat(obj_a) else "standing_in"
            edges.append({"src": oa, "dst": ob, "relation": rel,
                          "weight": w, "edge_type": "support"})
            support_parent[oa] = ob
            seen.add(pk)
            continue
        if in_a and not in_b and vol_b < vol_a:
            w = round(max(1.0 - d3d / max(float(np.linalg.norm(b_a)), 1e-6), 0.0), 4)
            rel = "lying_in" if _is_flat(obj_b) else "standing_in"
            edges.append({"src": ob, "dst": oa, "relation": rel,
                          "weight": w, "edge_type": "support"})
            support_parent[ob] = oa
            seen.add(pk)
            continue

        # Rest test on the REAL boxes: A stands on B iff their actual XZ
        # footprints overlap (containment over the smaller box — see
        # _bbox_footprint_containment) and A's bottom FACE sits at B's top
        # FACE. Centroid-circle overlap misses an object on the corner of
        # a large desk, which then mislabels as a lateral edge.
        fp = _bbox_footprint_containment(obj_a, obj_b)
        if pk not in seen and fp is not None and fp >= footprint_iou_thr:
            a_lo, a_hi_y = float(obj_a["bbox_min"][1]), float(obj_a["bbox_max"][1])
            b_lo, b_hi_y = float(obj_b["bbox_min"][1]), float(obj_b["bbox_max"][1])
            rest_ab = a_lo - b_hi_y          # A's bottom vs B's top
            rest_ba = b_lo - a_hi_y
            hi = lo = None
            if -REST_INTERPEN_M <= rest_ab <= REST_GAP_MAX_M:
                hi, lo = oa, ob
            elif -REST_INTERPEN_M <= rest_ba <= REST_GAP_MAX_M:
                hi, lo = ob, oa
            if hi is not None:
                if objects[lo].get("label", "") in STRUCT_LABELS:
                    # structural element below → child supported_by parent
                    edges.append({"src": hi, "dst": lo,
                                  "relation": "supported_by",
                                  "weight": round(fp, 4),
                                  "edge_type": "support"})
                elif _is_flat(objects[hi]):
                    edges.append({"src": hi, "dst": lo,
                                  "relation": "lying_on",
                                  "weight": round(fp, 4),
                                  "edge_type": "support"})
                else:
                    edges.append({"src": hi, "dst": lo,
                                  "relation": "standing_on",
                                  "weight": round(fp, 4),
                                  "edge_type": "support"})
                support_parent[hi] = lo
                seen.add(pk)

        # hanging_on keeps the centroid-gap formulation (a hanging lamp's
        # box bottom is nowhere near the desk's top by construction), but
        # uses the real footprint overlap when boxes are available.
        if pk not in seen and horiz_d < 1.8 and gap >= above_delta:
            higher = oa if dy > 0 else ob
            lower  = ob if dy > 0 else oa
            overlap = fp if fp is not None else iou
            if overlap >= footprint_iou_thr:
                lbl_hi = objects[higher].get("label", "")
                if hanging_delta <= gap <= max_hang_gap and lbl_hi in _HANGING_LABELS:
                    edges.append({"src": higher, "dst": lower,
                                  "relation": "hanging_on",
                                  "weight": round(overlap, 4),
                                  "edge_type": "support"})
                    support_parent[higher] = lower
                    seen.add(pk)

        if pk not in seen and d3d < 0.45 and gap < above_delta:
            edges.append({"src": oa, "dst": ob,
                          "relation": "connected_to",
                          "weight": round(1.0 - d3d / 0.45, 4),
                          "edge_type": "support"})
            seen.add(pk)

    # Every object's support parent is now fixed; objects with no entry
    # are floor-supported. Two objects are SIBLINGS iff they share a
    # parent — same supporting object, or both on the floor of the same
    # area of the same room (the floor "instance" is per-area, matching
    # the room→area→object hierarchy).
    def _support_key(oid):
        if oid in support_parent:
            return ("obj", support_parent[oid])
        o = objects[oid]
        return ("floor", o.get("room_id", -1), o.get("area_id", -1))

    # ══ Tier b: PROXIMITY — only between support-siblings ════════════════
    n_nonsibling = 0
    for oa, ob, ca, cb, obj_a, obj_b in admissible:
        pk = frozenset([oa, ob])
        if pk in seen:          # support relation wins for the pair
            continue
        if _support_key(oa) != _support_key(ob):
            n_nonsibling += 1
            continue
        dx, dy, dz, d3d, horiz_d = _pair_geo(obj_a, obj_b)
        if d3d > CLOSE_BY_M:
            continue
        # Wall-aligned (u, v) version of dx/dz — rotation preserves length,
        # so horiz_d needs no recompute.
        dx_wall, dz_wall = _rotate_xz_deg(np.array([[dx, dz]]), -yaw_deg)[0] \
            if abs(yaw_deg) > 0.05 else (dx, dz)

        # One directed edge per pair — the reciprocal (B right-of A for an
        # A-left-of-B edge) is DERIVED by consumers, never stored. Storing
        # both directions invited exactly one bug: a renderer that drops or
        # mislabels one of the two lines shows the same relation from both
        # sides. The stored relation always describes SRC relative to DST
        # in the wall-aligned frame (the static topdown overlay's fixed
        # viewpoint); the interactive 3D viewer recomputes left/right/
        # front/behind against its own camera at render time, which is
        # 3DSSG's own rule for directional relations ("must be updated
        # automatically for the new viewpoint").
        if horiz_d > 0.30:
            abs_dx, abs_dz = abs(dx_wall), abs(dz_wall)
            if abs_dx >= DIR_DOMINANCE * horiz_d:
                w   = round(abs_dx / max(horiz_d, 1e-6), 4)
                rel = "right" if dx_wall > 0 else "left"
                edges.append({"src": oa, "dst": ob, "relation": rel,
                              "weight": w, "edge_type": "proximity"})
                seen.add(pk)
            elif abs_dz >= DIR_DOMINANCE * horiz_d:
                w   = round(abs_dz / max(horiz_d, 1e-6), 4)
                rel = "behind" if dz_wall > 0 else "front"
                edges.append({"src": oa, "dst": ob, "relation": rel,
                              "weight": w, "edge_type": "proximity"})
                seen.add(pk)
            else:
                edges.append({"src": oa, "dst": ob, "relation": "close_by",
                              "weight": round(1.0 - d3d / CLOSE_BY_M, 4),
                              "edge_type": "proximity"})
                seen.add(pk)
        else:
            edges.append({"src": oa, "dst": ob, "relation": "close_by",
                          "weight": round(1.0 - d3d / CLOSE_BY_M, 4),
                          "edge_type": "proximity"})
            seen.add(pk)

    if n_nonsibling:
        print(f"[sg] proximity restricted to support-siblings "
              f"(3DSSG rule): {n_nonsibling} non-sibling pairs skipped")

    # ══ Tier c: COMPARATIVE — property comparison ═════════════════════════
    # bigger/smaller + higher/lower between siblings (comparing two objects
    # in different corners of the scene isn't informative — 3DSSG bounds
    # comparatives the same way it bounds proximity); same_object_type
    # between ANY admissible same-label pair, since "these two are the
    # same kind of thing" is meaningful across a whole room.
    for oa, ob, ca, cb, obj_a, obj_b in admissible:
        la = obj_a.get("label", "")
        lb = obj_b.get("label", "")
        siblings = _support_key(oa) == _support_key(ob)
        same_lbl = bool(la) and la == lb and not la.startswith("obj_")
        if not (siblings or same_lbl):
            continue
        if same_lbl:
            edges.append({"src": oa, "dst": ob, "relation": "same_object_type",
                          "weight": 1.0, "edge_type": "comparative"})
        if not siblings:
            continue
        # Comparatives follow the same single-canonical-edge rule: only
        # bigger_than (from the bigger object) and higher_than (from the
        # higher one) are stored; smaller_than / lower_than exist only as
        # derived reciprocals in the consumers. Size compares the objects'
        # MATERIAL volume (point-cloud voxel occupancy via _eff_volume) —
        # a chair's bounding box is bigger than a cabinet's mostly because
        # it is mostly air, and comparing bbox hulls got that order wrong.
        dy    = float(ca[1]) - float(cb[1])
        vol_a = _eff_volume(obj_a)
        vol_b = _eff_volume(obj_b)
        if vol_a >= COMP_SIZE_RATIO * vol_b:
            w = round(min(vol_a / max(vol_b, 1e-6) / 10.0, 1.0), 4)
            edges.append({"src": oa, "dst": ob, "relation": "bigger_than",
                          "weight": w, "edge_type": "comparative"})
        elif vol_b >= COMP_SIZE_RATIO * vol_a:
            w = round(min(vol_b / max(vol_a, 1e-6) / 10.0, 1.0), 4)
            edges.append({"src": ob, "dst": oa, "relation": "bigger_than",
                          "weight": w, "edge_type": "comparative"})
        if abs(dy) >= COMP_HEIGHT_DIFF_M:
            higher = oa if dy > 0 else ob
            lower  = ob if dy > 0 else oa
            w      = round(min(abs(dy) / 2.5, 1.0), 4)
            edges.append({"src": higher, "dst": lower, "relation": "higher_than",
                          "weight": w, "edge_type": "comparative"})

    # ── Per-node edge cap ─────────────────────────────────────────────────
    # Dense spaces produce many directional/comparative edges. Sort by tier
    # priority (support > proximity > close_by > comparative), then walk
    # once keeping only edges where both endpoints are under cap.
    _PRIORITY = {
        "standing_in":0,"lying_in":0,
        "hanging_on":1,"standing_on":1,"lying_on":1,"supported_by":1,"connected_to":1,
        "left":2,"right":2,"front":2,"behind":2,
        "close_by":3,
        "same_object_type":4,"bigger_than":4,"higher_than":4,
    }
    edges.sort(key=lambda e: (_PRIORITY.get(e["relation"], 5), -e["weight"]))
    kept: list[dict] = []
    node_deg: dict[int, int] = defaultdict(int)
    for e in edges:
        if (node_deg[e["src"]] < MAX_EDGES_PER_NODE and
                node_deg[e["dst"]] < MAX_EDGES_PER_NODE):
            kept.append(e)
            node_deg[e["src"]] += 1
            node_deg[e["dst"]] += 1
    n_dropped = len(edges) - len(kept)
    if n_dropped:
        print(f"[sg] edge cap ({MAX_EDGES_PER_NODE}/node): dropped {n_dropped} "
              f"low-priority edges → {len(kept)} kept")
    edges = kept

    rel_counts = Counter(e["relation"] for e in edges)
    print(f"[sg] {len(edges)} edges: " +
          ", ".join(f"{r}={c}" for r, c in sorted(rel_counts.items())))
    return edges


# ══ 6. Serialise ══════════════════════════════════════════════════════════

UID_XZ_QUANT_M = 0.3   # centroid quantization step for the stable uid — coarse
                       # enough that normal run-to-run jitter doesn't change it


def _stable_object_uid(obj: dict) -> str:
    """Deterministic id derived from a coarse spatial fingerprint (room +
    quantized XZ position + label) rather than the raw integer object id,
    which is just this run's dict-iteration order and has no meaning across
    reruns. The goal is a per-object id that's stable across independent
    scene-graph builds of the *same physical space* over time, so the same
    object keeps the same uid unless it actually moved —
    the basis for future added/removed/moved change detection between scans.
    """
    c = obj["centroid"]
    key = (int(obj.get("room_id", -1)),
           round(c[0] / UID_XZ_QUANT_M), round(c[2] / UID_XZ_QUANT_M),
           obj.get("label", ""))
    return hashlib.sha1(repr(key).encode()).hexdigest()[:12]


# ══ 5b. Coarse/fine two-layer grouping ═══════════════════════════════════
#
# Complex furniture (desks/tables, storage racks) plus everything directly
# related to it — objects it supports, chairs pulled up next to it — form
# one COARSE unit ("this workstation"), which the viewer shows collapsed
# by default and can break down into its fine-grained member objects on
# demand. Grouping is derived from the relation edges already computed by
# build_edges, so "related" means exactly what the scene graph says it
# means: a support-tier edge onto the anchor, or a seat with a proximity
# edge to a desk-category anchor. Desks/tables/workbenches count as ONE
# category, so adjacent ones merge into a single coarse unit.

COARSE_DESK_LABELS = frozenset(["desk", "table", "workbench",
                                "workstation", "counter"])
COARSE_RACK_LABELS = frozenset(["rack", "storage_rack", "shelf", "bookshelf"])
COARSE_SEAT_LABELS = frozenset(["chair", "office_chair", "armchair",
                                "stool", "bench"])
_SUPPORT_CHILD_RELS = frozenset(["standing_on", "lying_on", "hanging_on",
                                 "standing_in", "lying_in"])

# find_fragment_groups: which labels can be fragments of ONE over-segmented
# physical object. Applies to EVERY label — any two same-label objects whose
# footprints actually OVERLAP (genuine intersection, not just close/touching)
# are candidates, plus a few labels that are really one semantic family split
# across CLIP's vocabulary (a shelf vs a storage_rack vs a bookshelf are
# visually the same kind of thing, and adjacent slices of one physical unit
# can get labeled inconsistently between those near-synonyms). Operates on
# the FINAL (post-CLIP) label, which is why an otherwise-risky small/discrete
# class like chair or person is safe to include unconditionally — CLIP's
# labels are far more reliable and evenly distributed than a raw detector
# class (see pipeline4/README.md for the dead end hit when this was first
# tried keyed on 3DETR's raw class instead: one class dominated almost every
# detection there, and merging on it chain-glued dozens of genuinely
# distinct nearby objects into one giant box via transitive union-find).
#
# Only genuine overlap counts as "touching" — no gap tolerance. An earlier
# version allowed boxes within a small gap (0.15m) to count as one object,
# which chain-connected distant, genuinely separate furniture (reported
# directly, with a concrete example: two workstation-layer walkway-
# separated table clusters that should never have merged). "I only want to
# merge the boxes that are overlapping" is the literal rule this enforces.
#
# Two follow-on problems, both verified directly against real data, needed
# a middle ground rather than an all-or-nothing fix:
#
# 1. In a busy room, genuine pairwise overlap still chains transitively
#    across many real, distinct tables (A overlaps B, B overlaps C, ... —
#    a whole row) into one implausibly large component. Discarding that
#    component WHOLESALE (the simplest safe response) meant none of those
#    tables merged at all, even pairs that obviously should have — reported
#    directly as "significantly undermerged." Fix: an oversized component
#    is now SPLIT (recursively, at its largest internal centroid gap along
#    whichever axis has more spread) instead of thrown away outright, so
#    real sub-clusters (e.g. one continuous run of touching tables on one
#    side of a walkway) still merge even though the whole chain doesn't.
# 2. Requiring only "any positive overlap" reopened a previously-verified
#    false positive: two separate real desks pushed together with no gap
#    (shinhan_space_p4, nodes 5/37) genuinely DO overlap by this measure
#    (~0.45m x ~0.12m), so they merge as an isolated pair — with nothing
#    else to corroborate that link. A component with 3+ members doesn't
#    have this problem (multiple independent overlapping pairs agreeing IS
#    corroboration a lone pair lacks), so only an ISOLATED 2-member result
#    (whether from the initial union-find or from splitting an oversized
#    one) gets a second, stricter check: overlap-AREA fraction of the
#    smaller box (`FRAG_PAIR_AREA_FRAC`) — 5/37 fails this clearly (~12%),
#    a verified genuine 2-fragment pair passes easily (~47%).
FRAG_MERGE_FAMILIES = [COARSE_DESK_LABELS, COARSE_RACK_LABELS, COARSE_SEAT_LABELS]
FRAG_Z_OVERLAP_FRAC = 0.30  # min fraction of the shorter box's height overlapping
FRAG_MAX_MEMBERS = 8        # triggers an attempt to split — NOT a hard cap
                            # (see FRAG_MAX_DIAG_REJECT_M: a group that can't
                            # be split further is kept whole regardless of
                            # member count, only an extreme diagonal rejects)
FRAG_MAX_DIAG_M = 6.0       # ditto — triggers a split attempt past this
FRAG_MAX_DIAG_REJECT_M = 12.0  # the ONLY final-rejection criterion, after
                               # every splittable gap has already been cut:
                               # a group this size or smaller is accepted as
                               # ONE merged object no matter how many
                               # members it has — member count alone is no
                               # longer a reason to throw a merge away.
                               # Verified directly why this matters: a real,
                               # genuinely continuous dense cluster (many
                               # overlapping detections around one work
                               # table, factory_space_13) had NO internal
                               # gap above ~0.36m anywhere, so it could never
                               # split down to <= FRAG_MAX_MEMBERS, and the
                               # old "still oversized -> reject" rule threw
                               # the whole real merge away — reported
                               # directly, with a screenshot, as still
                               # massively under-merged. "Merge the
                               # overlapping boxes into one big object, only
                               # break down for really large gaps" is the
                               # literal rule this enforces.
FRAG_PAIR_AREA_FRAC = 0.30  # stricter 2nd gate for an ISOLATED pair only
FRAG_SPLIT_MIN_GAP_M = 0.4  # an oversized component only splits at a gap
                            # at least this wide between neighboring
                            # centroids — smaller gaps are just normal
                            # spacing between touching pieces of one object.
                            # Left unchanged from the prior round: verified
                            # this exact value (not larger) is what
                            # correctly separates two real, distinct
                            # clusters in the same room (the true
                            # separating gap measured 0.405m) — raising it
                            # to chase "really large" would have re-merged
                            # them back together, the opposite of what was
                            # asked for there.


def _frag_family(label: str):
    if not label:
        return None
    for fam in FRAG_MERGE_FAMILIES:
        if label in fam:
            return fam
    return frozenset([label])


def _oversized(objects: dict, members: list, max_members: int, max_diag_m: float) -> bool:
    """Whether to ATTEMPT splitting a component — not a final rejection
    test. See `_still_too_large` for the (much more permissive) final
    acceptance check applied after splitting has already been tried."""
    bmin = np.min([objects[m]["bbox_min"] for m in members], axis=0)
    bmax = np.max([objects[m]["bbox_max"] for m in members], axis=0)
    return len(members) > max_members or float(np.linalg.norm(bmax - bmin)) > max_diag_m


def _still_too_large(objects: dict, members: list,
                     max_diag_reject_m: float = FRAG_MAX_DIAG_REJECT_M) -> bool:
    """Final rejection check AFTER splitting has already been attempted —
    deliberately diagonal-only, no member-count limit: a group that
    couldn't be split at any real gap is one genuinely continuous merge,
    however many detector fragments it's made of."""
    bmin = np.min([objects[m]["bbox_min"] for m in members], axis=0)
    bmax = np.max([objects[m]["bbox_max"] for m in members], axis=0)
    return float(np.linalg.norm(bmax - bmin)) > max_diag_reject_m


def _split_oversized(objects: dict, members: list, max_members: int, max_diag_m: float,
                     min_gap_m: float, depth: int = 0) -> list:
    """Recursively break an oversized component at its largest internal
    centroid gap (whichever of X/Z has more spread) until every piece fits
    the size caps or no gap wide enough to be a real separation remains.
    Depth-capped at 8 (real rooms never need more; guards against
    pathological recursion). Returns a list of member-id lists, each
    length >= 1 — a piece that still can't be split just comes back
    unchanged as the last resort."""
    if len(members) < 2 or not _oversized(objects, members, max_members, max_diag_m) or depth >= 8:
        return [members]
    cents = np.array([objects[m]["centroid"] for m in members])
    ext = np.array([cents[:, 0].ptp(), cents[:, 2].ptp()])
    ax = 0 if ext[0] >= ext[1] else 2
    order = np.argsort(cents[:, ax])
    sorted_members = [members[i] for i in order]
    sorted_vals = cents[order, ax]
    gaps = np.diff(sorted_vals)
    if len(gaps) == 0 or gaps.max() < min_gap_m:
        return [members]   # no real separation found — last resort, still oversized
    split_at = int(np.argmax(gaps)) + 1
    left, right = sorted_members[:split_at], sorted_members[split_at:]
    return (_split_oversized(objects, left, max_members, max_diag_m, min_gap_m, depth + 1) +
            _split_oversized(objects, right, max_members, max_diag_m, min_gap_m, depth + 1))


def find_fragment_groups(objects: dict,
                         z_overlap_frac: float = FRAG_Z_OVERLAP_FRAC,
                         max_members: int = FRAG_MAX_MEMBERS,
                         max_diag_m: float = FRAG_MAX_DIAG_M,
                         pair_area_frac: float = FRAG_PAIR_AREA_FRAC,
                         split_min_gap_m: float = FRAG_SPLIT_MIN_GAP_M) -> list:
    """Detect same-family objects whose footprints actually OVERLAP that are
    really fragments of one physical object a detector split into several
    adjacent boxes, rather than distinct objects placed near each other.
    General-purpose: operates only on each object's final `label` and
    world-frame `bbox_min`/`bbox_max` (+Y up — horizontal = x,z; vertical =
    y), so it works for any upstream pipeline's output, not just pipeline4.

    Union-find over same-family pairs whose XZ footprints genuinely
    intersect (not merely close/touching) and whose vertical extent
    overlaps by >= `z_overlap_frac` of the shorter box's height. A
    component that grows past the size cap is SPLIT at its largest
    internal centroid gap (see `_split_oversized`) rather than discarded —
    a chain of many genuinely touching objects still yields real
    sub-clusters this way instead of losing everything. Any resulting
    ISOLATED (2-member) group additionally needs `pair_area_frac` overlap
    (both axes at once) to survive — a lone pair has no third fragment
    corroborating it, so a merely-technical sliver of overlap isn't
    enough (see module comment above for the verified false-positive this
    guards against). Returns a list of {"id", "member_ids"} dicts, ready
    to pass as `frag_groups` to serialise()/build_coarse_groups()."""
    oids = [oid for oid, o in objects.items()
           if _frag_family(o.get("label", "")) is not None]
    fam_of = {oid: _frag_family(objects[oid]["label"]) for oid in oids}
    parent = {oid: oid for oid in oids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(oids)):
        oi = oids[i]
        bi_lo = np.asarray(objects[oi]["bbox_min"])
        bi_hi = np.asarray(objects[oi]["bbox_max"])
        for j in range(i + 1, len(oids)):
            oj = oids[j]
            if fam_of[oi] != fam_of[oj]:
                continue
            bj_lo = np.asarray(objects[oj]["bbox_min"])
            bj_hi = np.asarray(objects[oj]["bbox_max"])
            ox = min(bi_hi[0], bj_hi[0]) - max(bi_lo[0], bj_lo[0])
            oz = min(bi_hi[2], bj_hi[2]) - max(bi_lo[2], bj_lo[2])
            if ox <= 0.0 or oz <= 0.0:
                continue   # footprints don't actually intersect — no merge
            lo = max(bi_lo[1], bj_lo[1])
            hi = min(bi_hi[1], bj_hi[1])
            overlap = max(hi - lo, 0.0)
            h_i = bi_hi[1] - bi_lo[1]
            h_j = bj_hi[1] - bj_lo[1]
            if overlap < z_overlap_frac * min(h_i, h_j):
                continue
            union(oi, oj)

    comps: dict = {}
    for oid in oids:
        comps.setdefault(find(oid), []).append(oid)

    raw_groups: list = []
    n_split = 0
    for members in comps.values():
        if len(members) < 2:
            continue
        if _oversized(objects, members, max_members, max_diag_m):
            pieces = _split_oversized(objects, members, max_members, max_diag_m, split_min_gap_m)
            if len(pieces) > 1:
                n_split += 1
            raw_groups.extend(p for p in pieces if len(p) >= 2)
        else:
            raw_groups.append(members)

    groups: list = []
    n_rejected = 0
    n_pair_rejected = 0
    for members in raw_groups:
        if _still_too_large(objects, members):
            n_rejected += 1
            continue
        if len(members) == 2:
            # Isolated pair, no third fragment corroborating it — needs
            # the stricter area-fraction check (see module comment).
            a, b = members
            bi_lo = np.asarray(objects[a]["bbox_min"]); bi_hi = np.asarray(objects[a]["bbox_max"])
            bj_lo = np.asarray(objects[b]["bbox_min"]); bj_hi = np.asarray(objects[b]["bbox_max"])
            ox = min(bi_hi[0], bj_hi[0]) - max(bi_lo[0], bj_lo[0])
            oz = min(bi_hi[2], bj_hi[2]) - max(bi_lo[2], bj_lo[2])
            area_i = (bi_hi[0] - bi_lo[0]) * (bi_hi[2] - bi_lo[2])
            area_j = (bj_hi[0] - bj_lo[0]) * (bj_hi[2] - bj_lo[2])
            area_frac = (ox * oz) / min(area_i, area_j) if ox > 0 and oz > 0 else 0.0
            if area_frac < pair_area_frac:
                n_pair_rejected += 1
                continue
        groups.append({"id": len(groups), "member_ids": sorted(members)})

    if groups or n_rejected or n_split or n_pair_rejected:
        n_mem = sum(len(g["member_ids"]) for g in groups)
        msg = (f"[sg] fragment-merge: {len(groups)} groups covering {n_mem} "
               f"objects (over-segmented by the detector)")
        if n_split:
            msg += f"; split {n_split} oversized components"
        if n_rejected:
            msg += (f"; rejected {n_rejected} sub-pieces still spanning "
                    f">{FRAG_MAX_DIAG_REJECT_M}m even after splitting")
        if n_pair_rejected:
            msg += (f"; rejected {n_pair_rejected} isolated pairs "
                    f"(<{pair_area_frac:.0%} mutual footprint area)")
        print(msg)
    return groups


def build_coarse_objects(objects: dict, frag_groups: list,
                         yaw_deg: float = 0.0) -> tuple[dict, dict]:
    """Collapse each fragment-merge group (find_fragment_groups) into ONE
    object — this becomes the graph's real node from here on (edges, rooms,
    areas all operate on the result), not an overlay on top of the original
    fragments. Objects not in any group pass through unchanged, keeping
    their original id.

    Returns (coarse_objects, fragments): `fragments` maps a new merged id to
    lightweight records of the fine-grained boxes it replaced (id, label,
    box_center, bbox_size, n_world_pts only — NOT the full object dict,
    which carries the raw point cloud) so the viewer can still render them
    for visual drill-down. They are NOT separate graph nodes/edges anymore —
    per-fragment relations don't survive the merge; only the merged object's
    own relations (recomputed fresh by the caller via build_edges on the
    result of this function) do.

    Merged geometry mirrors apply_building_yaw()'s own convention: rotate
    the union of member points into the wall-aligned frame, take the
    trimmed AABB there, rotate the center back. This computes the merged
    box the same way a single detected object's box already is computed —
    NOT by unioning the members' (already wall-aligned) boxes as if they
    were world-axis-aligned, which is the exact bug this file's
    apply_building_yaw was written to avoid in the first place."""
    grouped_oids = {oid for fg in frag_groups for oid in fg["member_ids"]
                    if oid in objects}
    coarse_objects: dict = {}
    fragments: dict = {}
    next_id = (max(objects.keys(), default=-1) + 1)

    for oid, obj in objects.items():
        if oid not in grouped_oids:
            coarse_objects[oid] = obj

    for fg in frag_groups:
        members = [oid for oid in fg["member_ids"] if oid in objects]
        if len(members) < 2:
            continue
        member_objs = [objects[m] for m in members]
        pts_list = [o["bbox_pts"] for o in member_objs
                   if o.get("bbox_pts") is not None and len(o["bbox_pts"])]
        pts = np.concatenate(pts_list, axis=0) if pts_list else None

        if pts is not None and len(pts) >= 3:
            if abs(yaw_deg) >= 0.05:
                xz_local = _rotate_xz_deg(pts[:, [0, 2]], -yaw_deg)
                pts_local = np.column_stack([xz_local[:, 0], pts[:, 1], xz_local[:, 1]])
                bmin, bmax = _trimmed_bbox(pts_local)
                mid = (bmin + bmax) / 2.0
                mid_xz = _rotate_xz_deg(np.array([[mid[0], mid[2]]]), yaw_deg)[0]
                box_center = [float(mid_xz[0]), float(mid[1]), float(mid_xz[1])]
            else:
                bmin, bmax = _trimmed_bbox(pts)
                box_center = ((bmin + bmax) / 2.0).tolist()
            centroid = pts.mean(axis=0).tolist()
        else:
            # No usable points on any member (shouldn't happen for pipeline4
            # output, which always has bbox_pts) — fall back to unioning in
            # the wall-aligned LOCAL frame via each member's own box_center,
            # not raw world axes (that was the exact viewer-side bug fixed
            # earlier: unioning rotated boxes as if they were axis-aligned).
            lo = np.array([1e9, 1e9, 1e9]); hi = np.array([-1e9, -1e9, -1e9])
            for o in member_objs:
                bc = np.asarray(o.get("box_center") or o["centroid"])
                s = np.asarray(o.get("bbox_size", [0.2, 0.2, 0.2]))
                lxz = _rotate_xz_deg(np.array([[bc[0], bc[2]]]), -yaw_deg)[0]
                local = np.array([lxz[0], bc[1], lxz[1]])
                lo = np.minimum(lo, local - s / 2)
                hi = np.maximum(hi, local + s / 2)
            bmin, bmax = lo, hi
            mid = (bmin + bmax) / 2.0
            mid_xz = _rotate_xz_deg(np.array([[mid[0], mid[2]]]), yaw_deg)[0]
            box_center = [float(mid_xz[0]), float(mid[1]), float(mid_xz[1])]
            centroid = box_center

        size = bmax - bmin
        labels = [o.get("label", "") for o in member_objs]
        label = Counter(labels).most_common(1)[0][0] if labels else ""
        # Non-geometric fields inherited from the largest member (most
        # representative single fragment) rather than re-derived —
        # material/state/room/area don't have a meaningful "average".
        largest = max(member_objs, key=lambda o: o.get("n_world_pts", 0))

        gid = next_id
        next_id += 1
        coarse_objects[gid] = {
            "centroid":    centroid,
            "bbox_pts":    pts,
            "bbox_min":    bmin.tolist() if hasattr(bmin, "tolist") else list(bmin),
            "bbox_max":    bmax.tolist() if hasattr(bmax, "tolist") else list(bmax),
            "bbox_size":   size.tolist() if hasattr(size, "tolist") else list(size),
            "box_center":  box_center,
            "sigma":       (size / 3.46).tolist() if hasattr(size, "tolist")
                          else [v / 3.46 for v in size],
            "volume":      float(max(np.prod(np.maximum(size, 1e-6)), 1e-6)),
            "max_side":    float(np.max(size)),
            "n_proposals": sum(int(o.get("n_proposals", 0)) for o in member_objs),
            "n_world_pts": sum(int(o.get("n_world_pts", 0)) for o in member_objs),
            "label":       label,
            "clip_topk":   largest.get("clip_topk", []),
            "mean_rgb":    largest.get("mean_rgb"),
            "material":    largest.get("material"),
            "state":       largest.get("state"),
            "occ_volume":  largest.get("occ_volume"),
            "point_shape": largest.get("point_shape"),
            "room_id":     largest.get("room_id"),
            "area_id":     largest.get("area_id"),
        }
        fragments[gid] = [
            {"id": m, "label": o.get("label"),
             "box_center": o.get("box_center") or o.get("centroid"),
             "bbox_size": o.get("bbox_size"),
             "n_world_pts": o.get("n_world_pts")}
            for m, o in zip(members, member_objs)
        ]

    if fragments:
        print(f"[sg] coarse nodes: {len(fragments)} merged objects replacing "
              f"{sum(len(v) for v in fragments.values())} fine-grained "
              f"fragments ({len(coarse_objects)} total nodes going forward)")
    return coarse_objects, fragments


def build_coarse_groups(objects: dict, edges: list,
                        frag_groups: list | None = None) -> tuple[list, dict]:
    """Build the coarse layer in two independent tiers, highest-priority
    first, so an object is never claimed by both:

      1. Fragment-merge groups (from find_fragment_groups(), passed in as
         `frag_groups` if the caller opted in): boxes already flagged as
         fragments of ONE over-segmented physical object (e.g. a long
         table/shelf that a query-based detector split into several
         adjacent boxes). These are taken as-is — no further geometry
         heuristics — and their members are removed from consideration below.
      2. The existing anchor-based "workstation" grouping (desk/rack
         category + directly related objects — support-tier children, or a
         seat proximity-linked to a desk anchor), run only over whatever
         remains ungrouped.

    Returns (groups, obj_group) where obj_group maps member oid → group id
    (ids are contiguous across both tiers)."""
    groups: list = []
    obj_group: dict = {}
    grouped_oids: set = set()

    if frag_groups:
        for fg in frag_groups:
            members = [m for m in fg["member_ids"] if m in objects]
            if len(members) < 2:
                continue
            cents = np.array([objects[m]["centroid"] for m in members])
            labels = [objects[m].get("label", "") for m in members]
            lbl = Counter(labels).most_common(1)[0][0] if labels else "object"
            gid = len(groups)
            groups.append({
                "id":         gid,
                "kind":       "fragment_merge",
                "label":      f"{lbl}_group",
                "anchor_ids": [],
                "member_ids": sorted(members),
                "n_objects":  len(members),
                "centroid":   [round(float(v), 4) for v in cents.mean(0)],
            })
            for m in members:
                obj_group[m] = gid
                grouped_oids.add(m)
        if groups:
            n_mem = sum(g["n_objects"] for g in groups)
            print(f"[sg] fragment-merge layer: {len(groups)} groups covering "
                  f"{n_mem} objects (over-segmented by the detector)")

    anchor_groups, anchor_obj_group = _build_anchor_coarse_groups(
        {oid: o for oid, o in objects.items() if oid not in grouped_oids},
        [e for e in edges if e["src"] not in grouped_oids
         and e["dst"] not in grouped_oids])
    id_offset = len(groups)
    for g in anchor_groups:
        g["id"] += id_offset
        g.setdefault("kind", "workstation")
        groups.append(g)
    for oid, gid in anchor_obj_group.items():
        obj_group[oid] = gid + id_offset
    return groups, obj_group


def _build_anchor_coarse_groups(objects: dict, edges: list,
                                max_members: int = FRAG_MAX_MEMBERS,
                                max_diag_m: float = FRAG_MAX_DIAG_M,
                                split_min_gap_m: float = FRAG_SPLIT_MIN_GAP_M
                                ) -> tuple[list, dict]:
    """Group anchor objects (desk/rack category) with OTHER anchors of the
    SAME category, via a support/proximity edge between them. Returns
    (groups, obj_group) where obj_group maps member oid → group id.

    Same-type only, no exceptions, no per-label hardcoding: a group here
    NEVER contains any object outside the anchor's own category (desk or
    rack) — not a chair, not a person, not an item resting on it. Two
    earlier, narrower attempts both had to be walked back: first a
    "seat with a proximity edge joins the desk" rule (a chair ends up
    part of the same unit as the table it's pulled up to) and separately
    a one-off "except person" patch on top of a generic "any support-
    tier child joins its anchor" rule (a person leaning on/standing near
    a table also joined) — both are cross-category inclusion, which the
    user rejected outright as a general principle, not case by case:
    "only merge objects of the same type." Removed entirely rather than
    hardcode a growing exception list.

    Two same-category anchors merge into one workstation via ANY
    support/proximity edge between them (a row of desks is one
    workstation) — same as the true original design. That alone
    over-merges in a busy room (verified directly, factory_space_13: 21
    anchors spanning a walkway glued into one "unit" purely by proximity
    radius, reported as wrong) — but discarding an oversized result
    wholesale (the first fix) just meant nothing merged at all, reported
    right back as "significantly undermerged." Landed on the same
    resolution already proven for find_fragment_groups' oversized
    fragment chains: SPLIT the oversized component at its largest
    internal centroid gap (`_split_oversized`) instead of either merging
    everything or rejecting everything — a real walkway/aisle gap splits
    the group there; a genuinely continuous run of touching desks has no
    such gap and stays one workstation (or gets rejected only if truly
    unsplittable and still oversized)."""
    def _cat(lbl: str) -> str | None:
        if lbl in COARSE_DESK_LABELS:
            return "desk"
        if lbl in COARSE_RACK_LABELS:
            return "rack"
        return None

    anchor_cat = {oid: _cat(o.get("label", "")) for oid, o in objects.items()}
    anchors = {oid for oid, c in anchor_cat.items() if c}
    if not anchors:
        return [], {}

    parent = {oid: oid for oid in anchors}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges:
        s, d = e["src"], e["dst"]
        if (s in anchors and d in anchors
                and anchor_cat[s] == anchor_cat[d]
                and e.get("edge_type") in ("support", "proximity")):
            union(s, d)

    comps: dict = {}
    for a in anchors:
        comps.setdefault(find(a), []).append(a)

    groups: list = []
    obj_group: dict = {}
    n_split = 0
    n_rejected = 0
    for root, members in sorted(comps.items()):
        if len(members) < 2:
            continue                        # lone anchor — nothing to merge
        pieces = [members]
        if _oversized(objects, members, max_members, max_diag_m):
            pieces = _split_oversized(objects, members, max_members,
                                      max_diag_m, split_min_gap_m)
            if len(pieces) > 1:
                n_split += 1
        for piece in pieces:
            if len(piece) < 2:
                continue
            if _still_too_large(objects, piece):
                n_rejected += 1
                continue
            cents = np.array([objects[m]["centroid"] for m in piece])
            gid = len(groups)
            # Name the group after the anchors' actual object label (e.g.
            # "table"), not the internal category constant ("desk" — the
            # name COARSE_DESK_LABELS categorizes table/desk/workbench/
            # workstation/counter under). Individual nodes are already
            # unified to a single canonical label per family
            # (unify_table_labels collapses "desk" -> "table" everywhere)
            # — using the category name here instead of the real label
            # made otherwise-identical objects display as "table"
            # (fragment-merge groups, named from the real label) in one
            # place and "desk" (workstation groups, named from the
            # category) in another, for the exact same underlying label.
            anchor_lbl = Counter(objects[a].get("label", "") for a in piece) \
                .most_common(1)[0][0]
            groups.append({
                "id":         gid,
                "label":      f"{anchor_lbl}_group",
                "anchor_ids": sorted(piece),
                "member_ids": sorted(piece),
                "n_objects":  len(piece),
                "centroid":   [round(float(v), 4) for v in cents.mean(0)],
            })
            for m in piece:
                obj_group[m] = gid
    if groups or n_split or n_rejected:
        n_mem = sum(g["n_objects"] for g in groups)
        msg = (f"[sg] workstation layer: {len(groups)} groups covering "
               f"{n_mem} objects ({len(objects) - n_mem} standalone)")
        if n_split:
            msg += f"; split {n_split} oversized components"
        if n_rejected:
            msg += (f"; rejected {n_rejected} sub-pieces still spanning "
                    f">{FRAG_MAX_DIAG_REJECT_M}m even after splitting")
        print(msg)
    return groups, obj_group


def serialise(objects: dict, edges: list, room_meta: dict, areas: list,
              space_name: str, yaw_deg: float = 0.0,
              frag_groups: list | None = None,
              fine_fragments: dict | None = None) -> dict:
    """
    Serialise the scene graph with a Hydra/IRS-style hierarchy:
        building (1) → rooms (N) → areas (K) → objects (M)
    plus the intra-object relational edges (standing_on, left, higher_than, etc.)
    which live entirely within the "objects" layer.

    JSON structure:
      {
        "space": ...,
        "hierarchy": {
          "building": {"id": "B0", "n_rooms": N},
          "rooms": [{"id": "R0", "area_m2":.., "n_objects":..}, ...],
          "areas": [{"id": "A0", "room_id": "R0", "n_objects":.., ...}, ...],
        },
        "nodes": [...],          # object-layer nodes (unchanged schema + area_id/n_absorbed)
        "edges": [...],          # object-layer relational edges (unchanged)
        "hierarchy_edges": [     # cross-layer edges for Hydra-style viz
          {"src":"B0","dst":"R0","kind":"contains"},
          {"src":"R0","dst":"A0","kind":"contains"},   # room → area
          {"src":"A0","dst":12,  "kind":"contains"},   # area → object id
          ...
        ],
        "rooms": [...],          # unchanged (kept for backward-compat)
        "areas": [...],          # NEW: raw area records (xz_min/xz_max/member_ids)
        "stats": {...},
      }
    """
    # Two genuinely co-located same-label objects (e.g. adjacent ceiling
    # lights) can quantize to the same base uid — tie-break deterministically
    # by oid so every node still gets a distinct uid.
    base_uids: dict[int, str] = {oid: _stable_object_uid(obj) for oid, obj in objects.items()}
    uid_groups: dict[str, list[int]] = defaultdict(list)
    for oid, u in base_uids.items():
        uid_groups[u].append(oid)
    final_uid: dict[int, str] = {}
    for u, oid_list in uid_groups.items():
        if len(oid_list) == 1:
            final_uid[oid_list[0]] = u
        else:
            for k, oid in enumerate(sorted(oid_list)):
                final_uid[oid] = f"{u}-{k}"

    coarse_groups, obj_group = build_coarse_groups(objects, edges, frag_groups)

    nodes = []
    for oid, obj in objects.items():
        nodes.append({
            "id":           int(oid),
            "uid":          final_uid[oid],
            "group_id":     int(obj_group.get(oid, -1)),
            "label":        obj.get("label", f"object_{oid}"),
            "caption":      obj.get("caption", ""),
            "label_score":  round(float(obj.get("label_score",  0.0)), 3),
            "label_entropy": round(float(obj.get("label_entropy", 0.0)), 4),
            "centroid":     [round(v, 4) for v in obj["centroid"]],
            "box_center":   [round(float(v), 4) for v in
                             (obj.get("box_center")
                              or ((np.asarray(obj["bbox_min"])
                                   + np.asarray(obj["bbox_max"])) / 2.0
                                  if obj.get("bbox_min") is not None
                                  and obj.get("bbox_max") is not None
                                  else obj["centroid"]))],
            "bbox_size":    [round(v, 4) for v in obj.get("bbox_size", [0., 0., 0.])],
            "room_id":      int(obj.get("room_id", -1)),
            "area_id":      int(obj.get("area_id", -1)),
            "n_proposals":  int(obj["n_proposals"]),
            "n_world_pts":  int(obj["n_world_pts"]),
            "on_floor":     bool(obj.get("on_floor", False)),
            "n_absorbed":   int(obj.get("n_absorbed", 0)),
            "absorbed_ids": [int(x) for x in obj.get("absorbed_ids", [])],
            # 3DSSG-style node semantics (§3.1–3.2)
            "class_hierarchy": obj.get("class_hierarchy",
                                       class_hierarchy(obj.get("label", ""))),
            "attributes":      obj.get("attributes", {}),
            "affordances":     obj.get("affordances", []),
        })

    room_ids = sorted([k for k in room_meta.keys()
                       if isinstance(k, int) and k >= 0])
    rooms = [
        {"id": int(k),
         "centroid_xz": [round(x,4) for x in room_meta[k]["centroid_xz"]],
         "n_pts": int(room_meta[k]["n_pts"]),
         "area_m2": room_meta[k].get("area_m2", 0.0)}
        for k in room_ids
    ]

    ser_edges = [{"src": int(e["src"]), "dst": int(e["dst"]),
                  "relation": e["relation"], "weight": float(e["weight"]),
                  "edge_type": e.get("edge_type", "")}
                 for e in edges]

    ser_areas = [
        {"id": int(a["id"]), "room_id": int(a["room_id"]),
         "n_objects": int(a["n_objects"]),
         "centroid_xz": [round(v, 4) for v in a["centroid_xz"]],
         "xz_min": [round(v, 4) for v in a["xz_min"]],
         "xz_max": [round(v, 4) for v in a["xz_max"]],
         "member_ids": [int(x) for x in a["member_ids"]]}
        for a in areas
    ]

    def _room_node_id(rid: int) -> str:
        return f"R{rid}" if rid >= 0 else "R_unassigned"

    # ── Hierarchy: building → rooms → areas → objects ───────────────────
    hierarchy_edges = []
    building_id = "B0"
    for rid in room_ids:
        hierarchy_edges.append({"src": building_id, "dst": _room_node_id(rid),
                                "kind": "contains"})
    for a in ser_areas:
        area_node_id = f"A{a['id']}"
        hierarchy_edges.append({"src": _room_node_id(a["room_id"]),
                                "dst": area_node_id, "kind": "contains"})
        for oid in a["member_ids"]:
            hierarchy_edges.append({"src": area_node_id, "dst": oid,
                                    "kind": "contains"})

    hierarchy = {
        "building": {"id": building_id, "n_rooms": len(room_ids)},
        "rooms": [
            {"id": f"R{r['id']}", "area_m2": r["area_m2"],
             "n_objects": sum(a["n_objects"] for a in ser_areas if a["room_id"] == r["id"]),
             "centroid_xz": r["centroid_xz"]}
            for r in rooms
        ],
        "areas": [
            {"id": f"A{a['id']}", "room_id": _room_node_id(a["room_id"]),
             "n_objects": a["n_objects"], "centroid_xz": a["centroid_xz"]}
            for a in ser_areas
        ],
    }

    return {
        "space": space_name,
        "building_yaw_deg": round(float(yaw_deg), 3),
        "hierarchy": hierarchy,
        "hierarchy_edges": hierarchy_edges,
        "nodes": nodes,
        "edges": ser_edges,
        "rooms": rooms,
        "areas": ser_areas,
        "coarse_groups": coarse_groups,
        # Fine-grained fragments a merged node replaced (build_coarse_objects)
        # — visual drill-down only, NOT separate graph nodes/edges. One entry
        # per merged node id that had >=2 original fragments.
        "fragments": [{"coarse_id": gid, "members": members}
                      for gid, members in (fine_fragments or {}).items()],
        "stats": {"n_nodes": len(nodes), "n_edges": len(ser_edges),
                  "n_rooms": len(rooms), "n_areas": len(ser_areas),
                  "n_coarse_groups": len(coarse_groups)},
    }


# ══ 7. Overlay ════════════════════════════════════════════════════════════

def render_overlay(graph: dict, space_name: str):
    """Generate multiple topdown overlay PNGs for different views."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    topdown_dir = REPO / "ui" / "_spaces" / space_name / "topdown"
    img_path    = topdown_dir / "topdown.png"
    bounds_path = topdown_dir / "bounds.json"
    if not img_path.exists() or not bounds_path.exists():
        print("[sg] topdown assets not found — skipping overlay"); return

    import numpy as np
    from PIL import Image as PILImage

    img    = np.array(PILImage.open(img_path))
    bounds = json.loads(bounds_path.read_text())
    W, H   = bounds["width"], bounds["height"]
    ax_u, ax_v   = bounds["axis_u"], bounds["axis_v"]
    u_min, u_max = bounds["u_min"], bounds["u_max"]
    v_min, v_max = bounds["v_min"], bounds["v_max"]

    def w2px(c):
        px = (c[ax_u] - u_min) / max(u_max - u_min, 1e-6) * W
        pf = (c[ax_v] - v_min) / max(v_max - v_min, 1e-6)
        py = (1 - pf) * H if bounds.get("v_flipped") else pf * H
        return float(px), float(py)

    LABEL_COLOR = {
        "chair":"#ff6b6b","desk":"#ffa94d","table":"#ffd43b",
        "monitor":"#69db7c","ceiling_light":"#c0eb75","pendant_lamp":"#a9e34b",
        "cabinet":"#f59f00","shelf":"#fd7e14","rack":"#e67700",
        "whiteboard":"#f8f9fa","machine":"#ff0a54","conveyor":"#c9184a",
        "pallet":"#adb5bd","trash_bin":"#da77f2","box":"#adb5bd",
        "pillar":"#74b816","column":"#5c940d","support_beam":"#5c940d",
        "duct":"#339af0","safety_barrier":"#f03e3e","counter":"#ffe066",
        "workbench":"#ff922b","sofa":"#e599f7","door":"#cc5de8",
    }
    DEF = "#cccccc"

    REL_STYLE = {
        # support tier (3DSSG §3.3a)
        "standing_on":  dict(color="#ff9f43", lw=1.4, ls="-",  alpha=0.80),
        "lying_on":     dict(color="#ffa94d", lw=1.2, ls="-",  alpha=0.75),
        "hanging_on":   dict(color="#ffd43b", lw=1.4, ls="-",  alpha=0.80),
        "connected_to": dict(color="#74c0fc", lw=1.2, ls=":",  alpha=0.70),
        "supported_by": dict(color="#a9e34b", lw=1.4, ls="-",  alpha=0.80),
        "standing_in":  dict(color="#ff6b6b", lw=1.2, ls="-",  alpha=0.70),
        "lying_in":     dict(color="#f03e3e", lw=1.0, ls="-",  alpha=0.65),
        # proximity tier (3DSSG §3.3b — support-siblings only)
        "left":         dict(color="#4dabf7", lw=0.8, ls="-.", alpha=0.55),
        "right":        dict(color="#339af0", lw=0.8, ls="-.", alpha=0.55),
        "front":        dict(color="#63e6be", lw=0.8, ls="-.", alpha=0.55),
        "behind":       dict(color="#20c997", lw=0.8, ls="-.", alpha=0.55),
        "close_by":     dict(color="#868e96", lw=0.6, ls=":",  alpha=0.40),
        # comparative tier (3DSSG §3.3c)
        "higher_than":      dict(color="#da77f2", lw=0.9, ls="--", alpha=0.60),
        "lower_than":       dict(color="#cc5de8", lw=0.9, ls="--", alpha=0.60),
        "bigger_than":      dict(color="#e64980", lw=0.9, ls="--", alpha=0.55),
        "smaller_than":     dict(color="#f783ac", lw=0.9, ls="--", alpha=0.55),
        "same_object_type": dict(color="#94d82d", lw=0.7, ls=":",  alpha=0.45),
    }

    nodes = graph["nodes"]
    edges = graph["edges"]
    node_px = {n["id"]: w2px(n["centroid"]) for n in nodes}

    def _draw(ax, edge_filter=None, node_filter=None, title="",
              show_labels=False, show_rooms=False, show_areas=False,
              color_by_area=False, show_legend=False):
        ax.imshow(img, origin="upper")

        if show_rooms and graph.get("rooms"):
            room_colors = plt.cm.tab10.colors
            for i, room in enumerate(graph["rooms"]):
                col = room_colors[i % len(room_colors)]
                # centroid_xz is [world_x, world_z] — w2px() itself picks
                # out axis_u/axis_v, so this synthetic point must be built
                # in fixed world [x, y, z] order, NOT indexed by ax_u/ax_v
                # (those can be swapped per-space; indexing by them here
                # silently transposes X and Z whenever axis_u != 0).
                pt = [0.0, 0.0, 0.0]
                pt[0] = room["centroid_xz"][0]
                pt[2] = room["centroid_xz"][1]
                rx, ry = w2px(pt)
                ax.text(rx, ry, f"R{room['id']}\n{room.get('area_m2',0):.0f}m²",
                        color="white", fontsize=11, ha="center", va="center",
                        weight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc=col, alpha=0.55, ec="white"))

        area_colors = plt.cm.tab20.colors
        area_color_by_id = {a["id"]: area_colors[i % len(area_colors)]
                            for i, a in enumerate(graph.get("areas", []))}

        if show_areas and graph.get("areas"):
            # Draw each area as a buffer-union "blob" (see _draw_area_blob):
            # the union of a fixed-radius disk around every member point.
            # Every point's own disk always contains that point (nothing
            # can be dropped, unlike the Delaunay-based alpha shape this
            # replaces), and two points farther apart than ~2x the radius
            # never overlap, so the blob naturally leaves a real hole/
            # obstacle uncovered with no hull or triangulation math at all.
            node_xz = {n["id"]: (n["centroid"][0], n["centroid"][2]) for n in nodes}
            blob_radius_m = AREA_MIN_GAP_M * 0.55
            for a in graph["areas"]:
                col = area_color_by_id[a["id"]]
                member_xz = np.array([node_xz[oid] for oid in a["member_ids"]
                                      if oid in node_xz])
                if len(member_xz) == 0:
                    continue
                ctr_world = member_xz.mean(0)
                _draw_area_blob(ax, member_xz, blob_radius_m, bounds, col, 0.16)
                # Same fixed [x, y, z] ordering as the room label above —
                # ctr_world is [world_x, world_z], not [u, v].
                ctr = [0.0, 0.0, 0.0]
                ctr[0], ctr[2] = float(ctr_world[0]), float(ctr_world[1])
                tx, ty = w2px(ctr)
                ax.text(tx, ty, f"A{a['id']}\n{a['n_objects']} obj",
                        color="white", fontsize=6.5, ha="center", va="center",
                        weight="bold", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.22", fc=col,
                                 alpha=0.80, ec="white", lw=0.5))

        # area_id -> obj ids lookup, used only when color_by_area is set
        obj_area_id = {oid: a["id"] for a in graph.get("areas", [])
                       for oid in a["member_ids"]}

        for e in edges:
            if edge_filter and not edge_filter(e): continue
            sp=node_px.get(e["src"]); dp=node_px.get(e["dst"])
            if not sp or not dp: continue
            st=REL_STYLE.get(e["relation"],dict(color="white",lw=0.5,ls="-",alpha=0.2))
            ax.plot([sp[0],dp[0]],[sp[1],dp[1]],
                    color=st["color"],lw=st["lw"],ls=st["ls"],
                    alpha=st["alpha"],zorder=1)
        for n in nodes:
            if node_filter and not node_filter(n): continue
            px,py = node_px[n["id"]]
            if color_by_area and n["id"] in obj_area_id:
                nc = area_color_by_id.get(obj_area_id[n["id"]], DEF)
            else:
                nc = LABEL_COLOR.get(n["label"],DEF)
            sz = 3 + min(n["n_world_pts"]*0.04, 5)
            ax.plot(px,py,"o",color=nc,markersize=sz,
                    markeredgecolor="#00000044",markeredgewidth=0.3,
                    alpha=0.85,zorder=2)
            if show_labels:
                ax.annotate(n["label"], (px,py), xytext=(px+sz+2,py-sz),
                           fontsize=4.5, color="white", alpha=0.9, zorder=3,
                           bbox=dict(boxstyle="round,pad=0.08",
                                    fc="#000000aa", ec="none"))
        if show_legend:
            import matplotlib.lines as mlines
            # only relations that actually occur in this graph's edges —
            # derived reciprocals (smaller_than, ...) are never stored
            present = {e["relation"] for e in edges}
            handles = [
                mlines.Line2D([], [], color=st["color"], lw=max(st["lw"], 1.5),
                              ls=st["ls"], alpha=min(st["alpha"] + 0.15, 1.0),
                              label=rel)
                for rel, st in REL_STYLE.items() if rel in present
            ]
            ax.legend(handles=handles, loc="lower left", fontsize=6.5,
                      framealpha=0.75, facecolor="#0e0e0e", labelcolor="white",
                      edgecolor="#666666", ncol=2, title="edge relation",
                      title_fontsize=7)

        ax.set_title(title,color="white",fontsize=9)
        ax.axis("off")

    def save_single(out_name, edge_f=None, node_f=None, title="",
                    show_labels=False, show_rooms=False, show_areas=False,
                    color_by_area=False, show_legend=False, figsize=(14,14)):
        fig,ax = plt.subplots(figsize=figsize,dpi=120)
        _draw(ax, edge_f, node_f, title, show_labels, show_rooms,
              show_areas, color_by_area, show_legend)
        plt.tight_layout(pad=0)
        out = topdown_dir / out_name
        plt.savefig(str(out), facecolor="#0e0e0e", bbox_inches="tight")
        plt.close(fig)
        print(f"[sg] overlay → {out}")

    n_nodes = graph["stats"]["n_nodes"]
    n_edges = graph["stats"]["n_edges"]
    n_areas = len(graph.get("areas", []))

    # 1. Full overlay (all edges)
    save_single("scene_graph_overlay.png",
                title=f"Full — {n_nodes}n · {n_edges}e",
                show_rooms=True, show_legend=True)
    # 2. Room boundaries only (debug view — verify room detection quality)
    save_single("scene_graph_rooms_only.png",
                edge_f=lambda e: False, node_f=lambda n: False,
                title="Room boundaries", show_rooms=True)
    # 3. Room → Area subdivision — how each room splits into working areas
    # (the KD-style recursive split in build_areas). No edges drawn: this
    # view is specifically about the area boundaries, not object relations.
    save_single("scene_graph_areas.png",
                edge_f=lambda e: False,
                title=f"Room → Area subdivision — {n_areas} areas",
                show_rooms=True, show_areas=True, color_by_area=True,
                figsize=(16, 16))


def render_hydra_diagram(graph: dict, space_name: str):
    """
    Render the 3D building/room/area/object hierarchy diagram, saved as
    scene_graph_3d.png (a Hydra-style [Hughes et al. 2022] layered tree):
        Building (top)
           │
         Rooms
           │
         Areas (working areas within each room — see build_areas)
           │
        Objects (bottom, one node per object, positioned by XZ footprint)
    with object-layer intra-edges (standing_on, similar, etc.) drawn
    within the bottom layer, exactly like Fig. 1 in Hydra's paper.
    """
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("[sg] matplotlib unavailable — skipping Hydra diagram"); return

    nodes  = graph["nodes"]
    edges  = graph["edges"]
    rooms  = graph["rooms"]
    if not rooms:
        print("[sg] no rooms — skipping Hydra diagram"); return

    LABEL_COLOR = {
        "chair":"#ff6b6b","desk":"#ffa94d","table":"#ffd43b",
        "monitor":"#69db7c","ceiling_light":"#c0eb75","cabinet":"#f59f00",
        "shelf":"#fd7e14","whiteboard":"#f8f9fa","machine":"#ff0a54",
        "pallet":"#adb5bd","trash_bin":"#da77f2","box":"#adb5bd",
        "pillar":"#74b816","support_beam":"#5c940d","duct":"#339af0",
    }
    DEF = "#cccccc"
    REL_COLOR = {
        "standing_on":  "#ff9f43", "lying_on":     "#ffa94d",
        "hanging_on":   "#ffd43b", "connected_to": "#74c0fc",
        "supported_by": "#a9e34b", "standing_in":  "#ff6b6b",
        "lying_in":     "#f03e3e",
        "left":         "#4dabf7", "right":        "#339af0",
        "front":        "#63e6be", "behind":       "#20c997",
        "close_by":     "#868e96",
        "higher_than":  "#da77f2", "lower_than":   "#cc5de8",
        "bigger_than":  "#e64980", "smaller_than": "#f783ac",
        "same_object_type": "#94d82d",
    }

    # ── Layer Z-heights (visual stacking, not physical) ──────────────────
    Z_BUILDING = 3.0
    Z_ROOM     = 2.0
    Z_AREA     = 1.0
    Z_OBJECT   = 0.0   # objects use their actual normalized XZ footprint

    areas = graph.get("areas", [])

    # Normalise object XZ to a [0,1] plotting box
    xs = np.array([n["centroid"][0] for n in nodes])
    zs = np.array([n["centroid"][2] for n in nodes])
    x_lo,x_hi = xs.min(), xs.max()
    z_lo,z_hi = zs.min(), zs.max()
    def norm_xz(x,z):
        nx = (x-x_lo)/max(x_hi-x_lo,1e-6)
        nz = (z-z_lo)/max(z_hi-z_lo,1e-6)
        return nx, nz

    obj_pos = {n["id"]: norm_xz(n["centroid"][0], n["centroid"][2]) for n in nodes}

    room_pos = {}
    for i, r in enumerate(rooms):
        rx, rz = norm_xz(r["centroid_xz"][0], r["centroid_xz"][1])
        room_pos[f"R{r['id']}"] = (rx, rz)

    # Same tab20 cycle + enumeration-order indexing as render_overlay's
    # scene_graph_areas.png, so area colors match across the two images.
    area_colors = plt.cm.tab20.colors
    area_color_by_id = {a["id"]: area_colors[i % len(area_colors)]
                        for i, a in enumerate(areas)}
    area_pos = {f"A{a['id']}": norm_xz(a["centroid_xz"][0], a["centroid_xz"][1])
               for a in areas}

    building_pos = (0.5, 0.5)

    fig = plt.figure(figsize=(16,14), dpi=120)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0e0e0e")
    fig.patch.set_facecolor("#0e0e0e")

    # ── Object layer: intra-object edges (drawn first, background) ──────
    for e in edges:
        srcp = obj_pos.get(e["src"]); dstp = obj_pos.get(e["dst"])
        if not srcp or not dstp: continue
        col = REL_COLOR.get(e["relation"], "#888888")
        alpha = 0.15 if e["relation"]=="similar" else 0.5
        ax.plot([srcp[0],dstp[0]], [srcp[1],dstp[1]], [Z_OBJECT,Z_OBJECT],
                color=col, alpha=alpha, lw=0.6, zorder=1)

    # ── Object nodes ──────────────────────────────────────────────────
    for n in nodes:
        px,pz = obj_pos[n["id"]]
        col = LABEL_COLOR.get(n["label"], DEF)
        sz  = 8 + min(n["n_world_pts"]*0.3, 25)
        ax.scatter([px],[pz],[Z_OBJECT], c=col, s=sz, alpha=0.75,
                  edgecolors="#00000044", linewidths=0.3, zorder=2)

    # ── Area → Object edges (vertical) ───────────────────────────────────
    # Uses each area's own member_ids directly (areas already carry this),
    # rather than re-deriving membership from the object side. Falls back
    # to direct Room → Object edges (skipping the area layer visually) if
    # this graph has no area data at all, so objects are never silently
    # left disconnected from the hierarchy.
    if not areas:
        room_node_objs = defaultdict(list)
        for n in nodes:
            if n["room_id"] >= 0:
                room_node_objs[f"R{n['room_id']}"].append(n["id"])
        room_colors_fb = plt.cm.tab10.colors
        for i, (room_key, obj_ids) in enumerate(room_node_objs.items()):
            if room_key not in room_pos: continue
            rx, rz = room_pos[room_key]
            col = room_colors_fb[i % len(room_colors_fb)]
            sample = obj_ids if len(obj_ids) <= 80 else \
                     list(np.random.RandomState(0).choice(obj_ids, 80, replace=False))
            for oid in sample:
                if oid not in obj_pos: continue
                ox, oz = obj_pos[oid]
                ax.plot([rx,ox],[rz,oz],[Z_ROOM,Z_OBJECT],
                        color=col, alpha=0.25, lw=0.5, zorder=1)

    for a in areas:
        area_key = f"A{a['id']}"
        if area_key not in area_pos: continue
        ax_pos, az_pos = area_pos[area_key]
        col = area_color_by_id[a["id"]]
        obj_ids = a["member_ids"]
        # Sample a subset of edges if there are too many (visual clarity)
        sample = obj_ids if len(obj_ids) <= 80 else \
                 list(np.random.RandomState(0).choice(obj_ids, 80, replace=False))
        for oid in sample:
            if oid not in obj_pos: continue
            ox,oz = obj_pos[oid]
            ax.plot([ax_pos,ox],[az_pos,oz],[Z_AREA,Z_OBJECT],
                    color=col, alpha=0.25, lw=0.5, zorder=1)

    # ── Area nodes ────────────────────────────────────────────────────
    for a in areas:
        area_key = f"A{a['id']}"
        if area_key not in area_pos: continue
        ax_pos, az_pos = area_pos[area_key]
        col = area_color_by_id[a["id"]]
        ax.scatter([ax_pos],[az_pos],[Z_AREA], c=[col], s=180, alpha=0.9,
                  edgecolors="white", linewidths=0.8, zorder=3)
        ax.text(ax_pos, az_pos, Z_AREA+0.1, area_key, color="white",
               fontsize=7, weight="bold", ha="center")

    # ── Room → Area edges (vertical) ─────────────────────────────────────
    room_areas = defaultdict(list)
    for a in areas:
        room_areas[f"R{a['room_id']}"].append(f"A{a['id']}")

    room_colors = plt.cm.tab10.colors
    for i, (room_key, area_keys) in enumerate(room_areas.items()):
        if room_key not in room_pos: continue
        rx,rz = room_pos[room_key]
        col = room_colors[i % len(room_colors)]
        for ak in area_keys:
            if ak not in area_pos: continue
            ax_pos, az_pos = area_pos[ak]
            ax.plot([rx,ax_pos],[rz,az_pos],[Z_ROOM,Z_AREA],
                    color=col, alpha=0.4, lw=0.8, zorder=2)

    # ── Room nodes ────────────────────────────────────────────────────
    for i, (rk,(rx,rz)) in enumerate(room_pos.items()):
        col = room_colors[i % len(room_colors)]
        ax.scatter([rx],[rz],[Z_ROOM], c=[col], s=400, alpha=0.95,
                  edgecolors="white", linewidths=1.2, zorder=3)
        ax.text(rx, rz, Z_ROOM+0.15, rk, color="white", fontsize=11,
               weight="bold", ha="center")

    # ── Building → Room edges ─────────────────────────────────────────
    bx,bz = building_pos
    for i,(rk,(rx,rz)) in enumerate(room_pos.items()):
        col = room_colors[i % len(room_colors)]
        ax.plot([bx,rx],[bz,rz],[Z_BUILDING,Z_ROOM],
                color=col, alpha=0.6, lw=1.5, zorder=2)

    # ── Building node ─────────────────────────────────────────────────
    ax.scatter([bx],[bz],[Z_BUILDING], c="#9b59ff", s=700, alpha=0.95,
              edgecolors="white", linewidths=1.5, zorder=4)
    ax.text(bx, bz, Z_BUILDING+0.15, graph.get("space","Building"),
           color="white", fontsize=13, weight="bold", ha="center")

    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(-0.3, Z_BUILDING+0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.text2D(0.02, 0.95,
             f"Building → Room → Area → Object hierarchy\n"
             f"{len(rooms)} rooms · {len(areas)} areas · {len(nodes)} objects · "
             f"{len(edges)} object edges",
             transform=ax.transAxes, color="white", fontsize=11)
    ax.view_init(elev=18, azim=-60)
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0,0,0,0))

    out_dir = REPO / "ui" / "_spaces" / space_name / "topdown"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "scene_graph_3d.png"
    plt.savefig(str(out), facecolor="#0e0e0e", bbox_inches="tight")
    plt.close(fig)
    print(f"[sg] 3D hierarchy diagram → {out}")



def print_summary(graph: dict):
    try:
        import networkx as nx
    except ImportError:
        return
    G = nx.Graph()
    for n in graph["nodes"]:
        G.add_node(n["id"], label=n["label"], room=n["room_id"])
    for e in graph["edges"]:
        G.add_edge(e["src"], e["dst"], relation=e["relation"])
    rel_cnt:   dict = defaultdict(int)
    label_cnt: dict = defaultdict(int)
    for _,_,d in G.edges(data=True): rel_cnt[d["relation"]] += 1
    for _,d  in G.nodes(data=True): label_cnt[d["label"]] += 1
    print(f"\n[sg] ── Graph summary ───────────────────────────────────────")
    print(f"  Nodes           : {G.number_of_nodes()}")
    print(f"  Edges           : {G.number_of_edges()}")
    for rel, cnt in sorted(rel_cnt.items()):
        print(f"    {rel:<16}  {cnt}")
    print(f"  Connected comps : {nx.number_connected_components(G)}")
    top = sorted(label_cnt.items(), key=lambda x: -x[1])[:15]
    print(f"  Top labels:")
    for lbl, cnt in top:
        print(f"    {lbl:<22} {cnt}")
    print(f"[sg] ────────────────────────────────────────────────────────\n")


# ── Post-label height correction ─────────────────────────────────────────
# After CLIP labels are assigned, certain labels are structurally inconsistent
# with their current centroid Y (e.g. a pendant_lamp at floor level).  This
# pass snaps those objects to the correct point-cloud surface cluster for
# their label category, improving 3D position accuracy in the viewer.

_SG_CEILING_LABELS = frozenset([
    "ceiling_light", "pendant_lamp", "projector",
    "smoke_detector", "sprinkler", "duct",
])
_SG_FLOOR_LABELS = frozenset([
    "pallet", "crate", "barrel", "safety_barrier", "railing", "conveyor",
])


def correct_heights_from_labels(objects: dict, ply_snap_data) -> dict:
    """
    Snap objects to the point-cloud surface cluster implied by their label.
    Ceiling-labelled objects that ended up at floor level are moved to the
    highest cluster; floor-labelled objects that ended up too high are
    moved to the lowest cluster.
    """
    if ply_snap_data is None:
        return objects

    ply_xyz, xz_tree, ply_y_lo, ply_y_hi = ply_snap_data
    ceiling_zone_min = ply_y_hi - 1.0   # upper 1 m = ceiling zone

    n_ceiling_fixed = 0
    n_floor_fixed   = 0

    for obj in objects.values():
        label = obj.get("label", "")
        cx, cy, cz = obj["centroid"]

        needs_ceiling = label in _SG_CEILING_LABELS and cy < ceiling_zone_min
        needs_floor   = label in _SG_FLOOR_LABELS   and cy > ply_y_lo + 0.8

        if not (needs_ceiling or needs_floor):
            continue

        idxs = xz_tree.query_ball_point([[cx, cz]], r=PLY_SNAP_XZ_M)[0]
        if len(idxs) < PLY_SNAP_MIN_PTS:
            continue

        near_y = np.sort(ply_xyz[idxs, 1])
        gaps   = np.diff(near_y)
        split_pts = np.where(gaps > 0.50)[0] + 1
        clusters  = np.split(near_y, split_pts)

        if needs_ceiling:
            # Snap to the highest LOCAL surface cluster only — never to the
            # scan's global max height. Industrial roofs are uneven (trusses,
            # ducts, pipes routinely sit higher than the light fixtures
            # themselves), so a global fallback pulls lights up into open air
            # above their real position whenever the local column has no
            # return near the building's single tallest point. If there's no
            # local cluster meaningfully above the object's current position,
            # leave the (likely already-correct) backprojected position alone.
            top_cluster = max(clusters, key=lambda c: float(np.median(c)))
            top_y = float(np.median(top_cluster))
            if top_y - cy < 0.15:
                continue
            new_y = top_y
            n_ceiling_fixed += 1
        else:
            new_y = float(np.median(
                min(clusters, key=lambda c: float(np.median(c)))
            ))
            n_floor_fixed += 1

        if abs(new_y - cy) > 0.10:
            obj["centroid"][1] = new_y
            # Update bbox extent to reflect new position
            if "bbox_pts" in obj:
                pts = obj["bbox_pts"]
                if pts is not None and hasattr(pts, "__len__") and len(pts):
                    try:
                        pts_arr = np.asarray(pts)
                        shift   = new_y - cy
                        pts_arr[:, 1] += shift
                        bmin, bmax = _trimmed_bbox(pts_arr)
                        obj["bbox_size"] = (bmax - bmin).tolist()
                        obj["bbox_min"]  = bmin.tolist()
                        obj["bbox_max"]  = bmax.tolist()
                    except Exception:
                        pass

    if n_ceiling_fixed or n_floor_fixed:
        print(f"[sg] post-label height fix: {n_ceiling_fixed} ceiling + "
              f"{n_floor_fixed} floor objects re-snapped")
    return objects


# ══ UI space setup ════════════════════════════════════════════════════════

def _ensure_ui_space(space_name: str, sp_paths: dict):
    """Create / repair ui/_spaces/<space>/ so the viewer works out of the box.

    Sets up Data_ symlinks, viewer symlink, index.html, and topdown assets
    (topdown.png + bounds.json).  Safe to call on every scene_graph run —
    skips anything already in place.
    """
    vdir     = REPO / "ui" / "_spaces" / space_name
    data_src = sp_paths["data_root"]
    vdir.mkdir(parents=True, exist_ok=True)

    # ── viewer symlink ────────────────────────────────────────────────────
    viewer_link = vdir / "viewer"
    if not viewer_link.exists():
        viewer_link.symlink_to(REPO / "ui" / "viewer")
        print(f"[sg] ui: created viewer symlink")

    # ── Data_ directory with symlinks ─────────────────────────────────────
    data_dir = vdir / "Data_"
    data_dir.mkdir(exist_ok=True)
    for fname, target in [
        ("cameras.json", data_src / "cameras.json"),
        ("views",        data_src / "views"),
        ("panos",        data_src / "views"),   # cameras.json uses "panos/" prefix
    ]:
        link = data_dir / fname
        if not link.exists() and target.exists():
            link.symlink_to(target)
    frames_src = data_src / "frames"
    if frames_src.exists() and not (data_dir / "frames").exists():
        (data_dir / "frames").symlink_to(frames_src)

    # ── downsampled PLY — must come from the same data_root ──────────────
    ply_dst = data_dir / "downsampled_web.ply"
    if not ply_dst.exists():
        from _paths import _load_config
        cfg     = _load_config()
        my_root = cfg["spaces"].get(space_name, {}).get("data_root", "")
        copied  = False
        # Only copy from a sibling that shares the exact same data_root
        for sname, scfg in cfg["spaces"].items():
            if sname == space_name or scfg.get("data_root", "") != my_root:
                continue
            candidate = REPO / "ui" / "_spaces" / sname / "Data_" / "downsampled_web.ply"
            if candidate.exists():
                shutil.copy2(candidate, ply_dst)
                print(f"[sg] ui: copied downsampled_web.ply from sibling '{sname}' (same data_root)")
                copied = True
                break
        if not copied:
            try:
                subprocess.run(
                    [sys.executable, str(REPO / "pipeline" / "downsample_ply.py"),
                     space_name],
                    check=True)
                print(f"[sg] ui: generated downsampled_web.ply")
            except Exception as e:
                print(f"[sg] ui: warning — could not create downsampled_web.ply: {e}")

    # ── index.html from template ──────────────────────────────────────────
    index_path = vdir / "index.html"
    if not index_path.exists():
        template = REPO / "ui" / "space_template.html"
        if template.exists():
            html = template.read_text()
            html = html.replace("__SPACE_NAME__",        space_name)
            html = html.replace("__SPACE_TITLE__",       sp_paths.get("title", space_name))
            html = html.replace("__SPACE_DESCRIPTION__", "Pick a view mode below.")
            index_path.write_text(html)
            print(f"[sg] ui: wrote index.html")

    # ── topdown/topdown.png + bounds.json ─────────────────────────────────
    topdown_dir = vdir / "topdown"
    topdown_dir.mkdir(exist_ok=True)
    topdown_png = topdown_dir / "topdown.png"
    bounds_json = topdown_dir / "bounds.json"
    if not topdown_png.exists() or not bounds_json.exists():
        # Prefer copying from a sibling space that shares the same data_root —
        # this guarantees axes are identical and avoids PLY-source divergence.
        from _paths import _load_config
        cfg      = _load_config()
        my_root  = cfg["spaces"].get(space_name, {}).get("data_root", "")
        copied   = False
        for sname, scfg in cfg["spaces"].items():
            if sname == space_name:
                continue
            if scfg.get("data_root", "") != my_root:
                continue
            src_png    = REPO / "ui" / "_spaces" / sname / "topdown" / "topdown.png"
            src_bounds = REPO / "ui" / "_spaces" / sname / "topdown" / "bounds.json"
            if src_png.exists() and src_bounds.exists():
                shutil.copy2(src_png,    topdown_png)
                shutil.copy2(src_bounds, bounds_json)
                print(f"[sg] ui: copied topdown assets from sibling '{sname}' (same data_root)")
                copied = True
                break
        if not copied:
            # No same-data sibling — generate fresh, forcing the web PLY so
            # axis selection is consistent with any future downsample run.
            web_ply = data_dir / "downsampled_web.ply"
            ply_arg = ["--ply", str(web_ply)] if web_ply.exists() else []
            try:
                subprocess.run(
                    [sys.executable, str(REPO / "pipeline" / "gen_topdown.py"),
                     "--space", space_name] + ply_arg,
                    check=True)
                print(f"[sg] ui: generated topdown.png + bounds.json")
            except Exception as e:
                print(f"[sg] ui: warning — topdown generation failed: {e}; overlays will be skipped")

    print(f"[sg] ui: space {space_name} ready at {vdir}")


# ══ main ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Build a rich 3DSSG/ConceptGraphs-style scene graph.")
    ap.add_argument("--space",          required=True, choices=space_choices())
    ap.add_argument("--out-dir",        default=None)
    ap.add_argument("--above-delta",    type=float, default=ABOVE_DELTA_M)
    ap.add_argument("--hanging-delta",  type=float, default=HANGING_DELTA_M)
    ap.add_argument("--min-world-pts",  type=int,   default=MIN_WORLD_PTS)
    ap.add_argument("--min-proposals",  type=int,   default=MIN_PROPOSALS)
    ap.add_argument("--max-nodes",      type=int,   default=MAX_NODES)
    ap.add_argument("--dedup-m",        type=float, default=DEDUP_M)
    ap.add_argument("--dedup-cos",      type=float, default=DEDUP_COS)
    ap.add_argument("--n-rooms",        type=int,   default=0,
                    help="Known number of rooms/zones (0=auto-detect)")
    ap.add_argument("--room-eps",       type=float, default=ROOM_EPS_M)
    ap.add_argument("--room-subsample", type=int,   default=ROOM_SUBSAMPLE)
    ap.add_argument("--wall-cell-m",    type=float, default=WALL_CELL_M,
                    help="Occupancy grid cell size for wall detection")
    ap.add_argument("--wall-min-bands", type=int,   default=WALL_MIN_BANDS,
                    help="Min height bands (of 6) a cell must span to count as wall")
    ap.add_argument("--sim-cos",        type=float, default=SIM_COS_THR)
    ap.add_argument("--sim-k",          type=int,   default=SIM_K)
    ap.add_argument("--labels",         default="",
                    help="Extra comma-separated label words added to vocab")
    ap.add_argument("--absorb-size-ratio",  type=float, default=ABSORB_SIZE_RATIO,
                    help="Max volume ratio (small/container) to be an absorption candidate")
    ap.add_argument("--absorb-containment", type=float, default=ABSORB_CONTAINMENT,
                    help="Min fraction of the small object's volume overlapping the container")
    ap.add_argument("--absorb-radius-m",    type=float, default=ABSORB_RADIUS_M,
                    help="Candidate-pair search radius (centroid distance) for absorption")
    ap.add_argument("--absorb-vert-margin-m", type=float, default=ABSORB_VERT_MARGIN_M,
                    help="Vertical tolerance for 'resting on top of' contact")
    ap.add_argument("--no-absorb",      action="store_true",
                    help="Disable container absorption (report every object as its own node)")
    ap.add_argument("--area-min-gap-m",   type=float, default=AREA_MIN_GAP_M,
                    help="Min real-world gap (m) between object clusters to "
                         "count as a genuine area boundary")
    ap.add_argument("--area-max-objects", type=int,   default=AREA_MAX_OBJECTS,
                    help="Safety valve: force a split above this object count "
                         "even with no qualifying gap")
    ap.add_argument("--area-max-size-m",  type=float, default=AREA_MAX_SIZE_M,
                    help="Safety valve: force a split above this footprint (m) "
                         "even with no qualifying gap")
    ap.add_argument("--area-max-room-frac", type=float, default=AREA_MAX_ROOM_FRAC,
                    help="An area may not span more than this fraction of "
                         "its own room's footprint (either axis)")
    ap.add_argument("--no-area-split", action="store_true",
                    help="Skip area subdivision entirely — each room "
                         "becomes exactly one area containing all its objects")
    ap.add_argument("--no-viewer-copy", action="store_true")
    ap.add_argument("--no-overlay",     action="store_true")
    ap.add_argument("--no-yaw-correct", action="store_true",
                    help="Disable wall-alignment box rotation — keep boxes "
                         "axis-aligned to raw world X/Z even if the scan "
                         "isn't itself axis-aligned.")
    ap.add_argument("--no-corrected-centroids", action="store_true",
                    help="Ignore out_dir/corrected_centroids.json (from "
                         "03c_refine_objects.py) even if present — use the "
                         "raw trimmed-median XZ estimate instead.")
    ap.add_argument("--no-refined-ids", action="store_true",
                    help="Ignore out_dir/object_ids_refined.npy (from "
                         "03c_refine_objects.py) even if present — use "
                         "object_ids.npy (03b's output) directly instead.")
    args = ap.parse_args()

    extra_labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    sp_paths     = space(args.space)
    if args.out_dir: sp_paths["out_dir"] = Path(args.out_dir)
    out_dir   = sp_paths["out_dir"]
    views_dir = sp_paths["views"]

    for req in ("metadata.json", "embeddings.npy", "object_ids.npy"):
        if not (out_dir / req).exists():
            print(f"[sg] ERROR: {out_dir / req} not found — run pipeline first.")
            sys.exit(1)

    corrected_centroids = None
    corrected_path = out_dir / "corrected_centroids.json"
    if not args.no_corrected_centroids and corrected_path.exists():
        raw = json.loads(corrected_path.read_text())
        corrected_centroids = {int(k): v for k, v in raw.items()}
        print(f"[sg] loaded {len(corrected_centroids)} BEV-corrected centroids "
              f"from {corrected_path}")

    data    = load_pipeline(sp_paths, use_refined_ids=not args.no_refined_ids)
    meta    = data["meta"]
    ply_snap = load_ply_for_snap(sp_paths)
    objects = build_objects(data, args.min_world_pts, args.min_proposals,
                            args.max_nodes, args.dedup_m, args.dedup_cos,
                            ply_snap_data=ply_snap,
                            corrected_centroids=corrected_centroids)
    if not objects:
        print("[sg] ERROR: no objects survived filtering."); sys.exit(1)

    if not args.no_absorb:
        objects = absorb_contained_objects(objects,
                                            size_ratio_thr=args.absorb_size_ratio,
                                            containment_frac=args.absorb_containment,
                                            search_radius_m=args.absorb_radius_m,
                                            vert_margin_m=args.absorb_vert_margin_m)

    room_meta = detect_rooms(sp_paths, eps_m=args.room_eps,
                              min_pts=ROOM_MIN_PTS,
                              slice_lo=ROOM_SLICE_LO, slice_hi=ROOM_SLICE_HI,
                              subsample=args.room_subsample,
                              n_rooms_hint=args.n_rooms,
                              wall_cell_m=args.wall_cell_m,
                              wall_min_bands=args.wall_min_bands)
    objects   = assign_rooms(objects, room_meta)
    objects, areas = build_areas(objects,
                                  min_gap_m=args.area_min_gap_m,
                                  max_objects_per_area=args.area_max_objects,
                                  max_area_size_m=args.area_max_size_m,
                                  max_room_frac=args.area_max_room_frac,
                                  split=not args.no_area_split)
    objects   = label_objects(objects, meta, views_dir,
                              extra_labels=extra_labels)
    objects   = build_attributes(objects)
    objects   = correct_heights_from_labels(objects, ply_snap)
    yaw_deg   = 0.0 if args.no_yaw_correct else room_meta.get("_yaw_deg", 0.0)
    objects   = apply_building_yaw(objects, yaw_deg)
    edges     = build_edges(objects,
                            above_delta       = args.above_delta,
                            hanging_delta     = args.hanging_delta,
                            max_direct_gap    = MAX_DIRECT_GAP_M,
                            max_hang_gap      = MAX_HANG_GAP_M,
                            footprint_iou_thr = FOOTPRINT_IOU_THR,
                            on_floor_m        = ON_FLOOR_M,
                            yaw_deg           = yaw_deg)

    graph   = serialise(objects, edges, room_meta, areas, args.space, yaw_deg=yaw_deg)
    sg_path = out_dir / "scene_graph.json"
    sg_path.write_text(json.dumps(graph, indent=2))
    print(f"[sg] → {sg_path}  ({graph['stats']})")

    if not args.no_viewer_copy:
        _ensure_ui_space(args.space, sp_paths)
        vdir = REPO / "ui" / "_spaces" / args.space
        dest = vdir / "scene_graph.json"
        dest.write_text(json.dumps(graph, indent=2))
        print(f"[sg] viewer copy → {dest}")

    if not args.no_overlay:
        render_overlay(graph, args.space)
        render_hydra_diagram(graph, args.space)

    print_summary(graph)
    print(f"[sg] Done — {graph['stats']['n_nodes']} nodes · "
          f"{graph['stats']['n_edges']} edges · "
          f"{graph['stats']['n_rooms']} rooms")


if __name__ == "__main__":
    main()