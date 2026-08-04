#!/usr/bin/env python
"""Reconcile a finished box set into a consistent scene.

Several passes contribute boxes and labels independently, so a final pass is
needed to make the result coherent. This applies six corrections:

  1. ORIENTATION. Every box is snapped to the room grid or its perpendicular.
     Furniture in these rooms is square to the room, and an angle fitted to a
     noisy footprint is noise rather than information.
  2. SURFACE SYNONYMS. table / desk / workbench / bench / counter name the same
     physical thing, and a classifier picks between them per view. One name per
     space is both consistent and honest about the distinction not being
     measurable from geometry.
  3. WORK SURFACES STAND ON THE FLOOR. A box labelled as a surface whose base
     is well above the floor is something resting ON a surface, not a surface.
     It is restored to its most informative recorded label rather than deleted.
  4. SEATING VS SLAB. A dense flat slab at a box's own top means a table or
     podium, not a chair — a chair has a backrest and a gap above the seat.
  5. SHELF TIERS. A surface whose footprint sits inside a shelf or rack is that
     shelf's own tier, not a workbench.
  6. DUPLICATES AND LOOK-ALIKES. Overlapping same-kind boxes are suppressed,
     and groups of geometrically identical objects are given a common label
     when one already holds a clear majority.

Nothing here invents geometry; it only reconciles what earlier passes left
inconsistent.

Usage:
  python harmonize_scene.py --boxes boxes.json --ply data/scene.ply \
    --scale-to-meters 6.8 --room-yaw-deg 28.07 --collapse-surfaces workbench \
    --out boxes_final.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEATING_CLASSES = {"chair", "office_chair", "seat", "stool", "armchair"}
# Classes a dense flat slab at the box's own top should override. A chair has a
# backrest and a gap above the seat; a person is not a slab either — CLIP
# labelled one of shinhan's tables "person", which this catches.
SLAB_TESTABLE = SEATING_CLASSES | {"person", "object", "unclassified_object",
                                   "whiteboard", "partition_panel"}
SURFACE_CLASSES = {"table", "desk", "workbench", "bench", "counter", "workstation"}
CONTAINER_CLASSES = {"shelf", "storage_rack", "rack", "cabinet", "shelving"}

# OWLv2 is prompted with spaced phrases ("storage rack") while the rest of the
# pipeline uses underscored labels ("storage_rack"). Restoring a recorded prior
# label reintroduced the prompt spelling, so factory_space_14 ended up listing
# storage_rack AND "storage rack", cardboard_box AND "cardboard box",
# computer_monitor AND "monitor" as separate classes in the viewer legend.
ALIAS = {
    "storage rack": "storage_rack", "cardboard box": "cardboard_box",
    "trash bin": "trash_bin", "fire extinguisher": "fire_extinguisher",
    "computer monitor": "computer_monitor", "monitor": "computer_monitor",
    "chair": "office_chair", "office chair": "office_chair", "desk": "table",
    "partition panel": "partition_panel", "air duct": "air_duct", "air conditioner": "air_conditioner",
    "ceiling light": "light", "ceiling_light": "light", "lamp": "light",
    "potted plant": "plant", "pottedplant": "plant",
}


def canon(lbl):
    return ALIAS.get(lbl, lbl)


def yaw_rot2(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def footprint_corners(b):
    c = np.asarray(b["center"], dtype=float)[:2]
    h = np.asarray(b["size"], dtype=float)[:2] / 2.0
    R = yaw_rot2(float(b.get("angle", 0.0)))
    return np.array([c + R @ (h * s) for s in
                     ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def clip_poly(subject, clip):
    """Sutherland-Hodgman: intersection of two convex polygons."""
    out = subject
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a
        nxt, n = [], len(out)
        if n == 0:
            return np.zeros((0, 2))
        for j in range(n):
            p, q = out[j], out[(j + 1) % n]
            sp = edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
            sq = edge[0] * (q[1] - a[1]) - edge[1] * (q[0] - a[0])
            if sp <= 0:
                nxt.append(p)
            if (sp <= 0) != (sq <= 0):
                t = sp / (sp - sq) if (sp - sq) != 0 else 0.0
                nxt.append(p + t * (q - p))
        out = np.array(nxt)
    return out


def footprint_iou(b1, b2):
    p1, p2 = footprint_corners(b1), footprint_corners(b2)
    inter = clip_poly(p1, p2)
    if len(inter) < 3:
        return 0.0, 0.0
    ai = poly_area(inter)
    a1, a2 = poly_area(p1), poly_area(p2)
    union = a1 + a2 - ai
    return (ai / union if union > 0 else 0.0), (ai / min(a1, a2) if min(a1, a2) > 0 else 0.0)


def z_overlap(b1, b2):
    lo1, hi1 = b1["center"][2] - b1["size"][2] / 2, b1["center"][2] + b1["size"][2] / 2
    lo2, hi2 = b2["center"][2] - b2["size"][2] / 2, b2["center"][2] + b2["size"][2] / 2
    inter = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    return inter / max(min(hi1 - lo1, hi2 - lo2), 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale-to-meters", type=float, required=True)
    ap.add_argument("--room-yaw-deg", type=float, required=True)
    ap.add_argument("--ply", default=None,
                    help="splat in the frame of --boxes; enables the seating-vs-surface "
                         "test, which needs the geometry and not just the box")
    ap.add_argument("--top-band-m", type=float, default=0.06,
                    help="thickness of the slab band tested at a box's top")
    ap.add_argument("--top-concentration", type=float, default=0.20,
                    help="share of a box's TOP-REGION material that must fall in the "
                         "thin top band for it to count as a slab rather than a backrest. "
                         "CALIBRATED on shinhan_space, where the two classes separate "
                         "cleanly: tables 0.22/0.34/0.42 at p25/p50/p75 against chairs "
                         "0.03/0.07/0.16. Anything above ~0.45 sits over BOTH "
                         "distributions and flips nothing")
    ap.add_argument("--top-density", type=float, default=500.0,
                    help="minimum gaussians per m^2 in that band; measured medians were "
                         "1159 for tables against 242 for chairs")
    ap.add_argument("--surface-max-base-m", type=float, default=0.35,
                    help="a horizontal work surface stands on the floor; a box labelled "
                         "as one whose base is higher than this is something resting ON "
                         "a surface, not a surface")
    ap.add_argument("--collapse-surfaces", default=None,
                    help="rename EVERY horizontal-surface class to this one label. "
                         "table/desk/workbench/bench/counter are synonyms for the same "
                         "physical thing and CLIP picks between them per crop, so a row "
                         "of identical benches came out split across two names. One name "
                         "per space is both consistent and honest about the distinction "
                         "not being measurable here")
    ap.add_argument("--snap-tol-deg", type=float, default=45.0,
                    help="snap a box's yaw to the room grid (or its perpendicular) when "
                         "it lies within this of it. 45 means ALWAYS snap, which put "
                         "every box in a scene on a single angle and left visibly wrong "
                         "orientations on objects that genuinely sit askew (a shelf "
                         "pushed at an angle, a swivelled chair). 20 treats a small "
                         "deviation as fit noise and a large one as real. The default "
                         "45 always snaps, putting every box on the room grid with no "
                         "exceptions, which is what reads cleanly in the viewer. Lower "
                         "it only if a room genuinely contains off-grid furniture")
    ap.add_argument("--merge-surfaces", action="store_true", default=False,
                    help="merge collinear adjacent surfaces into continuous runs. OFF "
                         "by default: it chains transitively, so in a factory with rows "
                         "of benches pushed end to end it fused 22 segments into a "
                         "single 10.70 m box and cut factory_space_13 from 47 "
                         "workbenches to 13. Over-segmentation is better handled at "
                         "source with --max-long-m than by merging afterwards")
    ap.add_argument("--merge-gap-m", type=float, default=0.30,
                    help="two same-label surfaces at the same height whose footprints "
                         "come within this of each other are one continuous run")
    ap.add_argument("--merge-top-tol-m", type=float, default=0.10,
                    help="how closely their top surfaces must agree in height")
    ap.add_argument("--cover-frac", type=float, default=0.60,
                    help="drop a same-kind box with this much of its footprint inside "
                         "another; it adds nothing the larger box does not already bound")
    ap.add_argument("--dup-iou", type=float, default=0.40,
                    help="footprint IoU above which two same-kind boxes are duplicates")
    ap.add_argument("--contained-frac", type=float, default=0.65,
                    help="a surface box with this much of its footprint inside a "
                         "shelf/rack is that shelf's own tier, not a workbench")
    ap.add_argument("--min-majority", type=float, default=0.6,
                    help="a look-alike group is only harmonised when one label already "
                         "holds this share of it; a genuinely mixed group is left alone")
    ap.add_argument("--size-tol", type=float, default=0.15,
                    help="relative footprint/height tolerance for treating two boxes "
                         "as the same KIND of object during label harmonisation")
    ap.add_argument("--base-tol-m", type=float, default=0.22)
    ap.add_argument("--min-group", type=int, default=3,
                    help="only harmonise labels within groups of at least this many "
                         "look-alike objects; below that there is no majority to trust")
    args = ap.parse_args()

    S = args.scale_to_meters
    boxes = json.loads(Path(args.boxes).read_text())["boxes"]
    n0 = len(boxes)
    room = np.radians(args.room_yaw_deg)
    tol = np.radians(args.snap_tol_deg)

    # ---- 1. yaw snapping -------------------------------------------------
    snapped = 0
    for b in boxes:
        a = float(b.get("angle", 0.0))
        best, bd = None, np.inf
        for k in range(-4, 5):                      # room grid and perpendiculars
            cand = room + k * (np.pi / 2)
            d = abs(a - cand)
            if d < bd:
                best, bd = cand, d
        if bd <= tol:
            if abs(best - a) > 1e-9:
                b["yaw_snapped_from"] = float(np.degrees(a))
                snapped += 1
            b["angle"] = float(best)
    print(f"[harmonise] yaw: snapped {snapped}/{n0} boxes to the room grid "
          f"(within {args.snap_tol_deg:.0f}°)")
    ym = np.mod(np.degrees([b.get("angle", 0.0) for b in boxes]), 90)
    print(f"[harmonise]      yaw mod 90 now: std={ym.std():.1f}° "
          f"(p5={np.percentile(ym, 5):.1f} p95={np.percentile(ym, 95):.1f})")

    # ---- 1b. collapse surface synonyms ------------------------------------
    if args.collapse_surfaces:
        n = 0
        for b in boxes:
            if b["label"] in SURFACE_CLASSES and b["label"] != args.collapse_surfaces:
                b["surface_synonym_from"] = b["label"]
                b["label"] = args.collapse_surfaces
                n += 1
        print(f"[harmonise] surfaces: collapsed {n} boxes to '{args.collapse_surfaces}'")

    # ---- 1b2. a work surface must stand on the floor ----------------------
    # Anything resting ON a bench is not itself a bench. Measured on
    # factory_space_14, 11 of 48 boxes labelled "workbench" had their base a
    # median 0.77 m above the floor, and their pre-collapse labels were printer
    # x4, monitor, computer_monitor, light, chair, shelf, storage rack and desk
    # — equipment on the benches that the surface-synonym collapse then renamed.
    # They are restored to the most informative non-surface label recorded for
    # them rather than deleted: the box is real, only the class was wrong.
    floor_z = min(b["center"][2] - b["size"][2] / 2.0 for b in boxes)
    max_base = args.surface_max_base_m / S
    restored = Counter()
    for b in boxes:
        if b["label"] not in SURFACE_CLASSES:
            continue
        if (b["center"][2] - b["size"][2] / 2.0) - floor_z <= max_base:
            continue
        prior = next((canon(b[k]) for k in ("clip_relabelled_from", "surface_synonym_from",
                                            "label_harmonised_from")
                      if b.get(k) and canon(b[k]) not in SURFACE_CLASSES), None)
        # With no recorded prior there is nothing to restore TO. Inventing an
        # "object" class is worse than useless — it is a label that says nothing
        # and still has to be looked at. Keep the box and flag it instead.
        b["elevated_surface"] = True
        if prior:
            b["not_a_surface_was"] = b["label"]
            b["label"] = prior
            restored[prior] += 1
        else:
            restored["(kept, no prior label)"] += 1
    if restored:
        print(f"[harmonise] elevated surfaces: {sum(restored.values())} boxes were "
              f"labelled as work surfaces but stand >{args.surface_max_base_m} m off the "
              f"floor — restored to {dict(restored)}")

    # ---- 1c. seating that is really a work surface ------------------------
    # A box labelled "chair" whose own TOP is a dense flat horizontal surface is
    # a table, a podium or a bench top — a chair has a backrest and a gap above
    # the seat, not a slab. This is a relabel, never a delete: the geometry is
    # right and only the class is wrong.
    if args.ply:
        from plyfile import PlyData
        pv = PlyData.read(args.ply)["vertex"]
        gxyz = np.stack([pv["x"], pv["y"], pv["z"]], axis=1).astype(float)
        gop = 1.0 / (1.0 + np.exp(-np.asarray(pv["opacity"], dtype=float)))
        gxyz = gxyz[gop >= 0.3]
        band = args.top_band_m / S
        flipped = 0
        for b in boxes:
            if b["label"] not in SLAB_TESTABLE:
                continue
            top = b["center"][2] + b["size"][2] / 2.0
            inside = np.all(np.abs((gxyz[:, :2] - np.asarray(b["center"])[:2])
                                   @ yaw_rot2(-float(b.get("angle", 0.0))).T)
                            <= np.asarray(b["size"])[:2] / 2.0, axis=1)
            if inside.sum() < 30:
                continue
            zin = gxyz[inside, 2]
            slab = np.abs(zin - top) <= band            # material right at the top
            upper = zin >= top - 3 * band               # material in the top region
            if upper.sum() < 20:
                continue
            # a slab concentrates its top-region material into the thin top band;
            # a chair back spreads it out
            concentration = slab.sum() / upper.sum()
            area_m2 = float(np.prod(np.asarray(b["size"])[:2])) * S * S
            density = slab.sum() / max(area_m2, 1e-6)
            if concentration >= args.top_concentration and density >= args.top_density:
                b["seating_reclassified_from"] = b["label"]
                b["label"] = args.collapse_surfaces or "table"
                b["top_concentration"] = round(float(concentration), 3)
                flipped += 1
        print(f"[harmonise] seating: {flipped} 'chair' boxes reclassified as work "
              f"surfaces (dense flat slab at their own top)")

    # ---- 2. surfaces inside shelving -------------------------------------
    containers = [b for b in boxes if b["label"] in CONTAINER_CLASSES]
    drop = set()
    for i, b in enumerate(boxes):
        if b["label"] not in SURFACE_CLASSES:
            continue
        for c in containers:
            if c is b:
                continue
            _iou, cover = footprint_iou(b, c)
            if cover >= args.contained_frac and z_overlap(b, c) > 0.1:
                drop.add(i)
                break
    if drop:
        print(f"[harmonise] shelving: dropped {len(drop)} surface boxes whose footprint "
              f"sits inside a shelf/rack — " +
              ", ".join(f"{k}={v}" for k, v in
                        Counter(boxes[i]["label"] for i in drop).items()))
    boxes = [b for i, b in enumerate(boxes) if i not in drop]

    # ---- 2b. merge collinear adjacent surfaces ----------------------------
    # One continuous bench should be one box. It arrives as several either
    # because detect_tables_topdown.py split it at --max-long-m or because
    # OWLv2 detected segments of it separately. Now that every box is snapped
    # to the room grid, "same run of bench" is simple to state: same label,
    # tops at the same height, and footprints that touch or overlap along one
    # axis while agreeing on the other.
    merged_n = 0
    changed = bool(args.merge_surfaces)
    while changed:
        changed = False
        for i in range(len(boxes)):
            if boxes[i] is None or boxes[i]["label"] not in SURFACE_CLASSES:
                continue
            for j in range(i + 1, len(boxes)):
                bj = boxes[j]
                if bj is None or bj["label"] != boxes[i]["label"]:
                    continue
                bi = boxes[i]
                if abs((bi["center"][2] + bi["size"][2] / 2) -
                       (bj["center"][2] + bj["size"][2] / 2)) > args.merge_top_tol_m / S:
                    continue
                if abs(float(bi.get("angle", 0)) - float(bj.get("angle", 0))) > np.radians(5):
                    continue
                th = float(bi.get("angle", 0.0))
                R = yaw_rot2(-th)
                ci, cj = R @ np.asarray(bi["center"])[:2], R @ np.asarray(bj["center"])[:2]
                hi_, hj = np.asarray(bi["size"])[:2] / 2, np.asarray(bj["size"])[:2] / 2
                lo1, up1 = ci - hi_, ci + hi_
                lo2, up2 = cj - hj, cj + hj
                gap = np.maximum(lo1, lo2) - np.minimum(up1, up2)   # <0 => overlap
                # touching/overlapping on one axis, aligned on the other
                if not (gap.min() <= args.merge_gap_m / S and gap.max() <= args.merge_gap_m / S):
                    continue
                lo, up = np.minimum(lo1, lo2), np.maximum(up1, up2)
                new_c2 = yaw_rot2(th) @ ((lo + up) / 2.0)
                zlo = min(bi["center"][2] - bi["size"][2] / 2, bj["center"][2] - bj["size"][2] / 2)
                zhi = max(bi["center"][2] + bi["size"][2] / 2, bj["center"][2] + bj["size"][2] / 2)
                bi["center"] = [float(new_c2[0]), float(new_c2[1]), float((zlo + zhi) / 2)]
                bi["size"] = [float(up[0] - lo[0]), float(up[1] - lo[1]), float(zhi - zlo)]
                bi["merged_count"] = bi.get("merged_count", 1) + bj.get("merged_count", 1)
                boxes[j] = None
                merged_n += 1
                changed = True
    if merged_n:
        boxes = [b for b in boxes if b is not None]
        runs = [b for b in boxes if b.get("merged_count", 1) > 1]
        print(f"[harmonise] merged {merged_n} collinear surface segments into "
              f"{len(runs)} continuous runs "
              f"(longest now {max((max(b['size'][:2]) * S for b in runs), default=0):.2f} m)")

    # ---- 3. duplicate suppression ----------------------------------------
    def support(b):
        return (b.get("n_mask_points") or b.get("n_instance_gaussians")
                or b.get("n_votes") or 0)

    order = sorted(range(len(boxes)),
                   key=lambda i: (-float(np.prod(boxes[i]["size"])), -support(boxes[i])))
    keep = [True] * len(boxes)
    removed = []
    for a_i in range(len(order)):
        i = order[a_i]
        if not keep[i]:
            continue
        for b_i in range(a_i + 1, len(order)):
            j = order[b_i]
            if not keep[j]:
                continue
            same_kind = (boxes[i]["label"] == boxes[j]["label"] or
                         (boxes[i]["label"] in SURFACE_CLASSES and
                          boxes[j]["label"] in SURFACE_CLASSES))
            if not same_kind:
                continue
            iou, cover = footprint_iou(boxes[i], boxes[j])
            # `cover` catches the case seen in the viewer: a third box sitting
            # between two that already bound their benches, mostly inside one of
            # them, which IoU alone scores too low to suppress.
            if ((iou >= args.dup_iou or cover >= args.cover_frac)
                    and z_overlap(boxes[i], boxes[j]) > 0.3):
                keep[j] = False
                removed.append((boxes[j]["label"], boxes[i]["label"], iou))
    if removed:
        print(f"[harmonise] duplicates: removed {len(removed)} overlapping same-kind "
              f"boxes (IoU >= {args.dup_iou})")
        for lost, won, iou in removed[:12]:
            print(f"      {lost} absorbed by {won} (footprint IoU {iou:.2f})")
    boxes = [b for i, b in enumerate(boxes) if keep[i]]

    # ---- 4. label harmonisation across look-alikes ------------------------
    def feat(b):
        w, l = sorted(np.asarray(b["size"], dtype=float)[:2])
        return np.array([w, l, b["size"][2], b["center"][2] - b["size"][2] / 2])

    # Greedy grouping against a group REPRESENTATIVE, not transitive closure.
    # Union-find chains: A~B and B~C merge A with C even when A and C are
    # nothing alike, and with a loose tolerance that produced a single group of
    # 61 boxes spanning 12 classes which then all became "workbench". Comparing
    # every member to one representative keeps a group genuinely homogeneous.
    F = [feat(b) for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: -float(np.prod(boxes[i]["size"])))
    reps, groups = [], []
    base_tol = args.base_tol_m / S
    for i in order:
        placed = False
        for gi, r in enumerate(reps):
            rel = np.abs(F[i][:3] - r[:3]) / np.maximum(np.maximum(F[i][:3], r[:3]), 1e-9)
            if rel.max() <= args.size_tol and abs(F[i][3] - r[3]) <= base_tol:
                groups[gi].append(i)
                placed = True
                break
        if not placed:
            reps.append(F[i])
            groups.append([i])

    global_freq = Counter(b["label"] for b in boxes)
    changed = 0
    for members in groups:
        if len(members) < args.min_group:
            continue
        votes = Counter(boxes[i]["label"] for i in members)
        if len(votes) == 1:
            continue
        win, wn = max(votes.items(), key=lambda kv: (kv[1], global_freq[kv[0]]))
        share = wn / len(members)
        if share < args.min_majority:
            print(f"[harmonise] label group of {len(members)} left alone — no clear "
                  f"majority ({dict(votes)}, top {share:.0%})")
            continue
        for i in members:
            if boxes[i]["label"] != win:
                boxes[i]["label_harmonised_from"] = boxes[i]["label"]
                boxes[i]["label"] = win
                changed += 1
        print(f"[harmonise] label group of {len(members)} look-alikes "
              f"(footprint ~{F[members[0]][0] * S:.2f}x{F[members[0]][1] * S:.2f} m): "
              f"{dict(votes)} -> {win} ({share:.0%})")
    print(f"[harmonise] labels: {changed} boxes relabelled to match their look-alikes")

    n_alias = 0
    for b in boxes:
        c = canon(b["label"])
        if c != b["label"]:
            b["label"] = c
            n_alias += 1
    if n_alias:
        print(f"[harmonise] normalised {n_alias} label spellings to the pipeline's form")

    Path(args.out).write_text(json.dumps({"boxes": boxes}, indent=2))
    print(f"\n[harmonise] {n0} -> {len(boxes)} boxes")
    print(f"  {dict(Counter(b['label'] for b in boxes))}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
