#!/usr/bin/env python
"""Run the whole detection pipeline for one registered space.

Every stage exists as its own CLI; this defines the order they run in and
supplies the per-space constants each one needs. Stages:

   1. detect        open-vocabulary 2D detection over synthetic views of the
                    DEROTATED splat (axis-aligned boxes only fit axis-aligned
                    rooms)
   2. rotate        boxes back into the original splat frame by +yaw
   3. flux          drop boxes that enclose no reconstructed surface
   4. refit         extents refitted against the Gaussians
   5. align         ICP splat -> captured point cloud; discovers the true scale
   6. clip          label from the space's real photographs, via a round trip
                    through the point-cloud frame where the cameras live
   7. support       physical support prior
   8. surfaces      detect work surfaces from geometry
   9. ground        drop floating bases to the floor
  9b. masks         SAM silhouettes for compact classes
  9d. clip #2       label boxes the surface detector created (additive only)
  9f. verify        record reprojection IoU per box
  9g. harmonise     orientation, synonyms, duplicates, shelf tiers
  10. export        scene graphs in the splat-geo and point-cloud frames
  11. install       into ui/_spaces and tri-viewer

Two ordering rules matter:

  * CLIP labelling runs BEFORE the support prior. It labels from photo crops
    and has no notion of height, so it will happily call a ceiling fixture a
    trash bin; the support prior is what catches that and must run downstream.
  * Geometric evidence outranks photo-crop evidence. A label established by
    the surface detector is never overturned by CLIP — CLIP may only narrow a
    class geometry has already established.

Per-space constants are derived from the data wherever possible. The ICP scale
guess comes from the ratio of vertical extents, which is the one axis a room
yaw cannot change, and the detector's size cap is converted from a metre
target using that scale.

Usage:
  python run_space_pipeline.py --space shinhan_space
  python run_space_pipeline.py --space factory_space_13 --skip-detect --skip-align
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from plyfile import PlyData

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
OUT = REPO / "pipeline9" / "out"

# Per-space constants that cannot be derived from the data.
#   splat        : the ORIGINAL (un-derotated) splat; boxes live in this frame
#   yaw_deg      : this room's yaw. The derotated ply is the splat rotated by
#                  -yaw_deg; detected boxes are rotated back by +yaw_deg.
#                  Sourced from tri-viewer/scenes.json, which records the
#                  measured yaw each derotated ply was built with.
#   tri_frame    : which frame this scene's tri-viewer .ksplat is in, i.e.
#                  whichever ply it was built from (see tri-viewer/scenes.json).
#                  Boxes are produced in the original splat frame, so a scene
#                  whose ksplat is derotated needs them rotated by -yaw on
#                  install, or they sit at the room's yaw offset from their
#                  objects.
#   viewer_scale : metres-per-native-unit that this space's ui point cloud
#                  (ui/_spaces/<...>/Data_) was generated at. The splat-frame
#                  scene graph must be exported at this value to line up with
#                  that cloud, whatever the true metric scale is.
SPACES = {
    "shinhan_space": {
        # Chairs in this room cannot be bounded reliably: one pulled up to a
        # desk is seen against the desk from nearly every angle, so the box
        # covers both and any labeller then reasonably calls it a table.
        # Dropping the class is cleaner than carrying boxes we can neither
        # bound nor name.
        "drop_labels": "chair,office_chair",
        "max_long_m": 2.1,
        "surface_label": "table",
        "tri_frame": "original",
        "splat": "data/shinhan_hires_30k.ply",
        "derot": "data/shinhan_hires_30k_derotated.ply",
        "yaw_deg": 28.072,
        "viewer_scale": 4.94,
        "tri_scene": "shinhan_space",
        "pc_space": "shinhan_owlv2_pointcloud_scenegraph",
        "splat_space": "shinhan_space_splatanalyzer",
    },
    "factory_space_13": {
        "max_long_m": 7.0,
        "surface_label": "workbench",
        "tri_frame": "derotated",
        "splat": "data/factory13_detailed.ply",
        "derot": "data/factory13_detailed_derotated.ply",
        "yaw_deg": 8.521,
        "viewer_scale": None,
        "tri_scene": "factory13_detailed",
        "pc_space": "factory13detailed_owlv2_pointcloud_scenegraph",
        "splat_space": "factory13_detailed_splatanalyzer",
    },
    "factory_space_14": {
        "max_long_m": 7.0,
        "surface_label": "workbench",
        "tri_frame": "derotated",
        "splat": "data/factory14_detail.ply",
        "derot": "data/factory14_detail_derotated.ply",
        "yaw_deg": 13.639,
        "viewer_scale": None,
        "tri_scene": "factory14_detail",
        "pc_space": "factory14_owlv2_pointcloud_scenegraph",
        "splat_space": "factory14_detail_splatanalyzer",
    },
    "factory_space_15": {
        "max_long_m": 7.0,
        "surface_label": "workbench",
        "tri_frame": "derotated",
        "splat": "data/factory15_sharp.ply",
        "derot": "data/factory15_sharp_derotated.ply",
        "yaw_deg": 3.9,
        "viewer_scale": None,
        "tri_scene": "factory15_sharp",
        "pc_space": "factory15sharp_owlv2_pointcloud_scenegraph",
        "splat_space": "factory15_sharp_splatanalyzer",
    },
}

# ── infrascan integration shim ────────────────────────────────────────────────
# ONLY change made to this upstream file (see train/scenegraph/NOTES.md).
# Additive + backward-compatible: when the env var SG_SPACES_JSON points at a
# JSON file, its entries are merged into SPACES so the RunPod endpoint can
# register an S3-sourced space at runtime WITHOUT hand-editing this dict.
# Upstream behaviour is completely unchanged when SG_SPACES_JSON is unset.
import os as _sg_os  # noqa: E402
_sg_ext = _sg_os.environ.get("SG_SPACES_JSON")
if _sg_ext and Path(_sg_ext).exists():
    SPACES.update(json.loads(Path(_sg_ext).read_text()))
# ── end infrascan shim ────────────────────────────────────────────────────────

# Horizontal work-surface classes. A box the surface detector created is one of
# these by construction, so a CLIP answer outside this set is rejected rather
# than applied.
SURFACE_CLASSES = {"table", "desk", "workbench", "bench", "counter", "workstation"}

# The detection vocabulary. Anything absent here cannot be detected at all, so a
# later relabelling pass can only rename a box that some listed class already
# won - it can never create one. Keep it in sync with the runs you are comparing
# against: this is the 23-label list every result in the repo was produced with.
PROMPT = ("chair, table, desk, workbench, shelf, storage rack, cabinet, "
          "cardboard box, pallet, cart, machine, trash bin, whiteboard, bench, "
          "ladder, light, window, door, fire extinguisher, printer, plant, "
          "monitor, partition panel")


def run(cmd, cwd=REPO, label=""):
    t0 = time.time()
    print(f"\n=== {label} ===\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd))
    if r.returncode != 0:
        raise SystemExit(f"[{label}] failed with exit {r.returncode}")
    print(f"--- {label} done in {time.time() - t0:.0f}s", flush=True)


def vertical_span(path, pct=(1, 99), axis=2):
    p = PlyData.read(str(path))["vertex"]
    v = np.asarray(p["z" if axis == 2 else "y"], dtype=np.float64)
    lo, hi = np.percentile(v, pct)
    return float(hi - lo)


def scale_guess(splat_ply, pc_ply):
    """Metres-per-native-unit, from the VERTICAL extents.

    A room yaw is a rotation about the vertical axis, so it cannot change that
    axis's extent — which makes the vertical ratio the one comparison that is
    valid without first knowing the yaw. Comparing horizontal axis-aligned
    bounding boxes instead gives an inflated answer whenever the two clouds
    are rotated relative to each other (it read 7.57 vs a true 6.80 on
    shinhan). The splats here are Z-up and the captured clouds Y-up.
    """
    return vertical_span(pc_ply, axis=1) / max(vertical_span(splat_ply, axis=2), 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, choices=sorted(SPACES))
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--n-positions", type=int, default=100)
    ap.add_argument("--n-azimuth", type=int, default=12)
    ap.add_argument("--n-elevation", type=int, default=4)
    ap.add_argument("--max-object-diag-m", type=float, default=3.5,
                    help="largest plausible object diagonal in METRES; converted to "
                         "the detector's native-unit cap using this space's own scale")
    ap.add_argument("--skip-detect", action="store_true",
                    help="reuse an existing job dir / raw_detections.json")
    ap.add_argument("--skip-align", action="store_true",
                    help="reuse an existing transform json (ICP takes ~15 min)")
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--skip-masks", action="store_true")
    ap.add_argument("--skip-relabel2", action="store_true",
                    help="skip the second CLIP pass; without it, boxes created by the "
                         "topdown surface detector keep a hardcoded label")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--skip-harmonise", action="store_true")
    ap.add_argument("--mask-labels",
                    default="office_chair,chair,plant,trash_bin,computer_monitor,"
                            "cardboard_box,pallet,fire_extinguisher,person",
                    help="classes the SAM mask refit is applied to — compact objects "
                         "only, never horizontal surfaces")
    ap.add_argument("--no-install", action="store_true")
    args = ap.parse_args()

    S = SPACES[args.space]
    sp = args.space
    splat = REPO / S["splat"]
    derot = REPO / S["derot"]
    pc = REPO / "data" / sp / "pointcloud.ply"
    for f in (splat, derot, pc):
        if not f.exists():
            raise SystemExit(f"missing input: {f}")
    OUT.mkdir(parents=True, exist_ok=True)

    job = REPO / "external" / "splat_analyzer" / f"out_{sp}"
    B = lambda tag: OUT / f"{sp}_{tag}.json"  # noqa: E731

    print(f"[run] space={sp}\n[run] splat={S['splat']}  yaw={S['yaw_deg']}°\n[run] pointcloud={pc}")

    # The scale guess is needed BEFORE detection, not just for ICP:
    # --max_object_diag is a size cap in NATIVE units, so a fixed value means a
    # different real-world size in every space. tri-viewer/scenes.json records
    # factory15 needing it raised from 0.5 to 0.70 for exactly this reason
    # ("to match this splat's lower scale_to_meters, ~4.95"). Deriving it from
    # a real metre target removes the per-space hand-tuning.
    guess = scale_guess(splat, pc)
    max_diag = args.max_object_diag_m / guess
    print(f"[run] scale guess (vertical extents) = {guess:.3f} m/native-unit")
    print(f"[run] max_object_diag = {max_diag:.3f} native "
          f"(= {args.max_object_diag_m} m target)")

    # ---- 1. detection on the derotated splat -----------------------------
    if not args.skip_detect or not (job / "interactions.json").exists():
        run([PY, "run_local.py", "--ply", derot, "--prompt", args.prompt,
             "--n_positions", args.n_positions, "--n_azimuth", args.n_azimuth,
             "--n_elevation", args.n_elevation,
             "--score_threshold", 0.10, "--min_vote_frac", 0.026,
             "--min_peak_score", 0.35, "--max_per_label", 80,
             "--max_object_diag", round(max_diag, 4), "--job_dir", job.name],
            cwd=REPO / "external" / "splat_analyzer", label="1 detect")

    # ---- 2-4. rotate back, flux filter, extent refit ----------------------
    run([PY, "external/splat_analyzer/rotate_and_export.py",
         "--interactions", job / "interactions.json",
         "--yaw-deg", S["yaw_deg"], "--out", B("boxes_raw")], label="2 rotate")
    run([PY, "pipeline9/closed_surface_flux.py", "--ply", splat,
         "--boxes", B("boxes_raw"), "--min-gaussians", 50,
         "--filtered-out", B("boxes_flux")], label="3 flux")
    run([PY, "pipeline9/refit_box_extent.py", "--ply", splat,
         "--boxes", B("boxes_flux"), "--out", B("boxes_refit")], label="4 refit")

    # ---- 5. ICP alignment; discovers the TRUE metric scale ----------------
    tf = OUT / f"{sp}_splat_to_pc_transform.json"
    if not (args.skip_align and tf.exists()):
        run([PY, "pipeline9/align_splat_to_pointcloud.py", "--splat-ply", splat,
             "--pointcloud-ply", pc, "--scale-to-meters-guess", round(guess, 3),
             "--out", tf], label="5 align")
    true_scale = float(json.loads(tf.read_text())["true_scale_to_meters"])
    viewer_scale = S["viewer_scale"] or true_scale
    print(f"[run] true scale={true_scale}  viewer scale={viewer_scale}")

    cur = B("boxes_refit")

    def clip_relabel(in_boxes, tag, protect_geometric=False, only_new_surfaces=False):
        """Label boxes from the space's REAL photographs.

        Needs a round trip: relabel_with_clip.py works on a scene graph in the
        point cloud's frame (that is where the cameras live), so the boxes are
        exported, transformed, labelled, and the labels mapped back onto the
        splat-frame boxes by node id — which export_scene_graph_for_point_viewer
        sets to the box's list index.
        """
        geo, pc_sg, final = B(f"geo_{tag}"), B(f"pc_{tag}"), B(f"clip_{tag}")
        run([PY, "pipeline9/export_scene_graph_for_point_viewer.py", "--ply", splat,
             "--yaw-deg", S["yaw_deg"], "--scale-to-meters", true_scale,
             "--boxes", in_boxes, "--space", S["splat_space"], "--out", geo],
            label=f"{tag} export")
        run([PY, "pipeline9/apply_scenegraph_to_pointcloud.py", "--scene-graph", geo,
             "--transform", tf, "--source-scale-to-meters", true_scale,
             "--space", S["pc_space"], "--pointcloud-ply", pc, "--out", pc_sg],
            label=f"{tag} to pointcloud")
        run([PY, "pipeline9/relabel_with_clip.py", "--scene-graph", pc_sg,
             "--pointcloud-ply", pc, "--space", sp,
             "--geo-json-out", B(f"relabel_geo_{tag}"), "--out", final],
            label=f"{tag} clip")
        boxes_in = json.loads(Path(in_boxes).read_text())["boxes"]
        lab = {n["id"]: n["label"] for n in json.loads(final.read_text())["nodes"]}
        kept, protected, untouched = [], 0, 0
        for i, b in enumerate(boxes_in):
            # only_new_surfaces: this pass exists ONLY to label boxes the surface
            # detector created after pass 1, which have never been seen by a
            # photograph. Letting it relabel everything else is net harm — on
            # shinhan_space it took 96 boxes to 92, lost one of the two plants to
            # a "structural" flag and renamed a table to "person". Every other box
            # already has a pass-1 label from the same model on the same photos;
            # asking again just adds variance.
            if only_new_surfaces:
                is_new_surface = (b.get("source") == "topdown_surface"
                                  and "surface_confirmed" not in b
                                  and "relabelled_from" not in b)
                if not is_new_surface:
                    untouched += 1
                    kept.append(b)
                    continue
            if i not in lab:
                if only_new_surfaces:
                    kept.append(b)      # never DROP in this mode
                continue
            # A label backed by direct geometric evidence outranks one read off a
            # photo crop. detect_tables_topdown.py identifies a surface by its
            # height and footprint; CLIP is guessing from pixels and is measurably
            # bad at desk-vs-shelf — on the first attempt at this second pass it
            # renamed 17 of shinhan's 19 detected desks into shelf, office_chair,
            # whiteboard and even "person", taking the space from 20 tables to 3.
            # Boxes the surface detector labelled are therefore frozen here;
            # genuinely NEW surface boxes (no prior label) still take a CLIP label,
            # which is what stops factory floors reporting workbenches as tables.
            if protect_geometric and b.get("source") == "topdown_surface":
                # Already carries a decided label (an existing surface class kept,
                # or a non-surface class corrected): frozen outright.
                if "surface_confirmed" in b or "relabelled_from" in b:
                    protected += 1
                    kept.append(b)
                    continue
                # Genuinely new surface box: geometry knows it IS a horizontal
                # work surface but not which one. Take CLIP's answer only if it is
                # itself a surface class, otherwise keep the space's own default.
                # Without this, 15 of shinhan's 19 desks were "new" and CLIP
                # renamed them to shelf/whiteboard/person anyway.
                if lab[i] not in SURFACE_CLASSES:
                    b["clip_rejected"] = lab[i]
                    protected += 1
                    kept.append(b)
                    continue
            if b.get("label") != lab[i]:
                b["clip_relabelled_from"] = b.get("label")
            b["label"] = lab[i]
            kept.append(b)
        if protected:
            print(f"[run] {tag}: {protected} geometrically-labelled surface boxes protected")
        if only_new_surfaces:
            print(f"[run] {tag}: additive-only — {untouched} boxes left untouched, "
                  f"only newly-created surfaces relabelled")
        out = B(f"boxes_{tag}")
        out.write_text(json.dumps({"boxes": kept}, indent=2))
        print(f"[run] {tag}: clip kept {len(kept)}/{len(boxes_in)} -> "
              f"{dict(Counter(b['label'] for b in kept))}")
        return out

    # ---- 6. CLIP relabel, via a round trip through the point-cloud frame --
    if not args.skip_clip:
        cur = clip_relabel(cur, "pass1")

    # ---- 7-9. support prior, table detection, grounding -------------------
    if S.get("drop_labels"):
        drop = {x.strip() for x in S["drop_labels"].split(",") if x.strip()}
        bx = json.loads(Path(cur).read_text())["boxes"]
        keep = [b for b in bx if b["label"] not in drop]
        B("boxes_dropped").write_text(json.dumps({"boxes": keep}, indent=2))
        print(f"[run] dropped {len(bx) - len(keep)} boxes of unreliable classes "
              f"{sorted(drop)} -> {len(keep)} remain")
        cur = B("boxes_dropped")

    run([PY, "pipeline9/support_prior_filter.py", "--ply", splat, "--boxes", cur,
         "--scale-to-meters", true_scale, "--out", B("boxes_support")], label="7 support")
    run([PY, "pipeline9/detect_tables_topdown.py", "--ply", splat,
         "--boxes", B("boxes_support"), "--scale-to-meters", true_scale,
         "--room-yaw-deg", S["yaw_deg"], "--label", S.get("surface_label", "table"),
         "--min-fill", 0.58, "--min-area-m2", 0.45, "--max-aspect", 3.0,
         # A factory bench runs the length of a wall; an office desk does not.
         # The default 2.10 chopped factory_space_14's benches into 3-4 pieces
         # each (longest surviving box 1.86 m, median 1.26 m), which is the
         # "many boxes on one desk" seen in the viewer.
         "--max-long-m", S.get("max_long_m", 2.10),
         "--out", B("boxes_tables")], label="8 tables")
    run([PY, "pipeline9/ground_floor_standing_boxes.py", "--ply", splat,
         "--boxes", B("boxes_tables"), "--scale-to-meters", true_scale,
         "--out", B("boxes_grounded")], label="9 ground")

    # 9b. SAM silhouettes, for COMPACT classes only. A chair at a desk is seen
    # against the desk from nearly every angle, so its box bounds both; masks
    # from the views that see its free side fix that (0.80x0.99 -> 0.64x0.87 on
    # shinhan). Surfaces are excluded on purpose: an obliquely-viewed tabletop
    # gives a partial silhouette and masks shrink it wrongly (1.53 -> 1.02),
    # where detect_tables_topdown.py measures it correctly.
    if not args.skip_masks:
        run([PY, "pipeline9/refit_box_from_masks.py", "--job-dir", job,
             "--boxes", B("boxes_grounded"), "--yaw-deg", S["yaw_deg"],
             "--scale-to-meters", true_scale, "--n-frames", 8,
             "--only-labels", args.mask_labels,
             "--out", B("boxes_masked")], label="9b masks")
        run([PY, "pipeline9/ground_floor_standing_boxes.py", "--ply", splat,
             "--boxes", B("boxes_masked"), "--scale-to-meters", true_scale,
             "--out", B("boxes_final")], label="9c ground again")
    else:
        B("boxes_final").write_text(B("boxes_grounded").read_text())

    # 9d. SECOND CLIP pass. The first one (step 6) runs before the topdown surface
    # detector, so every box that detector CREATES has never been labelled from a
    # photograph — they carry the hardcoded --label, which is how factory_space_13
    # reported 27 "tables" on a floor of workbenches. Labelling again here covers
    # those, and covers every box whose geometry the mask refit has since changed.
    # The support prior is then re-applied, because CLIP labels from crops with no
    # notion of height and will happily call a ceiling fixture a trash bin.
    if not (args.skip_clip or args.skip_relabel2):
        relabelled = clip_relabel(B("boxes_final"), "pass2", protect_geometric=True,
                                  only_new_surfaces=True)
        B("boxes_final").write_text(relabelled.read_text())

    # 9f. reprojection IoU, recorded on every box but not used to drop anything —
    # the support prior and the surface detector already remove what they can
    # justify, and this is a diagnostic until it has been calibrated per class.
    if not args.skip_verify:
        run([PY, "pipeline9/verify_boxes_render_back.py", "--ply", derot,
             "--job-dir", job, "--boxes", B("boxes_final"),
             "--yaw-deg", S["yaw_deg"], "--out", B("boxes_final")],
            label="9f verify (annotate)")

    # 9g. Reconcile what the earlier passes left inconsistent: three of them set
    # `angle` independently (yaw mod 90 had std 31.7 deg on factory_space_13),
    # surface synonyms were split across table/workbench per CLIP crop, shelf
    # tiers were claimed as workbenches, and redundant boxes overlapped ones that
    # already bound their object.
    if not args.skip_harmonise:
        run([PY, "pipeline9/harmonize_scene.py", "--boxes", B("boxes_final"),
             "--ply", splat, "--scale-to-meters", true_scale,
             "--room-yaw-deg", S["yaw_deg"],
             "--collapse-surfaces", S.get("surface_label", "table"),
             "--out", B("boxes_final")], label="9g harmonise")

    # ---- 10. scene graphs in both frames ----------------------------------
    run([PY, "pipeline9/export_scene_graph_for_point_viewer.py", "--ply", splat,
         "--yaw-deg", S["yaw_deg"], "--scale-to-meters", true_scale,
         "--boxes", B("boxes_final"), "--space", S["splat_space"],
         "--out", B("geo_true")], label="10a export (true scale)")
    run([PY, "pipeline9/apply_scenegraph_to_pointcloud.py",
         "--scene-graph", B("geo_true"), "--transform", tf,
         "--source-scale-to-meters", true_scale, "--space", S["pc_space"],
         "--pointcloud-ply", pc, "--out", B("sg_pointcloud")], label="10b to pointcloud")
    run([PY, "pipeline9/export_scene_graph_for_point_viewer.py", "--ply", splat,
         "--yaw-deg", S["yaw_deg"], "--scale-to-meters", viewer_scale,
         "--boxes", B("boxes_final"), "--space", S["splat_space"],
         "--out", B("sg_splat_viewer")], label="10c export (viewer scale)")

    # ---- 11. install -------------------------------------------------------
    if not args.no_install:
        boxes = json.loads(B("boxes_final").read_text())["boxes"]
        if S.get("tri_frame") == "derotated":
            th = -np.radians(S["yaw_deg"])
            c, sn = np.cos(th), np.sin(th)
            R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
            for b in boxes:
                b["center"] = [float(v) for v in R @ np.asarray(b["center"])]
                b["angle"] = float(b.get("angle", 0.0) + th)
            print(f"[run] tri-viewer scene is in the DEROTATED frame — "
                  f"boxes rotated by {np.degrees(th):.3f}°")
        n = Counter()
        for b in boxes:
            n[b["label"]] += 1
            b["id"] = n[b["label"]]          # tri-viewer's stable per-label id
        tri = REPO / "tri-viewer" / "modes" / "threed" / "scene" / f"{S['tri_scene']}.boxes.json"
        if tri.exists():
            tri.with_suffix(".json.bak").write_text(tri.read_text())
        tri.write_text(json.dumps({"boxes": boxes}, indent=2))
        print(f"[run] tri-viewer <- {tri.name}: {dict(n)}")

        for src, space in ((B("sg_pointcloud"), S["pc_space"]),
                           (B("sg_splat_viewer"), S["splat_space"])):
            d = REPO / "ui" / "_spaces" / space
            if not d.exists():
                print(f"[run] NOTE: ui/_spaces/{space} does not exist — skipping install "
                      f"(scene graph is still at {src})")
                continue
            tgt = d / "scene_graph.json"
            if tgt.exists():
                (d / "scene_graph.json.bak").write_text(tgt.read_text())
            tgt.write_text(src.read_text())
            print(f"[run] ui/_spaces/{space} <- "
                  f"{len(json.loads(src.read_text())['nodes'])} nodes")

    print(f"\n[run] {sp} COMPLETE  ->  {B('boxes_final')}")


if __name__ == "__main__":
    main()
