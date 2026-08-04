"""
Master pipeline:
  1. Render spherical camera views from multiple positions (render_cameras.py)
  2. Run OWLv2 open-vocabulary detection on every frame
  3. Back-project 2D boxes to 3D using per-pixel depth maps
  4. Cluster per-label detections → single 3D bounding box per object
  5. Write interactions.json in WebXR format

Usage:
  python pipeline.py --ply scene.ply --prompt "chair, desk" --job_dir /tmp/job123
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

# Let OWLv2 ops not yet implemented on Apple MPS fall back to CPU instead of hard-erroring.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import cv2
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

import render_cameras
from config import PipelineConfig, QUALITY_PRESETS, DEFAULT_QUALITY


def _select_device() -> str:
    """Compute device for OWLv2. Order: CUDA → Apple MPS → CPU.
    Set WMD_DEVICE=cuda|mps|cpu to override (e.g. WMD_DEVICE=cpu if MPS misbehaves)."""
    override = os.getenv("WMD_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# 2-D → 3-D lifting helpers
# ---------------------------------------------------------------------------

def _unproject_box(box_2d, depth, K_inv, c2w):
    """
    Unproject a 2-D bounding box centre to a 3-D ray and place it at `depth`
    along that ray in world space.
    """
    x1, y1, x2, y2 = box_2d
    cx_px = (x1 + x2) / 2.0
    cy_px = (y1 + y2) / 2.0
    w_px = x2 - x1
    h_px = y2 - y1

    p_cam_norm = K_inv @ np.array([cx_px, cy_px, 1.0])
    ray_cam = p_cam_norm / np.linalg.norm(p_cam_norm)
    point_cam = ray_cam * (depth / ray_cam[2])
    point_world = (c2w[:3, :3] @ point_cam) + c2w[:3, 3]

    return point_world, (w_px, h_px)


def _pixel_size_to_world(w_px, h_px, depth, fl_x, fl_y):
    world_w = (w_px / fl_x) * depth
    world_h = (h_px / fl_y) * depth
    return world_w, world_h


def _world_axis_extent(w_px, h_px, depth, fl_x, fl_y, c2w, depth_extent=None):
    """Convert a 2-D box's pixel extent into a WORLD-AXIS-ALIGNED (x, y, z) size.

    `_pixel_size_to_world` returns extents along the CAMERA's own image axes —
    "width" is along the camera's right vector, "height" along its down vector.
    Writing those straight into `scale = [x, y, z]` as world XYZ is only correct
    for a camera that happens to be axis-aligned. For a camera looking along +X,
    its image width lies along world Y and its image height along world Z, so the
    object's width would be recorded as an X extent. Since the pipeline sweeps a
    full panorama from every position, each detection is mis-assigned a different
    way and the per-cluster median across viewpoints averages the error instead of
    cancelling it — which is what makes box sizes not match the real object.

    The correct conversion projects the camera-frame extent vector onto the
    world axes. Taking |·| of the rotated basis vectors gives the axis-aligned
    bounding box of the (rotated) camera-frame box, which is the tightest
    world-axis-aligned box that still contains it.

    `depth_extent` is the object's thickness ALONG the view ray. A single view
    genuinely cannot observe it, so when it is unknown the smaller of the two
    measured extents is used rather than their mean: assuming an object is at
    least as deep as it is wide systematically inflates every box, and the
    cluster stage's `max_object_diag` check then rejects real large furniture.
    """
    ex_cam = (w_px / fl_x) * depth      # along camera right (+X_cam)
    ey_cam = (h_px / fl_y) * depth      # along camera down  (+Y_cam)
    ez_cam = depth_extent if depth_extent is not None else min(ex_cam, ey_cam)

    R = c2w[:3, :3]                     # camera axes expressed in world coords
    extent = (np.abs(R[:, 0]) * ex_cam
              + np.abs(R[:, 1]) * ey_cam
              + np.abs(R[:, 2]) * ez_cam)
    return extent                       # (3,) world-axis-aligned size


def _foreground_depth(depth_map, alpha_map, box_2d, alpha_thresh=0.5,
                      fg_pct=25.0, min_valid_px=12):
    """Estimate the depth of the OBJECT inside a 2-D box.

    The previous approach sampled a 5x5 patch at the box CENTRE. For most real
    furniture the box centre is not on the object: it is the gap between a
    chair's legs, the space under a table, or the opening of a shelf — so the
    sampled depth is the wall metres behind, and the detection gets lifted to a
    position well beyond the object it came from.

    Instead, take a low percentile of the valid depths across the box interior.
    The object that caused the detection is by definition the nearest
    substantial surface in that region, so the foreground sits in the low tail
    of the box's depth distribution while background occupies the high tail.
    A percentile (not the minimum) keeps this robust to a few stray near-camera
    floaters.

    Pixels are additionally required to have accumulated alpha above
    `alpha_thresh`: a low-alpha pixel is semi-transparent haze rather than
    reconstructed surface, and its depth — even correctly alpha-normalized — is
    an average over whatever sparse Gaussians happen to lie along the ray.

    Returns (depth, n_valid, coverage_frac). `depth` is 0.0 when the box has no
    usable surface in it at all, which is the caller's signal that the
    detection is on empty space and should be dropped rather than placed at a
    fabricated fallback distance.
    """
    H, W = depth_map.shape
    x1, y1, x2, y2 = box_2d
    # Shrink 15% toward the centre: OWLv2 boxes routinely include a margin of
    # background, and the border ring is where it lives.
    mx, my = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
    ix0 = int(np.clip(np.floor(x1 + mx), 0, W - 1))
    ix1 = int(np.clip(np.ceil(x2 - mx), 1, W))
    iy0 = int(np.clip(np.floor(y1 + my), 0, H - 1))
    iy1 = int(np.clip(np.ceil(y2 - my), 1, H))
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0, 0, 0.0

    d = depth_map[iy0:iy1, ix0:ix1]
    ok = d > 0.01
    if alpha_map is not None:
        ok &= alpha_map[iy0:iy1, ix0:ix1] > alpha_thresh

    n_valid = int(ok.sum())
    coverage = n_valid / float(d.size)
    if n_valid < min_valid_px:
        return 0.0, n_valid, coverage
    return float(np.percentile(d[ok], fg_pct)), n_valid, coverage


def _box_depth_extent(depth_map, alpha_map, box_2d, alpha_thresh=0.5,
                      lo_pct=10.0, hi_pct=85.0, min_valid_px=12):
    """Measure an object's thickness ALONG the view ray from the spread of the
    depths inside its 2-D box.

    A single view cannot observe the far side of an object, so this is a lower
    bound, not a true depth — but it is a MEASUREMENT rather than the previous
    `(world_w + world_h) / 2` guess, which had no relationship to the scene at
    all. The upper percentile is capped well below 100 so that background
    visible past the object's silhouette (through a chair's legs, over a
    desk's edge) does not stretch the extent out to the far wall.

    Returns None when the box has too little surface to measure, letting
    `_world_axis_extent` fall back to its own conservative estimate.
    """
    H, W = depth_map.shape
    x1, y1, x2, y2 = box_2d
    ix0 = int(np.clip(np.floor(x1), 0, W - 1)); ix1 = int(np.clip(np.ceil(x2), 1, W))
    iy0 = int(np.clip(np.floor(y1), 0, H - 1)); iy1 = int(np.clip(np.ceil(y2), 1, H))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    d = depth_map[iy0:iy1, ix0:ix1]
    ok = d > 0.01
    if alpha_map is not None:
        ok &= alpha_map[iy0:iy1, ix0:ix1] > alpha_thresh
    if int(ok.sum()) < min_valid_px:
        return None
    vals = d[ok]
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    return float(max(hi - lo, 0.0)) or None


def _sharpness_cut(transforms, cfg):
    """Resolve the run-relative sharpness threshold.

    Sharpness is gated relatively (bottom N% of THIS run's frames) rather than
    against a constant, because the metric's absolute level depends on
    resolution, FoV and how textured the scene is — see
    PipelineConfig.frame_sharpness_pct.
    """
    vals = [f["quality"]["sharpness"] for f in transforms["frames"]
            if f.get("quality") is not None]
    cut = 0.0
    if vals and cfg.frame_sharpness_pct:
        cut = float(np.percentile(vals, cfg.frame_sharpness_pct))
    if cfg.min_frame_sharpness is not None:
        cut = max(cut, cfg.min_frame_sharpness)
    if vals:
        print(f"[pipeline] sharpness: p5/p50/p95 = {np.percentile(vals,5):.1f}/"
              f"{np.percentile(vals,50):.1f}/{np.percentile(vals,95):.1f}  "
              f"-> cut at {cut:.1f} (bottom {cfg.frame_sharpness_pct:.0f}%)")
    return cut


def _frame_passes_gate(fq, cfg, bbox_diag, sharpness_cut):
    """True if a rendered frame is good enough to run detection on.

    Returns (ok, reason). See PipelineConfig's frame-quality-gate block for why
    each threshold exists and how it was calibrated.
    """
    if fq is None:
        return True, ""                      # older transforms.json, no metrics
    if fq["sharpness"] < sharpness_cut:
        return False, f"blur(sharp={fq['sharpness']:.1f})"
    if fq["alpha_frac"] < cfg.min_frame_alpha_frac:
        return False, f"empty(alpha_frac={fq['alpha_frac']:.2f})"
    if fq["median_depth"] < cfg.min_frame_median_depth_frac * bbox_diag:
        return False, f"buried(med_depth={fq['median_depth']:.4f})"
    return True, ""


HEIGHT_EXEMPT_LABELS = {"light", "window"}   # legitimately can sit near ceiling height


def _cluster_detections(detections, eps_m=0.5, max_per_label=3,
                        min_votes=8, min_peak_score=0.35, max_object_diag=None,
                        max_height_z=None, min_height_z_light=None, label_overrides=None,
                        label_eps_scale=None, min_object_extent=0.01):
    """
    Anchor-based greedy clustering with false-positive suppression.

    The seed detection's position is used as a FIXED anchor — no centroid drift.
    Drift was causing two nearby same-label objects (e.g. two sofas) to bleed
    into each other's clusters because the shifting centroid would migrate toward
    the second object and absorb its detections.

    max_object_diag drops any raw detection whose OWN implied real-world size
    is already implausible for furniture BEFORE it can seed or vote into a
    cluster, plus a second check on the final cluster scale as a safety net.
    The size cap matters because a single OWLv2 box that mis-reads a large
    blurry region as "door"/"window"/"cabinet" converts, via pixel-size-at-
    depth, into a real-world box several metres across. Unchecked, such
    clusters end up spanning most of the room.

    max_height_z drops any raw detection above this world-Z whose label isn't
    in HEIGHT_EXEMPT_LABELS. With enough camera coverage, "table" detections
    split cleanly by height: most near ceiling height, right where "light"
    clusters (OWLv2 reading ceiling content as a tabletop), and a minority at
    real desk height. That is a specific, filterable confusion, not noise.

    min_height_z_light drops any "light" detection BELOW this world-Z.
    Exempting "light" from max_height_z (it can legitimately be near
    ceiling) is not the same as "light" being plausible at ANY height —
    confirmed directly: 32% of raw "light" detections on a real run were
    at desk/floor height, visibly wrong boxes hovering over desks in the
    viewer once max_per_label was raised enough to stop hiding them.

    label_overrides: optional {label: {"min_votes": int, "min_peak_score":
    float}} to loosen/tighten SPECIFIC labels independently of the global
    thresholds above — added after the user explicitly asked to see every
    candidate for one label (even ones likely to be false positives)
    without touching another label's already-good result. Global
    min_votes/min_peak_score still apply to any label not listed here.

    label_eps_scale: optional {label: multiplier} on the merge radius `eps_m`.
    A single radius shared by every label is a physical-size prior that is
    wrong for all but one of them — see PipelineConfig.cluster_eps_frac for the
    measurement, and GaussianDet3D's NMS-threshold ablation for the same effect
    quantified on a benchmark (over-wide spatial suppression costs small
    objects almost exclusively).
    """
    if max_object_diag is not None:
        detections = [d for d in detections if float(np.linalg.norm(d["scale"])) <= max_object_diag]
    if max_height_z is not None:
        detections = [d for d in detections
                      if d["label"] in HEIGHT_EXEMPT_LABELS or d["position"][2] <= max_height_z]
    if min_height_z_light is not None:
        detections = [d for d in detections
                      if d["label"] != "light" or d["position"][2] >= min_height_z_light]
    by_label = defaultdict(list)
    for det in detections:
        by_label[det["label"]].append(det)

    results = []
    for label, dets in by_label.items():
        lbl_eps = eps_m * float((label_eps_scale or {}).get(label, 1.0))
        dets = sorted(dets, key=lambda d: d["score"], reverse=True)
        positions = np.array([d["position"] for d in dets])
        scales    = np.array([d["scale"]    for d in dets])
        scores    = np.array([d["score"]    for d in dets])

        used = [False] * len(dets)
        clusters = []
        for i in range(len(dets)):
            if used[i]:
                continue
            # Fixed anchor — never updated. Prevents cluster from drifting
            # toward a neighbouring object and stealing its detections.
            anchor = positions[i].copy()
            cluster_idx = [i]
            for j in range(i + 1, len(dets)):
                if not used[j] and np.linalg.norm(anchor - positions[j]) < lbl_eps:
                    cluster_idx.append(j)
            for j in cluster_idx:
                used[j] = True

            peak_score = float(scores[cluster_idx].max())
            vote_count = len(cluster_idx)

            lbl_min_votes = min_votes
            lbl_min_peak_score = min_peak_score
            if label_overrides and label in label_overrides:
                lbl_min_votes = label_overrides[label].get("min_votes", min_votes)
                lbl_min_peak_score = label_overrides[label].get("min_peak_score", min_peak_score)
            if vote_count < lbl_min_votes or peak_score < lbl_min_peak_score:
                continue

            cluster_pos   = (positions[cluster_idx] * scores[cluster_idx, None]).sum(0) / scores[cluster_idx].sum()
            cluster_scale = np.median(scales[cluster_idx], axis=0)
            # Degeneracy guard only — keep it far below any real object size.
            # At 0.1 native units (~0.68 m per axis here) it stops being a
            # floor and becomes a minimum object size larger than much of what
            # is being detected: it pins every "light" cluster to exactly the
            # clamp value and drags the scene's median z-extent with it.
            cluster_scale = np.maximum(cluster_scale, min_object_extent)
            if max_object_diag is not None and float(np.linalg.norm(cluster_scale)) > max_object_diag:
                continue
            # Carry the raw member dicts so callers can trace back to source frames
            member_dets = [dets[j] for j in cluster_idx]
            clusters.append((peak_score, vote_count, label, cluster_pos, cluster_scale, member_dets))

        clusters.sort(key=lambda c: c[0], reverse=True)
        for peak, votes, lbl, pos, scale, members in clusters[:max_per_label]:
            print(f"  [cluster] {lbl}: {votes} votes, peak={peak:.2f}, pos={pos.round(2)}")
            results.append((lbl, pos, scale, members))

    return results


def _dedup_cross_label(results, overlap_frac=0.3):
    """
    Suppresses one of a pair of DIFFERENT-label detections when their 3D
    boxes substantially overlap. "table" and "chair" clusters overlapping
    57-79% of the smaller box's volume are not adjacent furniture (a chair
    pulled up to a desk) but one physical object — an integrated desk+chair
    unit — kept under two competing labels. Whichever has the higher peak
    score survives.

    The 0.3 default rather than 0.5 also covers the case where the redundant
    box is badly oversized: a "door" detection actually looking at a plant
    reached only 31% overlap with the correctly sized "plant" box, so at a
    0.5 cutoff it survived dedup and appeared as a wrong, redundant label
    to the correct one. (Tried a "does one box's center fall inside the
    other" check as an alternative/additional signal first — didn't
    actually catch this specific case, since the two boxes overlap on a
    corner rather than one containing the other's center — so this is
    just a lower overlap_frac, not a smarter geometric test.)

    Same-label overlaps are left alone — max_per_label / the anchor-based
    clustering already handles those.
    """
    def box_overlap(pos1, scale1, pos2, scale2):
        lo1, hi1 = pos1 - scale1 / 2, pos1 + scale1 / 2
        lo2, hi2 = pos2 - scale2 / 2, pos2 + scale2 / 2
        inter = np.maximum(0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2))
        inter_vol = float(np.prod(inter))
        v1, v2 = float(np.prod(scale1)), float(np.prod(scale2))
        return inter_vol / max(min(v1, v2), 1e-9)

    peak_scores = [max(m["score"] for m in members) for _, _, _, members in results]
    order = sorted(range(len(results)), key=lambda i: peak_scores[i], reverse=True)
    keep = [True] * len(results)
    dropped = []
    for a in range(len(order)):
        i = order[a]
        if not keep[i]:
            continue
        for b in range(a + 1, len(order)):
            j = order[b]
            if not keep[j] or results[i][0] == results[j][0]:
                continue
            ov = box_overlap(results[i][1], results[i][2], results[j][1], results[j][2])
            if ov > overlap_frac:
                keep[j] = False
                dropped.append((results[j][0], peak_scores[j], results[i][0], peak_scores[i], ov))
    for lbl_lost, score_lost, lbl_won, score_won, ov in dropped:
        print(f"  [dedup] dropped {lbl_lost} (peak={score_lost:.2f}) — "
              f"{ov*100:.0f}% overlap with {lbl_won} (peak={score_won:.2f})")
    return [r for i, r in enumerate(results) if keep[i]]


# ---------------------------------------------------------------------------
# OWLv2 detection
# ---------------------------------------------------------------------------

def _run_owlv2(frames_dir: Path, labels: list[str], transforms: dict, scene_radius: float,
               score_threshold: float = 0.12, cfg: "PipelineConfig | None" = None):
    """
    Run OWLv2 on the rendered frames that pass the render-quality gate.

    Uses per-pixel depth + alpha maps for 3D back-projection. A detection whose
    box contains no reconstructed surface is DROPPED rather than placed at a
    fabricated fallback depth — see `_foreground_depth`.
    """
    cfg = cfg or PipelineConfig()
    device = _select_device()
    print(f"[pipeline] Loading OWLv2 on {device} …")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model     = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
    model.eval()

    fl_x = transforms["fl_x"]
    fl_y = transforms["fl_y"]
    cx   = transforms["cx"]
    cy   = transforms["cy"]
    W    = transforms["w"]
    H    = transforms["h"]

    K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1.0]], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    texts = [[f"a photo of a {lbl.strip()}" for lbl in labels]]

    raw_detections = []

    bbox_diag = float(transforms.get("bbox_diag", scene_radius * 2.0))

    n_gated, gate_reasons = 0, defaultdict(int)
    n_no_surface = 0
    sharpness_cut = _sharpness_cut(transforms, cfg) if cfg.frame_gate else 0.0

    for frame_idx, frame_meta in enumerate(transforms["frames"]):
        frame_path = frames_dir.parent / frame_meta["file_path"]
        if not frame_path.exists():
            continue

        # ── Render-quality gate ────────────────────────────────────────────
        if cfg.frame_gate:
            ok, reason = _frame_passes_gate(frame_meta.get("quality"), cfg,
                                            bbox_diag, sharpness_cut)
            if not ok:
                n_gated += 1
                gate_reasons[reason.split("(")[0]] += 1
                continue

        c2w = np.array(frame_meta["transform_matrix"], dtype=np.float64)

        # Load per-pixel depth + alpha maps
        depth_npy_path = (frames_dir.parent /
                          frame_meta.get("depth_path", "").replace(".png", ".npy"))
        depth_map = None
        if depth_npy_path.exists():
            depth_map = np.load(str(depth_npy_path)).astype(np.float64)
        if depth_map is None:
            # Without depth there is no way to place a detection in 3D except by
            # inventing a distance, which is what produced boxes floating in
            # empty space. Skip the frame instead.
            n_gated += 1
            gate_reasons["no-depth"] += 1
            continue

        alpha_map = None
        alpha_rel = frame_meta.get("alpha_path")
        if alpha_rel and (frames_dir.parent / alpha_rel).exists():
            alpha_map = np.load(str(frames_dir.parent / alpha_rel)).astype(np.float64)

        image  = Image.open(frame_path).convert("RGB")
        inputs = processor(text=texts, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[H, W]], device=device)
        results = processor.post_process_grounded_object_detection(
            outputs, threshold=score_threshold, target_sizes=target_sizes,
        )[0]

        boxes     = results["boxes"].cpu().numpy()
        scores    = results["scores"].cpu().numpy()
        label_ids = results["labels"].cpu().numpy()

        for box, score, lid in zip(boxes, scores, label_ids):
            label = labels[lid].strip()
            bx1, by1, bx2, by2 = box          # original pixel coords (kept for box_2d)

            # Depth of the OBJECT in this box, from the foreground of the box's
            # own depth distribution — not the box centre, which is frequently
            # background seen between/under the object.
            box_depth, n_valid, coverage = _foreground_depth(
                depth_map, alpha_map, box,
                alpha_thresh=cfg.frame_alpha_thresh,
                fg_pct=cfg.fg_depth_pct,
                min_valid_px=cfg.min_box_surface_px,
            )
            if box_depth <= 0.01 or coverage < cfg.min_box_surface_frac:
                # No reconstructed surface inside the box. This IS the
                # "detected an empty space" case, and it is only detectable
                # here — once lifted to 3D at a guessed distance it becomes
                # indistinguishable from a real detection.
                n_no_surface += 1
                continue

            world_pos, (w_px, h_px) = _unproject_box(box, box_depth, K_inv, c2w)
            # Depth extent along the view ray, measured from the spread of the
            # box's own depths rather than assumed from its pixel size.
            d_extent = _box_depth_extent(depth_map, alpha_map, box,
                                         cfg.frame_alpha_thresh)
            scale = _world_axis_extent(w_px, h_px, box_depth, fl_x, fl_y, c2w,
                                       depth_extent=d_extent)

            raw_detections.append({
                "label":     label,
                "score":     float(score),
                "position":  world_pos,
                "scale":     scale,
                "frame_idx": frame_idx,
                "box_2d":    [float(bx1), float(by1), float(bx2), float(by2)],
                "coverage":  coverage,
            })
            print(f"  [detect] {label} ({score:.2f}) depth={box_depth:.3f} "
                  f"cov={coverage:.2f} @ {world_pos.round(3)} {frame_path.name}")

    n_frames = len(transforms["frames"])
    if n_gated:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(gate_reasons.items()))
        print(f"[pipeline] frame gate: skipped {n_gated}/{n_frames} frames "
              f"({100*n_gated/max(n_frames,1):.1f}%) — {detail}")
    if n_no_surface:
        print(f"[pipeline] dropped {n_no_surface} detections with no reconstructed "
              f"surface inside the box (empty-space detections)")

    return raw_detections, n_frames - n_gated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(ply_path: str, prompt: str, job_dir: str,
                 cfg: PipelineConfig | None = None) -> list[dict]:
    cfg = cfg or PipelineConfig()
    job_dir = Path(job_dir)

    print("[pipeline] Step 1: Rendering camera views …")
    transforms_path = render_cameras.render_views(ply_path, str(job_dir), cfg)

    with open(transforms_path) as f:
        transforms = json.load(f)

    frames_dir = job_dir / "frames"

    cam_positions = np.array([
        frame["transform_matrix"]
        for frame in transforms["frames"]
    ])[:, :3, 3]
    scene_radius = float(np.linalg.norm(cam_positions, axis=1).mean())

    labels = [l.strip() for l in prompt.split(",") if l.strip()]
    if not labels:
        raise ValueError("prompt must contain at least one label")

    print(f"[pipeline] Step 2: Detecting {labels} in {len(transforms['frames'])} frames …")
    # Detection is by far the most expensive stage (tens of minutes for a few
    # thousand frames), while everything after it is threshold tuning that runs
    # in seconds. Cache the raw detections so re-clustering does not require
    # re-detecting. The cache is keyed on the inputs that would change what
    # OWLv2 actually returns; anything downstream of it is free to vary.
    cache_path = job_dir / "raw_detections.json"
    cache_key = {
        "labels": labels,
        "score_threshold": cfg.score_threshold,
        "n_frames": len(transforms["frames"]),
        "frame_gate": cfg.frame_gate,
        "frame_sharpness_pct": cfg.frame_sharpness_pct,
        "min_frame_sharpness": cfg.min_frame_sharpness,
        "min_frame_alpha_frac": cfg.min_frame_alpha_frac,
        "min_frame_median_depth_frac": cfg.min_frame_median_depth_frac,
        "frame_alpha_thresh": cfg.frame_alpha_thresh,
        "fg_depth_pct": cfg.fg_depth_pct,
        "min_box_surface_px": cfg.min_box_surface_px,
        "min_box_surface_frac": cfg.min_box_surface_frac,
    }
    raw_detections = n_detected_frames = None
    if cache_path.exists() and cfg.use_detection_cache:
        cached = json.load(open(cache_path))
        if cached.get("key") == cache_key:
            raw_detections = [
                {**d,
                 "position": np.array(d["position"]),
                 "scale":    np.array(d["scale"])}
                for d in cached["detections"]
            ]
            n_detected_frames = cached["n_detected_frames"]
            print(f"[pipeline] reusing {len(raw_detections)} cached raw detections "
                  f"from {cache_path} (detection inputs unchanged)")
        else:
            print(f"[pipeline] detection cache present but inputs changed — re-detecting")

    if raw_detections is None:
        raw_detections, n_detected_frames = _run_owlv2(
            frames_dir, labels, transforms, scene_radius,
            score_threshold=cfg.score_threshold, cfg=cfg)
        with open(cache_path, "w") as f:
            json.dump({
                "key": cache_key,
                "n_detected_frames": n_detected_frames,
                "detections": [
                    {**d,
                     "position": np.asarray(d["position"]).tolist(),
                     "scale":    np.asarray(d["scale"]).tolist()}
                    for d in raw_detections
                ],
            }, f)
        print(f"[pipeline] cached {len(raw_detections)} raw detections → {cache_path}")

    frame_annotations: dict = {}   # frame_idx (str) → [{label, object_idx, box, score}]

    if not raw_detections:
        print("[pipeline] No detections above threshold.")
        interactions = []
    else:
        print(f"[pipeline] Step 3: Clustering {len(raw_detections)} raw detections …")
        effective_min_votes = cfg.min_votes
        if cfg.min_vote_frac is not None:
            # Scale against the frames detection ACTUALLY ran on, not the total
            # rendered. The quality gate can remove a large fraction of a run's
            # frames, and dividing by the pre-gate count would silently raise
            # the effective bar by exactly that fraction — the same
            # selectivity drift min_vote_frac exists to prevent.
            n_views = n_detected_frames
            effective_min_votes = max(1, round(cfg.min_vote_frac * n_views))
            print(f"[pipeline] min_vote_frac={cfg.min_vote_frac} x {n_views} "
                  f"gate-passing views -> effective min_votes={effective_min_votes}")
        # Merge radius. Anchored to max_object_diag (the caller's own statement
        # of how big an object can be, in native units) rather than to
        # scene_radius, which is the size of the ROOM and has nothing to do with
        # how far apart two detections of one object land.
        if cfg.cluster_eps is not None:
            eps_m = cfg.cluster_eps
        elif cfg.max_object_diag is not None:
            eps_m = cfg.cluster_eps_frac * cfg.max_object_diag
        else:
            eps_m = transforms.get("scene_radius", scene_radius) * 0.20
        print(f"[pipeline] cluster radius = {eps_m:.4f} native units"
              + (f" (label overrides: {cfg.label_eps_scale})" if cfg.label_eps_scale else ""))

        clustered = _cluster_detections(
            raw_detections,
            eps_m=eps_m,
            max_per_label=cfg.max_per_label,
            min_votes=effective_min_votes,
            min_peak_score=cfg.min_peak_score,
            max_object_diag=cfg.max_object_diag,
            max_height_z=cfg.max_height_z,
            min_height_z_light=cfg.min_height_z_light,
            label_overrides=cfg.label_overrides,
            label_eps_scale=cfg.label_eps_scale,
            min_object_extent=cfg.min_object_extent,
        )
        n_before_dedup = len(clustered)
        clustered = _dedup_cross_label(clustered, overlap_frac=cfg.cross_label_overlap_frac)
        if len(clustered) < n_before_dedup:
            print(f"[pipeline] cross-label dedup: {n_before_dedup} -> {len(clustered)} "
                  f"({n_before_dedup - len(clustered)} dropped as same-object competing labels)")

        interactions = []
        for obj_idx, (label, pos, scale, members) in enumerate(clustered):
            # Deduplicate members by frame_idx (keep highest-score per frame)
            best: dict = {}
            for m in members:
                fi = m["frame_idx"]
                if fi not in best or m["score"] > best[fi]["score"]:
                    best[fi] = m

            obj_frames = []
            for m in sorted(best.values(), key=lambda x: x["score"], reverse=True):
                fi = m["frame_idx"]
                obj_frames.append({
                    "frame_idx": fi,
                    "box":       m["box_2d"],
                    "score":     round(m["score"], 4),
                })
                fkey = str(fi)
                frame_annotations.setdefault(fkey, []).append({
                    "label":      label,
                    "object_idx": obj_idx,
                    "box":        m["box_2d"],
                    "score":      round(m["score"], 4),
                })

            interactions.append({
                "label":    label,
                "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "scale":    {"x": float(scale[0]), "y": float(scale[1]), "z": float(scale[2])},
                "frames":   obj_frames,
            })

    output_path = job_dir / "interactions.json"
    with open(output_path, "w") as f:
        json.dump({"objects": interactions, "frame_annotations": frame_annotations}, f, indent=2)

    print(f"[pipeline] Done → {output_path} ({len(interactions)} objects)")
    return interactions


if __name__ == "__main__":
    _d = PipelineConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply",              required=True)
    parser.add_argument("--prompt",           required=True)
    parser.add_argument("--job_dir",          required=True)
    parser.add_argument("--quality",          choices=list(QUALITY_PRESETS.keys()),
                        default=DEFAULT_QUALITY,
                        help="Camera-coverage preset (controls number of views)")
    parser.add_argument("--score_threshold",  type=float, default=_d.score_threshold)
    parser.add_argument("--min_votes",        type=int,   default=_d.min_votes)
    parser.add_argument("--min_peak_score",   type=float, default=_d.min_peak_score)
    args = parser.parse_args()
    cfg = PipelineConfig.from_overrides(
        quality=args.quality,
        score_threshold=args.score_threshold,
        min_votes=args.min_votes,
        min_peak_score=args.min_peak_score,
    )
    result = run_pipeline(args.ply, args.prompt, args.job_dir, cfg)
    print(json.dumps(result, indent=2))
