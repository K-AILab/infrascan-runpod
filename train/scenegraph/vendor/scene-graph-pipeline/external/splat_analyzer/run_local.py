"""
Local one-command runner for the WorldModelData pipeline.

Runs the *exact same* rendering + detection pipeline the server uses (no
duplicated logic — this is a thin front door over pipeline.run_pipeline) on a
machine with an NVIDIA CUDA GPU.

Example:
  python run_local.py --ply scene.ply --prompt "chair, table" --quality high

Outputs <job_dir>/interactions.json plus the rendered frames/ and transforms.json.
"""

import argparse
import sys
from pathlib import Path

import torch

import pipeline
from config import PipelineConfig, QUALITY_PRESETS, DEFAULT_QUALITY


def main():
    d = PipelineConfig()
    p = argparse.ArgumentParser(description="Run 3DGS object detection locally.")
    p.add_argument("--ply",     required=True, help="Path to a .ply or .spz Gaussian Splat file")
    p.add_argument("--prompt",  required=True, help='Comma-separated labels, e.g. "chair, table"')
    p.add_argument("--quality", choices=list(QUALITY_PRESETS.keys()), default=DEFAULT_QUALITY,
                   help="Camera-coverage preset (controls number of views)")
    p.add_argument("--job_dir", default=None, help="Output directory (default: ./out_<name>)")
    p.add_argument("--score_threshold", type=float, default=d.score_threshold)
    p.add_argument("--min_votes",       type=int,   default=d.min_votes)
    p.add_argument("--min_vote_frac",   type=float, default=d.min_vote_frac,
                    help="OVERRIDES --min_votes with round(frac * total_rendered_views) — "
                    "use this instead of --min_votes whenever changing --n_positions/"
                    "--quality, since min_votes is an absolute recurrence count and "
                    "total views scales with position count; reusing the same "
                    "min_votes at higher n_positions silently lets more marginal/"
                    "false detections cross the same bar. E.g. 0.026 matched this "
                    "splat's own 8-position/min_votes=5 baseline (5/192)")
    p.add_argument("--min_peak_score",  type=float, default=d.min_peak_score)
    p.add_argument("--max_per_label",   type=int,   default=d.max_per_label,
                    help="cap on kept clusters per label; raise it if a label "
                    "plausibly has more real instances than are coming out")
    p.add_argument("--max_object_diag", type=float, default=d.max_object_diag,
                    help="native units — reject any detection whose implied real-world "
                    "diagonal exceeds this. A single bad 2D box can otherwise lift "
                    "into a multi-metre object")
    p.add_argument("--max_height_z", type=float, default=None,
                    help="native units, world Z (these splats are Z-up) — reject any "
                    "detection above this height whose label is not ceiling- or "
                    "wall-mounted. Useful where a label splits cleanly into a "
                    "ceiling-height false-positive group and a real desk-height group")
    p.add_argument("--min_height_z_light", type=float, default=None,
                    help="native units — reject any 'light' detection BELOW this "
                    "height. Lights are exempt from --max_height_z since they can "
                    "legitimately sit near the ceiling, but that is not the same as "
                    "being plausible at any height")
    p.add_argument("--cross_label_overlap_frac", type=float, default=d.cross_label_overlap_frac,
                    help="reject the lower-confidence detection of a DIFFERENT-label "
                    "pair whose 3D boxes overlap more than this fraction of the "
                    "smaller box's volume — one physical object detected under two "
                    "competing labels, e.g. an integrated desk+chair unit")
    p.add_argument("--n_positions", type=int, default=None,
                    help="override the quality preset's camera-position count. The "
                    "presets are small enough to leave coverage gaps in a large "
                    "room, and to make downward-facing views (needed for desks) a "
                    "small fraction of the total")
    p.add_argument("--n_azimuth", type=int, default=None,
                    help="override the quality preset's per-position azimuth "
                    "(horizontal) view count — more views around each camera "
                    "position, independent of how many positions there are")
    p.add_argument("--n_elevation", type=int, default=None,
                    help="override the quality preset's per-position elevation "
                    "(vertical) view count — more up/down angles at each "
                    "camera position, independent of how many positions there are")

    # ── render resolution / field of view ──────────────────────────────────
    p.add_argument("--width",  type=int, default=d.width,
                    help=f"render width in px (default {d.width}) — OWLv2 resizes "
                    "its input to 960x960 internally, so rendering much below "
                    "that just feeds it upsampled blur")
    p.add_argument("--height", type=int, default=d.height,
                    help=f"render height in px (default {d.height})")
    p.add_argument("--fov_deg", type=float, default=d.fov_deg,
                    help=f"horizontal field of view in degrees (default "
                    f"{d.fov_deg}). A fisheye-class FoV spreads the pixel budget "
                    "over a much larger solid angle and distorts frame edges outside "
                    "OWLv2's training distribution; buy coverage back with "
                    "--n_azimuth instead")

    # ── frame quality gate ─────────────────────────────────────────────────
    p.add_argument("--no_frame_gate", action="store_true",
                    help="run detection on EVERY rendered frame, including blurred "
                    "and camera-buried ones. Off by default: the detector reliably "
                    "invents furniture in near-field render haze")
    p.add_argument("--frame_sharpness_pct", type=float, default=d.frame_sharpness_pct,
                    help=f"drop the least-sharp N%% of this run's frames (default "
                    f"{d.frame_sharpness_pct}). Relative rather than absolute "
                    "because variance-of-Laplacian shifts by an order of "
                    "magnitude with resolution/FoV and depends on how textured "
                    "the scene is — a constant tuned on one splat gates away "
                    "either nothing or everything on the next. 0 disables")
    p.add_argument("--min_frame_sharpness", type=float, default=d.min_frame_sharpness,
                    help="optional ABSOLUTE sharpness floor on top of "
                    "--frame_sharpness_pct (512px-referenced units)")
    p.add_argument("--min_frame_alpha_frac", type=float, default=d.min_frame_alpha_frac,
                    help="minimum fraction of pixels backed by real reconstructed "
                    "surface for a frame to be used")

    # ── camera placement ───────────────────────────────────────────────────
    p.add_argument("--hard_min_surface_dist_frac", type=float,
                    default=d.hard_min_surface_dist_frac,
                    help="absolute minimum camera-to-nearest-splat distance as a "
                    "fraction of the bbox diagonal, enforced on EVERY placement "
                    "path including the relaxation rounds and farthest-point "
                    "fill (min_surface_dist_frac was only ever applied on the "
                    "first round, so at high --n_positions it was meaningless). "
                    "Placing fewer cameras is preferred over burying them")

    # ── clustering radius ──────────────────────────────────────────────────
    p.add_argument("--min_object_extent", type=float, default=d.min_object_extent,
                    help=f"native units — per-axis floor on final box size, a "
                    f"degeneracy guard only (default {d.min_object_extent}). Keep it "
                    "small: anything comparable to a real object's size becomes the "
                    "dominant size error for that class")
    p.add_argument("--no_detection_cache", action="store_true",
                    help="ignore <job_dir>/raw_detections.json and re-run OWLv2 "
                    "even when the detection inputs are unchanged")
    p.add_argument("--cluster_eps", type=float, default=d.cluster_eps,
                    help="absolute clustering merge radius in native units; "
                    "overrides --cluster_eps_frac")
    p.add_argument("--cluster_eps_frac", type=float, default=d.cluster_eps_frac,
                    help=f"clustering merge radius as a fraction of "
                    f"--max_object_diag (default {d.cluster_eps_frac}). Anchoring to "
                    "object size rather than room size matters: a radius tied to the "
                    "room merges genuinely separate same-label objects")
    args = p.parse_args()

    if not Path(args.ply).exists():
        sys.exit(f"error: file not found: {args.ply}")

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU detected — running on CPU will be extremely slow.\n"
              "         Install a CUDA-matched build of torch + gsplat for usable speed.",
              file=sys.stderr)

    job_dir = args.job_dir or f"out_{Path(args.ply).stem}"
    cfg = PipelineConfig.from_overrides(
        quality=args.quality,
        score_threshold=args.score_threshold,
        min_votes=args.min_votes,
        min_vote_frac=args.min_vote_frac,
        min_peak_score=args.min_peak_score,
        max_per_label=args.max_per_label,
        max_object_diag=args.max_object_diag,
        max_height_z=args.max_height_z,
        min_height_z_light=args.min_height_z_light,
        cross_label_overlap_frac=args.cross_label_overlap_frac,
        n_positions=args.n_positions,
        n_azimuth=args.n_azimuth,
        n_elevation=args.n_elevation,
        width=args.width,
        height=args.height,
        fov_deg=args.fov_deg,
        frame_gate=not args.no_frame_gate,
        frame_sharpness_pct=args.frame_sharpness_pct,
        min_frame_sharpness=args.min_frame_sharpness,
        min_frame_alpha_frac=args.min_frame_alpha_frac,
        hard_min_surface_dist_frac=args.hard_min_surface_dist_frac,
        cluster_eps=args.cluster_eps,
        cluster_eps_frac=args.cluster_eps_frac,
        min_object_extent=args.min_object_extent,
        use_detection_cache=not args.no_detection_cache,
    )

    objects = pipeline.run_pipeline(args.ply, args.prompt, job_dir, cfg)

    out = Path(job_dir)
    print(f"\nDone — {len(objects)} object(s) detected.")
    print(f"  results: {out / 'interactions.json'}")
    print(f"  frames:  {out / 'frames'}")


if __name__ == "__main__":
    main()
