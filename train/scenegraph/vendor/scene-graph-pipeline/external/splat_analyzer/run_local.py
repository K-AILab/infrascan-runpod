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
                    help="cap on kept clusters per label — was a hidden hardcoded "
                    "3 with no way to change it; raise this if a label plausibly "
                    "has more real instances than are coming out")
    p.add_argument("--max_object_diag", type=float, default=d.max_object_diag,
                    help="native units — reject any detection/cluster whose implied "
                    "real-world diagonal exceeds this (was unbounded; a single bad "
                    "OWLv2 box could produce a multi-meter 'door'/'window'/'cabinet')")
    p.add_argument("--max_height_z", type=float, default=None,
                    help="native units, world Z (this project's splats are Z-up) — "
                    "reject any detection above this height whose label isn't "
                    "light/window; confirmed directly that 'table' detections split "
                    "cleanly into a ceiling-height false-positive group and a real "
                    "desk-height group on this data")
    p.add_argument("--min_height_z_light", type=float, default=None,
                    help="native units — reject any 'light' detection BELOW this "
                    "height (light is exempt from --max_height_z since it can "
                    "legitimately be near ceiling, but that's not the same as being "
                    "plausible at any height — confirmed directly that 32%% of raw "
                    "light detections on this data were at desk/floor height)")
    p.add_argument("--cross_label_overlap_frac", type=float, default=d.cross_label_overlap_frac,
                    help="reject the lower-confidence detection of a DIFFERENT-label "
                    "pair whose 3D boxes overlap more than this fraction of the "
                    "smaller box's volume — confirmed directly that several "
                    "'table'/'chair' pairs overlapped 57-79%%, the same integrated "
                    "desk+chair unit kept under two competing labels")
    p.add_argument("--n_positions", type=int, default=None,
                    help="override the quality preset's camera-position count — "
                    "quality=high's fixed 8 positions left real gaps in room "
                    "coverage on this splat (confirmed: camera X range didn't "
                    "reach one edge of the room at all), and only 8 positions "
                    "means each downward-facing view (needed for tables/desks) "
                    "is a small fraction of total coverage")
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
    )

    objects = pipeline.run_pipeline(args.ply, args.prompt, job_dir, cfg)

    out = Path(job_dir)
    print(f"\nDone — {len(objects)} object(s) detected.")
    print(f"  results: {out / 'interactions.json'}")
    print(f"  frames:  {out / 'frames'}")


if __name__ == "__main__":
    main()
