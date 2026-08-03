"""
Shared pipeline configuration — the single source of truth for defaults used by
both the local CLI (run_local.py / pipeline.py) and the FastAPI server (server.py).

Keeping every default here prevents the kind of drift that crept in before
(e.g. n_positions defaulting to 10 in one place and 5 in another).

The user-facing knob is `quality` (low/medium/high), which expands to the three
internal camera counts via `apply_quality()`. The granular counts remain on the
dataclass for internal use but are not exposed in the UI/API.
"""

from dataclasses import dataclass, fields


# quality name -> (n_positions, n_azimuth, n_elevation)  → total frames
QUALITY_PRESETS = {
    "low":    (3, 4, 2),   # 24 frames  — fast preview
    "medium": (5, 6, 3),   # 90 frames  — balanced default
    "high":   (8, 8, 3),   # 192 frames — thorough coverage
}
DEFAULT_QUALITY = "medium"


@dataclass
class PipelineConfig:
    # ── rendering ──────────────────────────────────────────────────────────
    width: int = 512
    height: int = 512
    renderer: str = "auto"                  # "auto" | "gsplat" (CUDA) | "gsplat-metal" (Apple MPS)
    quality: str = DEFAULT_QUALITY          # drives the camera counts below
    n_positions: int = 5
    n_azimuth: int = 6
    n_elevation: int = 3

    # ── camera placement (density-aware band-pass sampler) ─────────────────
    bbox_pct_lo: float = 1.0                # robust bounding-box lower percentile
    bbox_pct_hi: float = 99.0               # robust bounding-box upper percentile
    min_sep_frac: float = 0.12              # Poisson-disk min camera separation (frac of bbox diagonal)
    density_radius_frac: float = 0.05       # neighbour-count radius (frac of bbox diagonal)
    bandpass_alpha: float = 1.0             # band-pass width; larger = more permissive
    min_surface_dist_frac: float = 0.04     # hard floor on distance to the NEAREST single
                                             # point (not just local density) — the density
                                             # band-pass alone assumes a scan has clearly
                                             # open floor space to reject "buried" positions
                                             # against; a densely cluttered scene (furniture
                                             # covering most of the floor at various heights)
                                             # has little true open space, so "near-median
                                             # density" can still land a camera millimeters
                                             # from a desk/wall — confirmed directly on a
                                             # real run (median rendered depth 0.2-0.5m,
                                             # minimum ~6mm), causing severe near-field
                                             # render blur that fed bad detections downstream.
    seed: int = 42                          # deterministic placement

    # ── detection / clustering ─────────────────────────────────────────────
    score_threshold: float = 0.12
    min_votes: int = 8
    min_vote_frac: float = None      # if set, OVERRIDES min_votes with
                                     # round(min_vote_frac * total_rendered_views).
                                     # min_votes is an ABSOLUTE recurrence count, but
                                     # total view count scales with n_positions —
                                     # confirmed directly: reusing the same min_votes=5
                                     # at n_positions=8 (192 views) vs n_positions=40
                                     # (960 views) let visibly more marginal/false
                                     # detections cross the SAME fixed bar (effective
                                     # selectivity dropped from ~2.6% of views to
                                     # ~0.5%), even though real objects' vote counts
                                     # also rose. Re-clustering the 40-position run's
                                     # raw detections at the PROPORTIONALLY-scaled
                                     # threshold (25 votes, i.e. the same ~2.6%) exactly
                                     # reproduced the 8-position run's chair/window/plant
                                     # counts. Set this instead of --min_votes whenever
                                     # changing --n_positions/--quality, so selectivity
                                     # stays constant. None = disabled (use min_votes as-is).
    min_peak_score: float = 0.40
    max_per_label: int = 3          # was a hardcoded literal at the pipeline.py call
                                     # site (not configurable at all) — silently
                                     # truncates real detections once a label has
                                     # more than this many above-threshold clusters.
                                     # Confirmed directly on a real run: every label
                                     # except one hit EXACTLY 3, including labels
                                     # (table, light) the user could see more
                                     # instances of in the scene.
    max_object_diag: float = 0.5    # native units (~2.5m real-world) — no per-detection
                                     # or per-cluster sanity check on implied real-world
                                     # size existed at all. Confirmed directly: several
                                     # window/door/cabinet detections on this splat had
                                     # real-world diagonals of 3.9-10.9m (backed by as
                                     # few as 1-2 frame votes) — a single OWLv2 box that
                                     # mis-reads a large blurry region as "door" converts,
                                     # via pixel-size-at-depth, straight into an
                                     # absurd world-space box with nothing to catch it.
    max_height_z: float = None      # native units, world Z (this project's splats are
                                     # Z-up) — optional per-run ceiling on how high a
                                     # FLOOR-based object (chair/table/cabinet/plant/
                                     # door; "light" and "window" are exempt, see
                                     # pipeline.py) can sit. Confirmed directly: with
                                     # enough camera coverage, "table" detections split
                                     # cleanly into two groups by height — 4/5 near
                                     # ceiling height (z~0.04-0.05, matching where
                                     # "light" clusters, i.e. OWLv2 confusing some
                                     # ceiling-height content for a tabletop) and 1/5 at
                                     # real desk height (z~-0.17) — not random noise,
                                     # a specific, filterable confusion. None = disabled
                                     # (this project's splats' native Z scale isn't
                                     # meaningful for an arbitrary other splat, so this
                                     # is opt-in per run, not a hardcoded default).
    min_height_z_light: float = None  # native units — required MINIMUM height for
                                     # "light" specifically. max_height_z originally
                                     # exempted "light" from any height check at all
                                     # (reasoning it could legitimately be near
                                     # ceiling) — but "could be high" isn't the same
                                     # as "must be high", and confirmed directly:
                                     # of 8,768 raw "light" detections on a real run,
                                     # 2,847 (32%) were at desk/floor height, not
                                     # ceiling — visibly wrong boxes hovering over
                                     # desks in the viewer. "light" needs its OWN
                                     # floor, not an exemption from one.
    cross_label_overlap_frac: float = 0.3  # reject the lower-confidence detection
                                     # of a DIFFERENT-label pair whose 3D boxes
                                     # overlap by more than this fraction of the
                                     # smaller box's volume. Confirmed directly:
                                     # several "table"/"chair" pairs on a real run
                                     # overlapped 57-79% — not adjacent furniture,
                                     # the SAME integrated desk+chair unit kept
                                     # under two competing labels. Lowered from 0.5
                                     # to 0.3 after a second confirmed case: a
                                     # "door" detection whose box was badly
                                     # oversized relative to the real plant it was
                                     # actually seeing only reached 31% overlap.

    label_overrides: dict = None    # {label: {"min_votes": int, "min_peak_score": float}}
                                     # per-label recall/precision override, layered on
                                     # top of min_votes/min_peak_score above for any
                                     # label listed here (unlisted labels use the
                                     # global values unchanged). `_cluster_detections`
                                     # already supported this; wasn't threaded through
                                     # `run_pipeline`/this config until a real case
                                     # needed it: a "table" directly confirmed missing
                                     # from detection entirely (crop-verified — a real,
                                     # clearly visible desk near the plant had ZERO
                                     # bounding boxes across all detected table
                                     # objects), while light/chair/window/plant were
                                     # already working well at the global thresholds —
                                     # needed a way to loosen just `table` (and
                                     # `chair`, also reported as still-imperfect)
                                     # without weakening labels that weren't broken.

    def apply_quality(self):
        """Expand the `quality` preset into the three camera-count fields."""
        if self.quality in QUALITY_PRESETS:
            self.n_positions, self.n_azimuth, self.n_elevation = QUALITY_PRESETS[self.quality]
        return self

    @classmethod
    def from_overrides(cls, **kw):
        """Build a config from a loose dict of overrides (ignores unknown/None keys).

        Applies the quality preset FIRST, then layers explicit overrides on
        top — apply_quality() unconditionally stomps n_positions/n_azimuth/
        n_elevation, so an explicit override passed alongside `quality` (e.g.
        `--n_positions 16 --quality high`) used to get silently discarded:
        `cls(**filtered).apply_quality()` set n_positions=16 on construction,
        then immediately overwrote it back to 8 (high's preset value)."""
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kw.items() if k in valid and v is not None}
        base = cls(quality=filtered.get("quality", DEFAULT_QUALITY)).apply_quality()
        for k, v in filtered.items():
            setattr(base, k, v)
        return base
