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
    width: int = 896               # OWLv2 (base-patch16) resizes every input to
    height: int = 896              # 960x960 internally, so rendering much below
                                   # that just feeds it upsampled blur. 896 sits
                                   # just under 960 so the model downsamples
                                   # slightly rather than inventing detail.
    fov_deg: float = 90.0          # horizontal field of view. A fisheye-class
                                   # FoV spreads the pixel budget over a much
                                   # larger solid angle and distorts frame edges
                                   # outside OWLv2's training distribution.
                                   # Coverage is cheaper to buy back with more
                                   # azimuth steps than with a wider lens.
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

    hard_min_surface_dist_frac: float = 0.03  # ABSOLUTE floor on camera-to-nearest-
                                     # splat distance, as a fraction of the bbox
                                     # diagonal. min_surface_dist_frac above was
                                     # only ever enforced on the FIRST sampling
                                     # round: the relaxation loop divides it by
                                     # 2**relax (up to 8x) and the farthest-point
                                     # fill that runs after it ignored the
                                     # distance test entirely, so at high
                                     # --n_positions (where round 1 cannot place
                                     # enough cameras and the fallbacks do most of
                                     # the work) the nominal limit was meaningless.
                                     # See README for how this was calibrated.
    z_band_lo_frac: float = 0.35    # restrict camera height to a band between
    z_band_hi_frac: float = 0.85    # these fractions of the scene's vertical
                                     # (Z) extent, measured from the robust bbox
                                     # floor. Sampling uniformly through the FULL
                                     # 3D bounding box instead puts cameras above
                                     # the ceiling and below the floor — neither can
                                     # see the room, and both render as near-field
                                     # haze. This band is the
                                     # rough equivalent of "stand a person in the
                                     # room": it does not need to be exact, only
                                     # to exclude the two dead zones. Set either
                                     # to None to disable the band.

    # ── frame quality gate (applied AFTER rendering, BEFORE detection) ─────
    # A rendered view of a splat is not automatically a usable detection input.
    # Views from positions the splat was never trained near reconstruct as
    # low-opacity smears, and OWLv2 reliably invents objects in them. Rather
    # than trying to suppress those detections downstream (where they are
    # indistinguishable from real ones once lifted to 3D), drop the frames.
    # All three are measured directly from the render, so they cost nothing.
    frame_sharpness_pct: float = 30.0   # drop the least-sharp N% of a run's own
                                     # frames (variance of Laplacian, measured on
                                     # a fixed 512px downsample — see
                                     # render_cameras._frame_quality). This is
                                     # deliberately RELATIVE, unlike the two
                                     # absolute gates below. An absolute blur
                                     # cutoff does not transfer: the metric shifts
                                     # by an order of magnitude with resolution
                                     # and FoV, and its absolute level depends on
                                     # how much texture the scene itself has — a
                                     # white-walled office scores low everywhere
                                     # even in its perfectly usable views, so a
                                     # constant tuned on one splat gates away
                                     # either nothing or everything on the next.
                                     # What IS stable across scenes is that a
                                     # run's worst views are its off-distribution
                                     # ones. Set to 0 to disable.
    min_frame_sharpness: float = None   # optional ABSOLUTE floor layered on top,
                                     # in the same 512px-referenced units. Off by
                                     # default; useful only when a run is known to
                                     # be mostly bad and the relative cut would
                                     # keep too much of it.
    min_frame_alpha_frac: float = 0.5   # minimum fraction of pixels whose
                                     # accumulated alpha exceeds
                                     # `frame_alpha_thresh`. Rejects views aimed
                                     # largely at empty space / out of the scan.
    frame_alpha_thresh: float = 0.5
    min_frame_median_depth_frac: float = 0.02  # fraction of the bbox diagonal;
                                     # rejects views whose median depth means the
                                     # camera is pressed against a surface. This
                                     # catches survivors of the placement bound
                                     # above, since a camera can clear the
                                     # NEAREST splat and still be facing a wall
                                     # from 2 cm away.
    use_detection_cache: bool = True  # reuse <job_dir>/raw_detections.json when
                                     # the detection-affecting inputs are
                                     # unchanged. Detection costs tens of minutes
                                     # while every stage after it is
                                     # second-scale threshold tuning, so without
                                     # this a clustering change means re-running
                                     # OWLv2 over the whole scene. The cache key
                                     # covers everything that changes what OWLv2
                                     # returns; clustering parameters are
                                     # deliberately excluded so they stay free to
                                     # vary.

    frame_gate: bool = True          # set False to render and detect on
                                     # everything (useful for measuring what the
                                     # gate is actually removing).

    # ── per-detection depth sampling ───────────────────────────────────────
    fg_depth_pct: float = 25.0      # percentile of the valid depths inside a 2-D
                                     # box taken as the object's distance. Depth
                                     # A 5x5 patch at the box CENTRE misses most
                                     # furniture entirely — the centre is the gap
                                     # between a chair's legs or the space under a
                                     # desk, so the sample lands on the wall behind
                                     # and the detection lifts metres past the thing
                                     # that produced it. A low percentile
                                     # takes the near (foreground) end of the box's
                                     # depth distribution while staying robust to
                                     # stray near-camera floaters.
    min_box_surface_px: int = 12    # a box needs at least this many valid,
                                     # sufficiently-opaque pixels to be placed in
                                     # 3D at all.
    min_box_surface_frac: float = 0.10  # ...and they must cover at least this
                                     # fraction of the box interior. Together
                                     # these replace the old `fallback_depth`
                                     # behaviour, which silently substituted the
                                     # camera-to-scene-centre distance whenever
                                     # the depth sample failed — inventing a 3-D
                                     # position for a detection that had no
                                     # reconstructed surface behind it. That is
                                     # the mechanism behind boxes appearing in
                                     # empty space, and it fired constantly
                                     # because the un-normalized depth decode
                                     # (see renderers/gsplat_backend.py) was
                                     # zeroing 18%+ of frames' pixels outright.

    # ── detection / clustering ─────────────────────────────────────────────
    score_threshold: float = 0.12
    min_votes: int = 8
    min_vote_frac: float = None      # if set, OVERRIDES min_votes with
                                     # round(min_vote_frac * total_rendered_views).
                                     # min_votes is an ABSOLUTE recurrence count, but
                                     # total view count scales with n_positions —
                                     # See README for how this was calibrated.
    min_peak_score: float = 0.40
    max_per_label: int = 3          # cap on kept clusters per label — truncates real
                                     # detections once a label genuinely has more
                                     # instances than this, so raise it per run when
                                     # that is the case. See README for calibration.
    max_object_diag: float = 0.5    # native units — reject any detection whose
                                     # implied real-world diagonal exceeds this.
                                     # Scale here is ~6.80 m per native unit, so 0.5
                                     # native is ~3.4 m. That figure comes from the
                                     # NON-derotated splat, which shares the point
                                     # cloud's axes: 16.05/2.352, 3.101/0.457,
                                     # 14.694/2.167 = 6.82/6.79/6.78. Comparing the
                                     # derotated splat's axis-aligned bbox instead
                                     # gives an inflated 7.57 on the horizontal axes,
                                     # because rotating a cloud changes its AABB.
                                     # Without this cap, window/door/cabinet
                                     # detections reach real-world diagonals of
                                     # 3.9-10.9 m on as few as 1-2 frame votes: one
                                     # OWLv2 box mis-reading a large blurry region as
                                     # "door" converts, via pixel-size-at-depth,
                                     # straight into an absurd world-space box.
    max_height_z: float = None      # native units, world Z (this project's splats are
                                     # Z-up) — optional per-run ceiling on how high a
                                     # FLOOR-based object (chair/table/cabinet/plant/
                                     # door; "light" and "window" are exempt, see
                                     # pipeline.py) can sit. With enough camera
                                     # coverage "table" detections split cleanly into
                                     # two height groups: most near ceiling height,
                                     # right where "light" clusters, and a minority at
                                     # real desk height. That is a specific, filterable
                                     # confusion rather than random noise. None =
                                     # disabled: native Z scale is not comparable
                                     # across splats, so this is opt-in per run.
    min_height_z_light: float = None  # native units — required MINIMUM height for
                                     # "light" specifically. "Light" is exempt from
                                     # max_height_z because it can legitimately sit
                                     # near the ceiling, but "could be high" is not
                                     # "must be high": of 8,768 raw "light" detections
                                     # on one run, 2,847 (32%) sat at desk or floor
                                     # height. Lights need their own floor rather than
                                     # an exemption from a ceiling.
    cross_label_overlap_frac: float = 0.3  # reject the lower-confidence detection
                                     # of a DIFFERENT-label pair whose 3D boxes
                                     # overlap by more than this fraction of the
                                     # smaller box's volume. "table"/"chair" pairs
                                     # overlapping 57-79% are not adjacent furniture
                                     # but the same integrated desk+chair unit kept
                                     # under two competing labels. The 0.3 default
                                     # rather than 0.5 also catches the case where the
                                     # redundant box is badly oversized: a "door"
                                     # detection actually looking at a plant reached
                                     # only 31% overlap with the plant's own box.

    min_object_extent: float = 0.01  # native units — per-axis floor on a final
                                     # cluster's size, purely a guard against a
                                     # degenerate zero-thickness box. Was a
                                     # hardcoded 0.1 (~0.5 m per axis on this
                                     # splat), which is not a degeneracy guard
                                     # but a minimum object size bigger than many
                                     # real detections: it pinned all 65 "light"
                                     # clusters to exactly 0.1 native on every
                                     # axis (~0.68 m) and
                                     # fixed the scene's median z-extent at the
                                     # clamp itself. Invisible while the extents
                                     # feeding it were camera-axis noise; the
                                     # dominant size error once they were not.

    cluster_eps: float = None       # native units — absolute clustering radius,
                                     # overrides cluster_eps_frac when set.
    cluster_eps_frac: float = 0.25  # clustering radius as a fraction of
                                     # max_object_diag. Tying it to scene_radius
                                     # instead gives a SINGLE radius shared by every
                                     # label and scaled to the size of the ROOM
                                     # rather than the object: at scene_radius * 0.20
                                     # this splat gives 0.211 native
                                     # units — roughly 1.05 m given the same run's
                                     # max_object_diag=0.5 (~2.5 m) — so any two
                                     # same-label detections within about a metre
                                     # of each other were merged into one object,
                                     # regardless of whether the label was
                                     # "storage rack" or "light". This is the
                                     # failure GaussianDet3D measures directly in
                                     # its NMS ablation (Sec. 4.4): dropping the
                                     # suppression threshold from 0.25 to 0.05
                                     # gained +4.4% mAP overall but +16.8% and
                                     # +22.4% on their two smallest classes, while
                                     # leaving large ones unchanged — over-wide
                                     # spatial suppression costs small objects
                                     # almost exclusively. It is also why
                                     # max_per_label had to be raised to 80 to see
                                     # real instances: the cap was hiding a
                                     # merging problem, not a scoring one.
                                     # Anchoring to max_object_diag keeps this in
                                     # the same native units the caller already
                                     # reasons about for object size.
    label_eps_scale: dict = None    # {label: multiplier} on the clustering radius
                                     # for specific labels, layered on top of the
                                     # value above. A light and a workbench do not
                                     # want the same merge distance; this is the
                                     # per-label physical size prior. Unlisted
                                     # labels use the base radius unchanged.

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
