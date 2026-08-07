# Scene-graph stage — integration notes

This stage runs the upstream scene-graph pipeline
**`k-ailab/scene-graph-pipeline` @ `e4d94093` (Approach A)** as-is, after training,
on the same warm RunPod train worker, and publishes a `scene_graph.json` our
viewer already understands.

The goal was to change the creator's pipeline as little as possible. This file
records **every** deviation from a stock upstream checkout. If you re-vendor a
newer upstream, re-apply only what's listed here.

## What is vendored (`vendor/scene-graph-pipeline/`)
Pristine copy of `e4d94093`, limited to the Approach-A packages:
`pipeline/`, `pipeline2b/`, `pipeline9/`, `external/splat_analyzer/`
(+ `spaces.example.json`, `environment.yml`, `README.md`). See
`VENDOR_PROVENANCE.txt`.

Omitted (not on the Approach-A path, or the creator's own separate apps, none
imported by the code we run): `pipeline4/` (Approach B / 3DETR), `ui/`,
`tri-viewer/`, `server/`.

## The ONE edit to upstream code
`pipeline9/run_space_pipeline.py` — a 6-line additive, backward-compatible shim
right after the `SPACES` dict:

```python
import os as _sg_os
_sg_ext = _sg_os.environ.get("SG_SPACES_JSON")
if _sg_ext and Path(_sg_ext).exists():
    SPACES.update(json.loads(Path(_sg_ext).read_text()))
```

Why: upstream's per-space constants live in a hand-edited `SPACES` dict. Our
endpoint registers an S3-sourced space at runtime instead. With `SG_SPACES_JSON`
unset, upstream behaviour is identical. No other upstream file is modified.
(An earlier stale-snapshot edit to `export_scene_graph_for_point_viewer.py` was
reverted by the full re-vendor.)

## Our glue (not upstream code)
- `run_scenegraph.py` — the driver. Stages the S3 scan into the pipeline's
  layout (`data/<slug>/{splat.ply,pointcloud.ply,cameras.json,intrinsics.json,
  views/}`), writes `spaces.json` + the `SG_SPACES_JSON` entry, runs
  `pipeline9/run_space_pipeline.py --space <slug> --no-install`, then converts
  the pipeline's own output to our `scene_graph.json` and uploads it. It
  reimplements none of the pipeline's logic.
- `prefetch_weights.py` — bakes OWLv2 + CLIP + `mobile_sam.pt` for offline runs.
- `train/handler.py` step 10 / `only_scenegraph` — invoke the driver (unchanged
  contract; added optional `yaw_deg`/`surface_label`/`max_long_m`/`drop_labels`/
  `skip_masks`/`skip_verify` job inputs).
- `train/Dockerfile` — see below.

## Deviations & assumptions (why, so they can be revisited)

1. **One combined venv (`/opt/venv-sg`), not the README's two separate envs.**
   Upstream's `run_space_pipeline.py` drives *every* stage — including the OWLv2
   detector — with a single `sys.executable`, so the pipeline as the creator runs
   it (their server does `conda activate infrascan`) expects one environment. We
   match that: one `--system-site-packages` venv reusing the base image's
   torch + prebuilt gsplat + global `ultralytics/open3d/opencv`, plus
   `transformers/open_clip/sklearn/plyfile/imageio/six`.

2. **Base gsplat, not the pinned `gsplat==1.5.3`.** The nerfstudio base ships a
   prebuilt gsplat and has no nvcc; the detector only uses gsplat's stable
   `rasterization` API, and `renderers/base.py` already handles the 1.5.3 scale
   convention. If detection misbehaves in a way traceable to gsplat, this is the
   first thing to revisit (would require a CUDA-devel base to build 1.5.3).

3. **`yaw_deg = 0` by default (detection on the splat as captured).** Upstream
   detects on a *pre-derotated* splat using a per-space yaw it measures **by
   hand** (floor min-area-rectangle) and hardcodes; there is **no committed
   auto-yaw tool** (`derotate_splat.py` only *applies* a given yaw, and
   `align`'s `building_yaw_deg` is the *pointcloud's* wall angle, not the
   splat's). Rather than invent a measurement, we pass `yaw=0`; every other
   upstream stage still runs. A measured yaw can be supplied per job via
   `input.yaw_deg`. Strongly-rotated rooms will have axis-aligned (slightly
   loose) boxes until a yaw is provided.

4. **`--no-install`; we don't use upstream's viewer install.** Upstream step 11
   writes into `tri-viewer/` and `ui/_spaces/`. We skip that and instead convert
   `pipeline9/out/<slug>_boxes_final.json` (geometry, splat-native frame == our
   `.ksplat` frame) + `<slug>_geo_true.json` (edges) into our viewer schema
   (`{slug,coord_frame:"splat",up_axis:"z",nodes:[{id,label,center,size,angle}],
   edges:[{src,dst,relation}],labels}`) and upload `scans/<slug>/scene_graph.json`.
   `angle` (added after the first version of this conversion) is carried straight
   through from `boxes_final.json`, where `harmonize_scene.py` already reconciles
   it across the topdown/mask-refit passes -- earlier, `_convert_to_viewer()` read
   `label/center/size` off each box but silently dropped `angle`, so every box's
   rotation was lost between generation and the viewer even though the pipeline
   computed it correctly. Note this is a per-OBJECT rotation, independent of the
   room-level `yaw_deg` in point 3 above -- a chair sitting at an angle in an
   otherwise axis-aligned (yaw_deg=0) room still has a real, useful `angle`.

5. **Per-space knob defaults.** `surface_label=table`, `max_long_m=2.1` (office).
   Factory-type spaces want `surface_label=workbench`, `max_long_m≈7.0`
   (`input.surface_label` / `input.max_long_m`). `viewer_scale=None` → the
   pipeline's ICP-measured true scale is used. `y_up=false` in `spaces.json`
   (our splats are Z-up; the pipeline's scale guess only seeds ICP, which
   recovers the true scale regardless).

## Diagnostics (all in S3 under `scans/<slug>/`)
- `scene_graph.json` — the viewer artifact.
- `scenegraph/<slug>_{boxes_final,geo_true,splat_to_pc_transform}.json` — the
  pipeline's own outputs, for provenance.
- `scenegraph/debug/pipeline.log` — full pipeline stdout/stderr every run.
- `scenegraph_error.txt` — traceback + log tail on failure.
