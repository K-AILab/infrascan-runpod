# Scene Graph Pipeline

Builds a 3D object scene graph (labeled boxes + spatial relationship edges)
for a scanned space, from a Gaussian Splat and/or a captured point cloud.

Two independent detection approaches are included; either can be used alone
or merged together.

- **Approach A — OWLv2 on the Gaussian Splat.** Renders synthetic views of
  the splat, runs open-vocabulary 2D detection (OWLv2), lifts to 3D boxes,
  then transforms those boxes onto the real point cloud and relabels them
  with CLIP using the space's own real photos.
- **Approach B — 3DETR on the point cloud.** A PointNet++/3DETR detector run
  directly on the captured point cloud. Useful as a fallback, or to merge
  with Approach A for higher recall, when Approach A's detections for a
  given space are too sparse or unreliable on their own.

## Setup

```bash
conda env create -f environment.yml
conda activate scene-graph-pipeline
```

`external/splat_analyzer` (OWLv2 detection) has its own, separate
dependencies (`gsplat`, `transformers`) — set it up in its own environment
per `external/splat_analyzer/README.md` rather than adding those packages to
the main environment above.

Both environments need a CUDA-matched `torch`/`torchvision` build installed
first (see each `environment.yml`/`requirements.txt` for the exact command).
OWLv2 and CLIP model weights are downloaded automatically from Hugging Face
on first run — an internet connection is required at least once.

`spaces.json` is the space registry and is **not tracked** — it holds local
paths. Create it before anything else:

```bash
cp spaces.example.json spaces.json
```

Then add one entry per space (`title`, `data_root`, `out_dir`, `y_up`). `y_up`
matters: it records whether that capture's point cloud is Y-up or Y-down, and
several stages read it rather than guessing a vertical axis from the geometry.
A room with a dense floor *and* a dense ceiling looks nearly the same either
way up, so guessing is unreliable — set it from what the capture rig produced.

## Approach A — one command per space

For any space registered in `pipeline9/run_space_pipeline.py`'s `SPACES` table:

```bash
python pipeline9/run_space_pipeline.py --space shinhan_space
```

Useful flags — detection takes ~40 min and ICP ~15 min, so both are skippable
once their outputs exist:

```bash
--skip-detect      reuse an existing job dir / raw_detections.json
--skip-align       reuse an existing splat-to-pointcloud transform
--skip-masks       skip the SAM silhouette refit
--skip-verify      skip reprojection-IoU annotation
--skip-harmonise   skip the final reconciliation pass
```

The run writes scene graphs into `ui/_spaces/<space>/scene_graph.json` (both
the splat-geo and point-cloud frames) and boxes into
`tri-viewer/modes/threed/scene/<scene>.boxes.json`, plus every intermediate
stage under `pipeline9/out/`.

### What it does

| # | Stage | Script |
|---|---|---|
| 1 | Render synthetic views and run open-vocabulary 2D detection | `external/splat_analyzer/run_local.py` |
| 2 | Rotate boxes back into the original splat frame | `rotate_and_export.py` |
| 3 | Drop boxes enclosing no reconstructed surface | `closed_surface_flux.py` |
| 4 | Refit extents against the Gaussians | `refit_box_extent.py` |
| 5 | Register splat to captured point cloud (ICP) | `align_splat_to_pointcloud.py` |
| 6 | Label from the space's real photographs | `relabel_with_clip.py` |
| 7 | Physical support prior | `support_prior_filter.py` |
| 8 | Detect work surfaces from geometry | `detect_tables_topdown.py` |
| 9 | Drop floating bases to the floor | `ground_floor_standing_boxes.py` |
| 9b | SAM silhouette refit for compact classes | `refit_box_from_masks.py` |
| 9d | Label boxes the surface detector created | `relabel_with_clip.py` |
| 9f | Record reprojection IoU per box | `verify_boxes_render_back.py` |
| 9g | Reconcile orientation, synonyms, duplicates | `harmonize_scene.py` |
| 10–11 | Export scene graphs and install into both viewers | `export_scene_graph_for_point_viewer.py`, `apply_scenegraph_to_pointcloud.py` |

Detection runs against the **derotated** splat, because the box fitters only
produce axis-aligned boxes and an axis-aligned box only fits an axis-aligned
room. Boxes are rotated back afterwards.

### Two ordering rules

**Labelling runs before the support prior.** CLIP labels from photo crops and
has no notion of height, so it will call a ceiling fixture a trash bin. The
support prior is what catches that, and must run downstream of it.

**Geometric evidence outranks photo-crop evidence.** A label established by
the surface detector is never overturned by CLIP — CLIP may only narrow a
class that geometry has already established. Without this rule a run can
rename correctly detected desks to whatever a single crop happened to look
like.

### Per-space configuration

`SPACES` in `run_space_pipeline.py` holds only what cannot be derived:

| Key | Meaning |
|---|---|
| `splat` / `derot` | original and derotated ply |
| `yaw_deg` | the room's yaw; the derotated ply is the splat rotated by `-yaw_deg` |
| `tri_frame` | whether this scene's tri-viewer `.ksplat` was built from the original or derotated ply |
| `viewer_scale` | metres-per-native-unit the space's existing ui point cloud was generated at |
| `surface_label` | what to call a newly discovered work surface (`table` in an office, `workbench` in a factory) |
| `max_long_m` | longest single work surface, before a blob is split |
| `drop_labels` | classes whose boxes are unreliable in this space and should be removed |

Everything else is measured from the data: the ICP scale guess comes from the
ratio of vertical extents (the one axis a room yaw cannot change), and the
detector's size cap is converted from a metre target using that scale.

## Geometry and labelling passes

These sit on top of the original detect-and-lift chain and are what make the
output usable.

**`support_prior_filter.py`** — an object must rest on the floor, on another
detection, or on a continuous column of geometry. Unsupported boxes at ceiling
height are reassigned; floating ones are dropped. This is the most effective
false-positive filter in the pipeline, because a 2D detector has no way to
express the constraint.

**`detect_tables_topdown.py`** — finds work surfaces directly from geometry as
a large horizontal plane at a self-calibrated height. More reliable than
detecting them in 2D, where a surface is usually seen edge-on, and it
recovers surfaces no 2D detection proposed at all.

**`ground_floor_standing_boxes.py`** — chair castors and table legs carry too
little reconstructed geometry to be fitted, so boxes float above the floor
with otherwise correct tops.

**`refit_box_from_masks.py`** — SAM silhouettes instead of box interiors, for
compact classes. For seating it selects only the views that see the object
from the side away from its desk, which is the one angle where the chair and
not the desk is the nearest surface.

**`label_from_panoramas.py`** — per-instance multi-view label fusion: top-k
views by visibility, several crop scales each, all embeddings averaged into
one before classifying. Fusing before deciding is what stops a row of
identical benches being split across two names.

**`harmonize_scene.py`** — final reconciliation: snap every box to the room
grid, collapse surface synonyms to one name, reclassify boxes whose top is a
dense flat slab, remove shelf tiers claimed as work surfaces, and suppress
duplicates.

## Verifying a change

Where two scans cover the same room (e.g. `factory_space_13` and
`factory_space_14`), their overlap gives a quality metric with no hand
labelling:

```bash
python pipeline9/cross_scan_compare.py \
  --space-a factory_space_13 --space-b factory_space_14 \
  --boxes-a pipeline9/out/factory_space_13_sg_pointcloud.json \
  --boxes-b pipeline9/out/factory_space_14_sg_pointcloud.json \
  --out pipeline9/out/crossscan.json
```

Register the clouds, match the detections, and any label disagreement is by
construction an error in one scan or the other. Run it before and after a
change to see whether the change helped.

It measures *consistency*, not correctness — two scans can agree on the same
wrong label — so treat a drop as a warning worth investigating rather than a
verdict, and still review the result visually.

## Viewers

```bash
cd tri-viewer && python3 server.py 8030          # Gaussian Splat, free-fly
cd server && python -m uvicorn server:app --port 8040   # point-cloud scene graph
```

- tri-viewer: `http://localhost:8030/modes/threed/index.html?scan=<scene>` —
  free-fly through the splat with boxes overlaid, and a Show Points toggle to
  the captured cloud in the same frame.
- ui viewer: `http://localhost:8040/<space>/viewer/sg_3d_viewer.html?space=<space>`
  — the `?space=` parameter is required.

### Making a space appear in the web viewer

The pipeline writes `ui/_spaces/<space>/scene_graph.json`, but the viewer needs
three more things in that directory — an `index.html`, a `Data_` link to the
capture, and a `topdown/` floor plan. Without them the space still lists on the
landing page and then fails quietly: Dev mode 404s and top-down opens to a
broken image. Provision them once per space:

```bash
python pipeline9/provision_ui_space.py \
  --space shinhan_owlv2_pointcloud_scenegraph \
  --from-space shinhan_space_p4
```

`--from-space` is any space whose capture is the one this graph was reprojected
onto — the floor plan and the point cloud are the same assets, so they are
linked rather than regenerated.

### Show Points in tri-viewer

To make the Show Points toggle work for a scene, the captured cloud has to be
converted into that scene's frame first:

```bash
python pipeline9/pointcloud_to_triviewer_points.py \
  --pointcloud-ply data/<space>/pointcloud.ply \
  --transform pipeline9/out/<space>_splat_to_pc_transform.json \
  --extra-yaw-deg -13.639 \
  --out tri-viewer/modes/threed/scene/<scene>.points.ply
```

`--extra-yaw-deg` is `-yaw_deg` when the scene's `.ksplat` was built from the
derotated ply, and omitted otherwise.

## Adding a new space

1. Put the splat at `data/<name>.ply` and the capture under
   `data/<space>/{cameras.json, intrinsics.json, views/, frames/, depth/, pointcloud.ply}`.
2. Measure the room yaw and derotate:
   ```bash
   python pipeline9/derotate_splat.py --in-ply data/<name>.ply \
     --out-ply data/<name>_derotated.ply --yaw-deg -<YAW>
   ```
3. Register the space in `spaces.json` and add an entry to `SPACES` in
   `run_space_pipeline.py`.
4. Run the pipeline.
5. Provision its viewer directory (see *Making a space appear in the web
   viewer* above) — the pipeline writes the scene graph there but not the
   `index.html`, `Data_`, and `topdown/` the viewer also needs.

## Approach B — 3DETR on the point cloud

`my_space` must already be registered in `spaces.json`, with its point cloud
available at `<data_root>/pointcloud.ply` (or pass `--ply` to override).

```bash
python pipeline4/p4_detect.py --space my_space \
  --checkpoint weights/3detr_scannet_masked_ep1080.pth
# writes pipeline4/out/my_space_p4_geo.json

python pipeline2b/geo_label_clip.py --space my_space \
  --geo-json pipeline4/out/my_space_p4_geo.json

python pipeline2b/geo_to_scenegraph.py --space my_space \
  --geo-json pipeline4/out/my_space_p4_geo.json \
  --out-space my_space_p4 --rooms-paper
# writes out/geo_my_space_p4/scene_graph.json
```

The pretrained checkpoint (`3detr_scannet_masked_ep1080.pth`, from the
upstream [facebookresearch/3detr](https://github.com/facebookresearch/3detr)
release) is not included and must be downloaded separately into `weights/`.

`geo_label_clip.py` relabels the detections in place (same file);
`geo_to_scenegraph.py` reads that file and builds the final scene graph.
`--rooms-paper` runs real room/area segmentation (`pipeline2b/room_segment.py`,
an implementation of Tang et al.'s indoor space segmentation method) and
tags every node with the room it physically stands in.

## Merging the two approaches

```bash
python pipeline9/merge_splat_with_p4.py \
  --splat-scene-graph my_space_final.json \
  --p4-scene-graph out/geo_my_space_p4/scene_graph.json \
  --pointcloud-ply /path/to/pointcloud.ply --space my_space --out my_space_merged.json
```

New objects from Approach A are accepted only after being deduplicated
against Approach B's detections and verified against real point-cloud
occupancy (rejecting any box that encloses too few real points to be a
genuine detection). Each accepted object inherits its room/area assignment
from its nearest Approach-B neighbor, and edges are rebuilt with
`pipeline2b/scene_graph.py`'s `build_edges()`, which never connects two
objects assigned to different rooms.

**Room segmentation only exists if Approach B (or a merge with it) is run.**
Approach A alone has no wall/room information of its own — every object
gets a single default room — so running the merge step above (or B alone)
is required whenever room-aware, no-cross-room edges matter.

## Notes on reproducibility

Re-running this pipeline on the same input data should reproduce closely
comparable results, but not necessarily bit-identical ones: OWLv2/CLIP
inference and gsplat rendering are sensitive to GPU architecture, driver,
and library-version differences (floating-point rounding can shift
borderline detection scores near a threshold), and the 3DETR checkpoint is
sourced separately from this repository. Randomness inside the pipeline
itself (camera placement sampling, DBSCAN/K-means clustering) is seeded, so
that is not a source of variation.

Two things must match for a comparison to mean anything:

- **The detection vocabulary.** `PROMPT` in `run_space_pipeline.py` is the
  23-label list every result here was produced with. A class absent from it
  cannot be detected at all, so changing the list changes which objects exist
  before any later stage runs.
- **The registered transform.** `pipeline9/out/<space>_splat_to_pc_transform.json`
  is found by search, and re-running the search can land on a different valid
  answer. Reuse the existing file (`--skip-align`) when comparing against an
  earlier run rather than re-deriving it.

Captured data (`data/`), model weights (`weights/`), rendered job directories,
and the per-space viewer assets under `ui/_spaces/` are all untracked, so a
fresh clone has the code and the box results but not the inputs.
