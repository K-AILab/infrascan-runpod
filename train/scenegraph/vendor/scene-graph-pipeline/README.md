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

## Approach A — OWLv2 → transform → CLIP relabel

1. **Detect on the splat.**
   ```bash
   cd external/splat_analyzer
   python run_local.py --ply /path/to/scene.ply \
     --prompt "chair, table, desk, workbench, shelf, storage rack, cabinet, \
   cardboard box, pallet, cart, machine, trash bin, whiteboard, bench, \
   ladder, light, window, door, fire extinguisher, printer, plant, monitor" \
     --quality high --n_positions 64 \
     --score_threshold 0.10 --min_vote_frac 0.026 --min_peak_score 0.35 \
     --max_per_label 80 --max_object_diag 0.5 \
     --job_dir out_scene
   python rotate_and_export.py --interactions out_scene/interactions.json \
     --out scene_boxes.json --yaw-deg 0
   ```
   `--yaw-deg` should match whatever rotation (if any) was applied to the
   splat before detection — pass `0` if detection ran directly on the splat
   as captured. `--n_positions 64` (vs. the `high` preset's default of 8) is
   what closes real coverage gaps in larger rooms; scale it down for a quick
   pass, up for full recall on a large or cluttered space.

2. **Refine box sizes.** The detector's own box-size estimate is a crude
   single-sample guess; refit it against the splat's own gaussians.
   ```bash
   python ../../pipeline9/closed_surface_flux.py \
     --ply /path/to/scene.ply --boxes scene_boxes.json \
     --min-gaussians 50 --filtered-out scene_boxes_filtered.json
   python ../../pipeline9/refit_box_extent.py \
     --ply /path/to/scene.ply --boxes scene_boxes_filtered.json \
     --out scene_boxes_refit.json
   ```
   `refit_box_extent.py` fits `table` footprints with a horizontal-plane
   RANSAC fit (tables are approximately flat) — this is what makes table
   boxes noticeably tighter and more accurate than the raw detector output.

3. **Transform onto the real point cloud.** First find the rigid transform
   between the splat and the independently captured point cloud — this
   discovers the true native-units-to-meters scale automatically (reflection
   probe → scale+yaw sweep → ICP refinement), rather than assuming it:
   ```bash
   cd ../../pipeline9
   python align_splat_to_pointcloud.py \
     --splat-ply /path/to/scene.ply --pointcloud-ply /path/to/pointcloud.ply \
     --scale-to-meters-guess <ROUGH_SCALE> --out my_space_transform.json
   ```
   `<ROUGH_SCALE>` only needs to be in the right order of magnitude — it
   seeds the search, it is not trusted as-is. Read the true value back out
   of the written transform file's `true_scale_to_meters` field and reuse it
   as `<SCALE>` below:
   ```bash
   python export_scene_graph_for_point_viewer.py \
     --ply /path/to/scene.ply --yaw-deg 0 --scale-to-meters <SCALE> \
     --boxes ../external/splat_analyzer/scene_boxes_refit.json \
     --space my_space --out my_space_geo.json

   python apply_scenegraph_to_pointcloud.py \
     --scene-graph my_space_geo.json --transform my_space_transform.json \
     --source-scale-to-meters <SCALE> --space my_space \
     --pointcloud-ply /path/to/pointcloud.ply --out my_space_pointcloud.json
   ```

4. **Relabel with CLIP from the space's real photos.**
   ```bash
   python relabel_with_clip.py \
     --scene-graph my_space_pointcloud.json --pointcloud-ply /path/to/pointcloud.ply \
     --space my_space --geo-json-out my_space_relabel_geo.json \
     --out my_space_final.json
   ```
   This requires `my_space` to be registered in `spaces.json` (see
   `spaces.example.json`) with real camera photos available under
   `data/my_space/{cameras.json, intrinsics.json, views/}`. CLIP crops each
   object's own real photos and re-labels it, dropping anything it flags as
   structural (a real door/window/wall misdetected as an object).

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

## Sample input needed for a new space

- `scene.ply` — the trained Gaussian Splat.
- `pointcloud.ply` — an independently captured point cloud of the same
  space, in real-world meters.
- `data/<space>/{cameras.json, intrinsics.json, views/}` — real per-view
  photos and camera poses, for the CLIP relabeling step.
- An entry for `<space>` in `spaces.json` (copy `spaces.example.json`).

## Notes on reproducibility

Re-running this pipeline on the same input data should reproduce closely
comparable results, but not necessarily bit-identical ones: OWLv2/CLIP
inference and gsplat rendering are sensitive to GPU architecture, driver,
and library-version differences (floating-point rounding can shift
borderline detection scores near a threshold), and the 3DETR checkpoint is
sourced separately from this repository. Randomness inside the pipeline
itself (camera placement sampling, DBSCAN/K-means clustering) is seeded, so
that is not a source of variation.
