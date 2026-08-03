# pipeline4 — learned 3D object detection (PointNet++ / 3DETR)

Unlike pipeline2/2b (pure-geometry DBSCAN clustering, optionally seeded by
2D detectors lifted through SAM2/ByteTrack), pipeline4 gets its box
*proposals* from a real 3D object-detection network: **3DETR**
(`facebookresearch/3detr`), which uses a **PointNet++** set-abstraction
layer as its point-cloud backbone followed by a transformer encoder/decoder
(no voting/anchors, unlike VoteNet). See
`external/3detr/` for the upstream repo and
`external/3detr/README.md` / the [Medium article on point-cloud 3D
detection](https://medium.com/@regis.loeb/playing-with-point-clouds-for-3d-object-detection-eff1d98e526a)
for background on how VoteNet and 3DETR relate (deep-Hough-voting +
clustering vs. transformer set-prediction over the same PointNet++ features).

## Why 3DETR over VoteNet here

Both are trained on SUN RGB-D / ScanNetV2 and get near-identical AP25 (~57%)
per the article above. 3DETR was chosen because:
- Its released code has **no custom CUDA ops for the detection head itself**
  — only the shared PointNet++ ops (furthest-point sampling, ball query),
  which this pipeline reimplements in pure PyTorch (`detr3d/ops.py`) so it
  runs on **aarch64 / Blackwell (GB10)** without compiling `pointnet2_ops`
  (which has no prebuilt wheel for this platform and needs source patches
  for CUDA 13 anyway).
- It's ~1/6 the parameter count of the equivalent VoteNet-based systems in
  some configs and only needs a forward pass at inference — no separate
  voting-and-clustering post-process to reimplement.
- The pretrained ScanNet checkpoint (`scannet_masked_ep1080.pth`, "masked"
  = local self-attention encoder variant) transfers reasonably to novel
  indoor/industrial scans without fine-tuning, per the article's SUN-RGBD
  fine-tuning result (30 epochs from a ScanNet init matches 180 from
  scratch) — i.e. the backbone has learned generic indoor-geometry priors,
  not just SUN-RGBD/ScanNet-specific categories.

## What's vendored, and why

`detr3d/` is a trimmed, import-clean copy of the parts of
`external/3detr/` needed for **inference only** (no training loop, no
dataset loaders, no Cython NMS):

| file | origin | change |
|---|---|---|
| `model_3detr.py` | `models/model_3detr.py` | import paths only |
| `transformer.py`, `helpers.py`, `position_embedding.py` | `models/*.py` | import paths only |
| `pointnet2_modules.py`, `pytorch_utils.py` | `third_party/pointnet2/*.py` | import paths only |
| `ops.py` | **new** | pure-PyTorch `furthest_point_sample`, `ball_query`, `QueryAndGroup`, `GroupAll` — drop-in replacements for the compiled `pointnet2._ext` kernels, same semantics (see docstring) |
| `pc_util.py`, `box_util.py` | `utils/pc_util.py`, `utils/box_util.py` | only the 2–6 functions the model actually calls, so we don't need `trimesh`/Cython |
| `scannet_config.py` | `datasets/scannet.py` (`ScannetDatasetConfig`) | trimmed to the 18 detection classes; semseg config dropped |
| `detector.py` | **new** | `Detr3DDetector` — builds the model from the checkpoint's saved args and runs single-window inference, returning AABBs (ScanNet has no box rotation: `angle_logits.shape[-1] == 1`, so `box_parametrization_to_corners` degenerates to axis-aligned) |

Correctness was checked directly: `Detr3DDetector` loads the released
checkpoint with `strict` key matching (no missing/unexpected keys), and the
`center ± size/2` AABB shortcut was numerically verified against the
model's own `box_corners` output (max error `0.0`) before being used for
export.

## Pipeline

```
python pipeline4/p4_detect.py           --space factory_space_14
python pipeline2b/geo_label_clip.py     --space factory_space_14 \
    --geo-json pipeline4/out/factory_space_14_p4_geo.json --no-annotations
python pipeline2b/geo_to_scenegraph.py  --space factory_space_14 \
    --geo-json pipeline4/out/factory_space_14_p4_geo.json \
    --out-space factory_space_14_p4 --no-structure-filters \
    --merge-fragments --no-audit-prune --reject-bad-geometry
```

`p4_detect.py` is the only new stage — it plugs into the **existing**
pipeline2b labeling (`geo_label_clip.py`) and scene-graph construction
(`geo_to_scenegraph.py`) unchanged, by writing the same stage-A
`<space>_geo.json` + `_geo_points.npz` contract that `geo_cluster.py`
produces. That contract is what carries `floor_y`/`ceil_y`, per-node
`bbox_min/max`, `centroid`, `n_points`, `mean_rgb`, and a `_probe_xyz`
subsample used by table-footprint consolidation. The final
`scene_graph.json` (rooms, areas, edges, `building_yaw_deg`) and the
`/api/scene_graph/{space}/lite` viewer endpoint are the same as every other
pipeline2b variant — open `sg_3d_viewer.html?space=factory_space_14_p4`.

### What `p4_detect.py` does

1. Loads the raw `pointcloud.ply` (meters, +Y up) and rotates it into the
   model's +Z-up frame, shifting the floor (1st-percentile Y) to z=0.
2. Slides a **9 m×9 m** window (4.5 m stride) over the scene; each window
   that has ≥8000 points is run through 3DETR **three** times with different
   random 40k-point subsamples (the "passes" — a cheap test-time ensemble,
   since 3DETR itself is deterministic given fixed input points and FPS
   seeding). `axis_starts()` guarantees the last tile on each axis always
   reaches the far edge of the scene, even when the extent isn't an exact
   multiple of the stride — a plain `np.arange(lo, hi-window, stride)` will
   silently drop a strip at the tail otherwise (this was a real bug caught
   mid-development: an early run only tiled 1 row on one axis and left a
   ~3.7 m band of the room with zero detections).
3. Each window's 256 query predictions are filtered to objectness
   probability > 0.10, size 4cm–6m, and "responsibility" for that window's
   Voronoi cell (so overlapping windows don't produce N duplicate boxes for
   the same object) before being pooled across all tiles.
4. Cross-tile duplicates are merged with axis-aligned 3D NMS (IoU < 0.3).
5. Each surviving box is used to crop the *real* point cloud (not just the
   detector's own coordinates) to get accurate `centroid`/`bbox_size`/
   `mean_rgb` for that node, dropping a thin floor-contact band so boxes
   don't get dragged down to include floor points.
6. Written as `pipeline4/out/<space>_p4_geo.json` (+ `_points.npz`
   sidecar) in the pipeline2/2b stage-A schema.

Key flags: `--window/--stride` (tile size/overlap), `--passes` (ensemble
size), `--min-prob` (objectness threshold), `--nms-iou`, `--max-z-m` (crop
height above floor — ScanNet rooms are ≤~3 m; raise for tall industrial
racking, though the model wasn't trained on anything that tall).

### Tuning recall: window size matters far more than the confidence threshold

3DETR is trained on **whole ScanNet room scans** — `RandomCuboid`, its only
crop augmentation, always crops 50–100% of *that room's own extent*, never
a small fixed absolute size. That means the model's learned position
embeddings and size regressors implicitly assume the input point cloud
spans something like a full room, with a normal furniture-to-room-size
ratio. Feed it a small window and that assumption breaks — measured
directly on this dataset (`min_prob=0.0`, same tile location, varying only
window size):

| window | raw objectness >0.1 | raw objectness >0.5 |
|---|---|---|
| 4 m  | 2   | 2  |
| 6 m  | 18  | 6  |
| 8 m  | 60  | 22 |
| 10 m | 133 | 38 |
| 14 m | 55  | 4  |

Recall peaks around 8–10 m and *collapses* below ~5 m — shrinking the
window for "denser tiling" is exactly backwards and was tried and reverted
during development. `--min-prob` (0.25→0.10) helps too, but it's a much
smaller effect than window size and is safe to lower because domain-gap
objects (pallets, racks, carts — anything with no ScanNet analog) score
lower even when they're real detections; the existing downstream filters
(CLIP structure-drop, 2nd-pass geometric rejection, impossible-person
prune, strip-shaped-box rejection in `geo_to_scenegraph.py`) are what
actually separate real low-confidence detections from noise, not the
detector's own threshold. On `factory_space_14` this raised the final
scene-graph node count from 59 → 295 with a much more plausible industrial
label mix (table/shelf/pallet/storage_rack/cart/machine vs. mostly
chair/table before) and no meaningful increase in near-duplicate boxes
(2 pairs within 10cm out of 379 raw detections).

## Known limitation: domain gap (read this before trusting the labels)

3DETR's pretrained weights only know **ScanNet's 18 indoor furniture
classes** (`detr3d/detector.py:SCANNET_CLASSES` — cabinet, bed, chair,
sofa, table, door, window, bookshelf, picture, counter, desk, curtain,
refrigerator, showercurtrain, toilet, sink, bathtub, garbagebin). It has
**never seen** pallets, shelving racks, industrial machines, carts, or
partition panels — the actual contents of `factory_space_14`. On the test
run, the detector reliably finds real objects and gives good boxes, but
mislabels most of them as `chair`/`table`/`garbagebin` because that's all
it knows.

The fix already built into this pipeline: `det_class`/`det_prob` from
3DETR are written into the geo JSON as informational fields only, and the
node's *actual* label is immediately overwritten by the existing
open-vocab CLIP labeler (`geo_label_clip.py`, industrial vocabulary —
shelf, cart, pallet, machine, etc.) run on the exact same boxes. Think of
3DETR here as a **domain-general box proposer**, exactly the same role
`geo8_track.py` (YOLO-World+SAM2+ByteTrack) plays in the geo9 chain — just
sourced from a learned 3D backbone instead of lifted 2D detections. On
`factory_space_14` (current tuning: 9 m windows, min-prob 0.10) this
produced 379 raw boxes → 343 after CLIP same-label dedup → 335 final
scene-graph nodes with `--no-audit-prune` (see below), with top labels
`table` (114), `shelf` (83), `office_chair` (25), `storage_rack` (21),
`pallet` (20), `person` (17), `cart` (15), `cardboard_box` (12), `cabinet`
(11), `machine` (10), `whiteboard` (6).

If recall on tall/large industrial objects (racks, machines) is too low,
that's most likely because such objects don't resemble anything in
ScanNet's geometry distribution, not a bug in this pipeline's plumbing —
the honest next step would be fine-tuning 3DETR (or training a small
VoteNet head) on a handful of hand-labeled boxes from this dataset, which
`external/3detr/main.py` supports directly since the vendored model is
byte-for-byte the same architecture as upstream.

## Over-segmentation: one physical table/shelf split into many boxes

3DETR predicts a fixed set of query boxes per window; for a **long** object
(an industrial table, a run of shelving) several queries can each grab a
different slice of the same surface. Plain 3D NMS doesn't fix this — NMS
only suppresses boxes that *overlap* enough, and adjacent slices of one
long surface often barely overlap each other at all.

**Fix, and where it lives:** `sg.find_fragment_groups()` in
`pipeline2b/scene_graph.py` — a general scene-graph utility, not
pipeline4-specific. It looks only at each object's **final label** (after
CLIP) and world-frame bbox, and unions same-family objects (table/desk/
counter, shelf/rack/bookshelf, cabinet, machine, whiteboard — see
`FRAG_MERGE_FAMILIES`) whose footprints touch/overlap within 15cm and whose
vertical extents overlap by ≥30%. The resulting groups feed into the
**existing** `build_coarse_groups()` / viewer collapse-expand machinery
(`ui/viewer/scene-graph.js:480-520`, already built for desk/rack
"workstation" units) — merged objects render as one collapsed box by
default, click to expand into the individual detected fragments. Opt in
with `geo_to_scenegraph.py --merge-fragments` (off by default — see below).

**Why it runs on the *final* label, not 3DETR's raw class — a real dead
end hit during development:** the first version of this merged fragments
inside `p4_detect.py`, right after NMS, using 3DETR's own raw ScanNet class
prediction. That's the wrong signal for two compounding reasons:

1. Under the ScanNet→industrial domain gap (see above), 3DETR's raw class
   is unreliable — on this dataset "chair" was the majority raw prediction
   for almost everything, including real tables and shelves that CLIP later
   correctly relabeled. Merging on the raw class either misses the real
   fragments (mislabeled "chair", so excluded from an eligible-class list)
   or, if "chair" is made eligible, chain-merges dozens of merely-nearby,
   genuinely distinct objects into one box via transitive union-find (A
   touches B, B touches C → all merge even though A and C never touch).
2. This was caught concretely: an early version merged **360 of 382** boxes
   into just 25 "groups" — an obvious blowup, since a real over-segmented
   table is usually 2-6 fragments, not dozens. A per-class eligibility list
   (only table/desk/counter/bookshelf/cabinet) got it down to 14 groups /
   36 boxes, which looked more plausible but was still keyed to the wrong
   (pre-CLIP) label — a table fragment 3DETR happened to mislabel "chair"
   would never be found.

Moving the merge to run **after** `geo_label_clip.py`, keyed on the CLIP
label, fixes both: CLIP's labels are far more reliable for this industrial
vocabulary (that's the whole reason 3DETR's own class head is only used as
informational metadata elsewhere in this pipeline too), and it's now a
generic capability any detector-based pipeline can opt into, not something
wired to 3DETR's particular failure mode. The size/member-count safety cap
(`FRAG_MAX_MEMBERS=8`, `FRAG_MAX_DIAG_M=6.0`) stays regardless, as defense
in depth against any future transitive-chaining surprise — a component that
grows implausibly large is **rejected wholesale and left unmerged**, never
force-merged, since a wrong merge is worse than leftover over-segmentation.

On `factory_space_14` (with `--no-audit-prune`, 335 total objects) this
produced 30 fragment-merge groups covering 121 objects, on top of the
pre-existing 22 workstation groups covering 156 objects — 52 total coarse
groups, collapsing 277 of 335 objects down to a single box each by default
in the viewer.

**Note on `--merge-fragments` defaulting to off:** `geo_to_scenegraph.py`
already has an unconditional `graph["coarse_groups"] = []` from an earlier,
unrelated request ("show only individual object boxes, no enclosing
units") that applies to every other pipeline2b/geo9-chain space. Rather
than silently changing that behavior everywhere, `--merge-fragments` is the
one thing that keeps the coarse layer instead of clearing it — every
existing space's invocation is completely unaffected unless it opts in.

**Eligible labels — every label, not a curated list.** The first version
of this restricted merging to a hand-picked allowlist (table/shelf/cabinet/
machine/whiteboard), which meant any *other* label a pipeline's vocabulary
produces (a "cart", a "cardboard box", or a future label like "wheel") was
silently never eligible for merging — the real safety net was always the
xy-gap/z-overlap/size-cap geometry checks, not which labels were allowed,
so the allowlist was pure unnecessary restriction. `_frag_family()` now
returns every label its own family unconditionally (falling back to the
three cross-label semantic families — `COARSE_DESK_LABELS`,
`COARSE_RACK_LABELS`, `COARSE_SEAT_LABELS` — for near-synonym cases like
shelf/storage_rack/bookshelf that CLIP can label inconsistently between
adjacent slices of one physical unit). On `factory_space_14` this raised
fragment-merge coverage from 22 groups/82 objects to 37 groups/138 objects,
newly picking up `cart` (5 groups) and `cardboard_box` (2 groups) that the
old allowlist excluded entirely, on top of `office_chair` (4 groups),
`person` (2 groups — most likely the same walking scan operator recurring
across the floor, a known scan artifact), `pallet` (2 groups),
`table`/`shelf`/`storage_rack`/`cabinet`/`machine`/`whiteboard` (1 each).

**Bug caught while generalizing this:** the union-find pairwise loop
compared family membership with `is not` (identity), which only works when
every call to `_frag_family()` for the same label returns the *same*
Python object — true for the three shared semantic-family constants, but
`frozenset([label])` constructs a brand-new object on every call, so two
objects with the identical label would never compare equal by identity.
This silently zeroed out merging for every label outside the three shared
families (person/pallet/cabinet/machine/whiteboard groups all vanished)
while table/shelf/storage_rack — which route through the shared constants
— kept working, which is exactly the kind of partial breakage that's easy
to miss without checking group counts by label after a change like this.
Fixed by comparing with `!=` (value equality, which frozensets support
correctly) instead of `is not`.

**Naming inconsistency also fixed:** the *pre-existing* workstation-layer
grouping (`_build_anchor_coarse_groups`, unchanged logic, only relevant
here because it emits into the same `coarse_groups` list) named every
group after an internal category constant — `"desk_group"` for *any*
table/desk/workbench/workstation/counter anchor, `"rack_group"` for any
shelf/rack/storage_rack/bookshelf anchor — regardless of the anchor's own
real label. Since individual object labels are already unified to one
canonical spelling per family (`unify_table_labels()` collapses "desk" into
"table" everywhere in `geo_label_clip.py`), this made semantically
identical objects display as `table_group` in one place (fragment-merge
groups, which were already named from the real member label) and
`desk_group` in another (workstation groups, named from the category) —
exactly the "some tables show as desk" inconsistency reported. Fixed by
naming workstation groups after the anchors' own majority real label too,
the same convention fragment-merge groups already used.

### Viewer bug: coarse-group boxes rendered tilted at the wrong angle

The first version of the coarse-group union box computed each member's
extent in **world**-axis-aligned terms (`centroid ± bbox_size/2` in world
x/y/z). That's wrong whenever the room has a non-zero wall yaw (this space
has `building_yaw_deg = -11.31°`): `apply_building_yaw()` stores every
object's `bbox_size` in the **wall-rotated** frame, not world axes — the
per-node fine-level boxes already know this and rotate their geometry back
by `buildingYawDeg` at render time (`ui/viewer/scene-graph.js` ~line 416),
but the coarse-group box skipped that step entirely and unioned raw
world-frame extents. The result: the whole coarse layer rendered visibly
tilted at the wrong angle relative to the (correctly-aligned) fine-level
boxes and the point cloud.

Fixed by rotating each member's `box_center` into the wall-aligned local
frame (`_rotateXZ(x, z, -buildingYawDeg)`, mirroring
`pipeline2b/scene_graph.py`'s `_rotate_xz_deg` exactly) before taking the
union, then rotating the resulting center back to world for display — the
same convention the fine-level boxes use. Also reverted the coarse-box line
width to match the fine level exactly (2.5px) — an earlier pass made it
thicker (4.5px) to fix an invisibility bug, but once actually visible,
thicker-than-fine-level read as too heavy.

### Focused coarse unit now visually distinguishable

Clicking a collapsed coarse unit previously only changed its sphere marker
(bigger + more opaque) — the box itself looked identical whether focused or
not, easy to lose track of which unit you'd actually selected among many
collapsed boxes. The focused unit's box now switches to a highlight color
(cyan, matching the existing focus-highlight convention used elsewhere in
the topdown 2D view) at full opacity, plus a translucent fill (a plain
`THREE.Mesh`/`BoxGeometry`, since it doesn't need fat-line rendering) shown
only while focused.

## CLIP labeling a shape-impossible object (phantom "pallet"s)

24 nodes came out of `geo_label_clip.py` labeled `pallet`, several of them
clearly wrong — e.g. a 0.55×**1.31**×0.63m box (a real pallet, even loaded,
doesn't get that tall). Root cause, verified against the actual CLIP scores
(`node["clip_topk"]` in the geo JSON): CLIP's own raw cosine similarity for
"pallet" on these crops was weak (0.25-0.33, near CLIP's noise floor for
this vocabulary) but still narrowly ahead of the next candidate (e.g.
storage_rack), and the geometric `shape_prior()` fusion in
`geo_label_clip.py` — which correctly computes a ~0 score for "pallet"
against a 1.3m-tall box (its height band caps at 0.65m) — was still letting
that near-zero prior contribute a **15% floor** (`PRIOR_MIN`) to the fused
score. Combined with CLIP's softmax temperature (scale 100, which amplifies
small cosine gaps into large probability gaps), that 15% floor was often
enough for a shape-*impossible* label to still win.

**Fix** (`shape_prior()`, `pipeline2b/geo_label_clip.py`): when a shape
band scores *exactly* 0 — a clear violation, not just "a bit outside the
band" (`_band_score` only clamps to 0 once a violation is well past its
tolerance, e.g. ~1.7x+ the boundary here) — return 0.0 instead of applying
the `PRIOR_MIN` floor, removing that label from candidacy entirely. This is
narrower than lowering `PRIOR_MIN` globally: verified directly that
borderline cases stayed unaffected (a 0.74m and a 0.79m box, only mildly
over the 0.65m pallet-height cap, both correctly stayed labeled `pallet`
since CLIP still favored it strongly there), while the clearly-impossible
cases (1.14-1.31m tall) moved to `storage_rack`/`cardboard_box`/`machine`.
Net effect on `factory_space_14`: pallet 24→20, with the difference
redistributed to storage_rack (18→21), cabinet (8→11), machine (9→10),
cardboard_box (7→12) — plausible reassignments in every case checked.

This is shared code (`geo_label_clip.py` is used by every pipeline2b/geo9
space, not just pipeline4) — the change only ever *removes* a candidate
that was already geometrically impossible for that node, so it shouldn't
regress any space's labeling of objects that actually fit their assigned
shape prior.

## Removing the automatic removal audit for pipeline4

`geo_to_scenegraph.py` runs several automatic geometric "audit" passes
(`_reject_floating_ceiling`, `_reject_structure_2nd_pass` — the
"wall-level" rejections, `_reject_faulty` — "impossible-person"/noise-
fragment, and a strip-shaped-table sliver check) even under
`--no-structure-filters`. These were tuned against pipeline2b's older
DBSCAN-cluster geometry, and on pipeline4's box population they false-
positive on real, plausible objects — verified directly: none of the 23
boxes one run removed overlapped with any fragment-merge candidate (so it
wasn't an over-segmentation side effect), and their sizes were exactly what
a real object looks like (e.g. a `shelf` sized 1.87×1.74×0.33m — an
entirely normal wall-mounted shelf — killed as "wall-level").

`--no-audit-prune` turns off all four passes for a run. CLIP's own
`is_structure` drop (door/window/wall/floor/ceiling labels) and explicit
annotation deletes still run regardless — this only disables the *extra*
automatic geometric second-guessing on top of that. Recommended for
pipeline4 runs; every other pipeline2b/geo9-chain invocation is unaffected
unless it opts in.

## Running on additional spaces

```
S=factory_space_13   # or shinhan_space, factory_space_15, factory_space_16
python pipeline4/p4_detect.py           --space $S
python pipeline2b/geo_label_clip.py     --space $S \
    --geo-json pipeline4/out/${S}_p4_geo.json --no-annotations
python pipeline2b/geo_to_scenegraph.py  --space $S \
    --geo-json pipeline4/out/${S}_p4_geo.json \
    --out-space ${S}_p4 --no-structure-filters \
    --merge-fragments --no-audit-prune
```

**`<space>_p4` must be registered in `spaces.json` before the first run of
`geo_to_scenegraph.py`** (copy the `factory_space_14_p4` entry's shape,
pointing `data_root` at the base space and `out_dir` at
`out/geo_<space>_p4`) — otherwise the auto-scaffolding in
`sg._ensure_ui_space()` can't find a same-`data_root` sibling to copy
`downsampled_web.ply` from, silently falls through to running
`pipeline/downsample_ply.py` (which fails outright for an unregistered
space name), and the viewer copy of `scene_graph.json` never gets written
(`geo_to_scenegraph.py` only copies it once `downsampled_web.ply` exists) —
the canonical `out/geo_<space>_p4/scene_graph.json` is still written
correctly, but the space is unviewable and the server returns 404. Caught
exactly this way when first running the batch below; the fix was adding
the four missing entries to `spaces.json` and re-running
`geo_to_scenegraph.py` (cheap — it doesn't need to redo detection or CLIP
labeling). The **server also needs a restart** afterward — it mounts each
space's static route once at startup from whatever was in `spaces.json`
at process start.

y_up shouldn't need special handling on the *detection* side even for a
`y_up: false` space (`shinhan_space`) — checked directly on this dataset's
raw point cloud: Y span was 3.8m vs ~17m for X/Z, i.e. Y really is the
physical up-axis with a normal room height, matching the assumption
`p4_detect.py`'s `world_to_model()` already makes. `y_up` in `spaces.json`
turned out to be a viewer/camera-display setting (`FLIP_Y`,
`camera.up.set`) consumed by `ui/viewer/app.js`/`streetview.js`, not a
point-cloud sign-convention issue the detection pipeline needs to know
about.

Results on the four additional spaces (all with `--merge-fragments
--no-audit-prune`):

| space | nodes | edges | rooms | coarse groups (frag / workstation) | top labels |
|---|---|---|---|---|---|
| factory_space_13 | 320 | 1690 | 2 | 59 (41 / 18) | table 130, shelf 72, office_chair 23, storage_rack 17, cart 15 |
| shinhan_space | 166 | 682 | 1 | 35 (30 / 5) | shelf 68, cabinet 53, cardboard_box 17, pallet 8, table 8 |
| factory_space_15 | 75 | 403 | 1 | 15 (12 / 3) | cardboard_box 21, pallet 19, storage_rack 15, shelf 13 |
| factory_space_16 | 50 | 199 | 1 | 10 (7 / 3) | shelf 16, cardboard_box 10, table 10, storage_rack 7 |

shinhan_space's label mix (shelf/cabinet-dominated, almost no tables) and
factory_space_15/16's small node counts (75, 50 — these are much smaller
scans: 126 and 127 scanpoints vs. factory_space_14's 340) both look
plausible relative to their scan sizes, but haven't been visually spot-
checked in the viewer the way factory_space_14 was throughout this file —
worth a look before trusting them the same way.

## Coarse (merged) objects are now the real graph nodes

Previously, `nodes` in `scene_graph.json` were always the fine-grained
detector fragments, with `coarse_groups` layered on top purely for the
viewer's collapse/expand display — the actual graph (edges, rooms, areas)
still operated at the fragment level underneath. That meant a table
over-segmented into 5 boxes contributed 5 separate graph nodes and a web of
`same_object_type`/proximity edges between its own fragments, which is
noise: they're not 5 related objects, they're 1 object the detector failed
to merge.

**`sg.build_coarse_objects()`** (`pipeline2b/scene_graph.py`) now collapses
each fragment-merge group (`find_fragment_groups`) into ONE real object
*before* edges, rooms, or areas are computed — objects not in any group
pass through unchanged. `build_edges()` then runs fresh on this reduced
object set, so every relation in the final graph is between real,
deduplicated objects, not detection artifacts. This does NOT apply to
workstation-style grouping (a chair pulled up to a desk) — those are two
genuinely distinct real objects, and collapsing them into one node would
silently delete the chair-desk relation; that grouping still runs as a
viewer-only overlay, just on top of the now-deduplicated node set instead
of raw fragments.

The original fragments aren't discarded — each merged node's constituent
boxes are preserved as lightweight drill-down data in a new top-level
`fragments` field (`{coarse_id, members: [{id, label, box_center,
bbox_size, n_world_pts}, ...]}`), rendered by the viewer as thin outline
boxes shown only while that specific merged node is focused
(`ui/viewer/scene-graph.js`, `nodeFragments`/`fragmentVisuals`). They are
**not** separate graph nodes or edges anymore.

Merged geometry is computed the same way `apply_building_yaw()` computes
any single object's box (rotate the union of member points into the
wall-aligned frame, trim, rotate the center back) — not by unioning the
members' already-rotated boxes as if they were world-axis-aligned, which
is the exact class of bug fixed in the viewer earlier in this file.

**Effect on cross-scan consistency:** factory_space_13/14 and
factory_space_15/16 are pairs of scans of the same physical space, so
their scene graphs should be close to each other. Before this change they
weren't (320 vs. 335 nodes with meaningfully different top labels, e.g.
`pallet`/`person` prominent in one and absent from the other's top-6) —
most of that gap was fragment-level noise (how a table happened to get
sliced varies scan to scan, even when the real furniture is identical).
After collapsing to coarse nodes: 229 vs. 234, with near-identical top
labels and counts (`table` 98/102, `shelf` 49/47, `office_chair` 18/16).
factory_space_15/16 tightened similarly (38 vs. 29, same label ordering
top-to-bottom). Real, deterministic differences between the two scans
still show up as node-count deltas — this isn't a guarantee of an exact
match, only a removal of the fragment-count noise that was drowning out
the real signal.

## Floor/wall boxes and hollow boxes labeled as ordinary objects

Two more classes of bad box, `--reject-bad-geometry` (opt-in, off by
default, non-destructive — routed to the red review layer like every
other precision pass here, not silently deleted):

**Structural mismatch** (`_reject_structural_mismatch`): when a learned
detector's own raw class (`det_class`/`det_prob`, only present on
detector-sourced nodes) was structural (wall/floor/ceiling/door/window)
with reasonable confidence but CLIP relabeled the crop as an ordinary
object, that's a real signal — verified concretely, two "cabinet"-labeled
boxes and a merged "shelf" both had raw `det_class="window"`. But raw
class alone over-triggers: validating this caught a **real false
positive** — a genuine 1.87×1.74×0.33m wall-mounted shelf (the exact
object `geo_label_clip.py`'s own wall-level heuristic was already found
wrongly killing earlier — see the CLIP-labeling section above) also had
`det_class="window"` at 0.42 confidence. A flat, thin, wall-adjacent
rectangle reads as "window" to 3DETR whether it's a real shelf or an
actual window, so the raw class isn't enough on its own. Fixed by adding
the same depth corroboration `geo_label_clip.py`'s own structure-veto
already uses for this exact ambiguity (`WALL_MIN_DEPTH_M = 0.30`: "a
wall/window/door is a thin sheet, a shelf/cabinet/rack has real depth") —
confirmed it cleanly separates the two cases (false-positive shelf: depth
0.33-0.41m, spared; genuinely-bad boxes: depth 0.25-0.28m, still caught).

**Hollow or vertically-skewed boxes** (`_reject_hollow_or_skewed`): a box
can be full of points and still not correspond to one real object.
Checked the actual point cloud inside several flagged boxes directly and
found two distinct failure signatures: (1) essentially zero point mass
anywhere near the box's own center — all points sit in a thin outer
shell, meaning the box mostly spans empty space and is only picking up
edge/background points from whatever it happens to border; (2) the point
mass is heavily skewed toward one vertical extreme (checked directly: one
box's points were 97% concentrated in the bottom 20% of its own reported
height, with the rest — a sparse background-noise tail — reaching a full
meter higher and dragging the box's reported extent up with it). Requires
**both** conditions together: tested a single-signal version first and it
flagged ~24-39% of all nodes in a real scene (open-frame furniture like
chairs and racks routinely has low center-point density too, so hollow-
center alone isn't discriminating), while the combined version flags a
much more plausible 5-10%.

**Known limitation, stated plainly:** this is a best-effort statistical
filter validated against a sample of reported bad boxes, not a precise
classifier verified against every case — some bad boxes will still slip
through (checked directly: a few of the originally-reported examples
weren't caught by either rule and would need their own follow-up look),
and it's intentionally conservative to avoid a repeat of the false
positive above. Non-destructive specifically because of that: anything
it flags is still fully recoverable via Show Removed.

## Root cause found: BBOX_TRIM_Q collapses real dimensions on pipeline4 data

Follow-up review surfaced three seemingly separate symptoms — real boxes
still floor-hugging after the checks above, a real ~1m-deep storage rack
in `factory_space_16` wrongly flagged structural-mismatch (thin, like a
door), and `shinhan_space` showing boxes that reach almost the entire
room's ceiling height — that turned out to share one root cause.
`_trimmed_bbox`'s `BBOX_TRIM_Q = 0.10` (drop the outer 10% per axis) was
calibrated for pipeline2b's own points: sparse, noisy multi-view
backprojection estimates, where trimming the outer 10% removes real
backprojection noise. Verified directly on the flagged rack: its **true**
depth (checked with a gentle 1%/99% trim in the exact wall-aligned frame
`apply_building_yaw` uses) is ~0.85m — a completely normal rack depth —
but the 10%/90% trim collapses it to 0.2m, making a real object read as
a thin wall/door-like sheet to every downstream structure check.

**Fix**: `apply_building_yaw()` (`pipeline2b/scene_graph.py`) and the
initial bbox computation (`geo_to_scenegraph.py`) now use a much gentler
1% trim specifically for detector-sourced objects (marked by a `det_class`
field present, even if `None` — a pipeline4 signal absent entirely for
pipeline2b's own geometric-clustering nodes, which keep `BBOX_TRIM_Q`
completely unchanged).

This exposed one more real issue: two `factory_space_14` boxes I'd
verified earlier as correctly caught (raw `det_class="window"`, CLIP
`cabinet`) turned out to have been caught for the **wrong reason** — the
depth measurement that flagged them as "thin" was itself the same
trim artifact, and their true depth (~0.55m) isn't actually thin at all.
Digging into why CLIP called them "cabinet" revealed the real signal:
**CLIP's own top-ranked guess for both was "wall"** (0.71–0.79 fused — a
decisive margin over cabinet's 0.12–0.13), overridden by
`geo_label_clip.py`'s existing wall-elevation corroboration check because
the box's (trim-artifact-inflated) depth read as "too thick to be a wall."
That corroboration is the right design in general, but it has the exact
same trim-sensitivity problem.

**New check, `_reject_clip_structural_override`**: when CLIP's own top-1
pick for a crop was structural (wall/floor/ceiling/door/window/
partition_panel) with a decisive fused score, but the final label isn't
structural, trust CLIP's original top-ranked verdict over the geometric
override that demoted it — no geometry involved at all, so it isn't
subject to the trim issue affecting the other two checks. Verified this
is safe: the confirmed-good shelf's own top-1 CLIP pick is "shelf" itself
(0.83–0.98 fused) on both its fragments, nowhere near the "wall" pattern,
so it's untouched. On `shinhan_space` (which has genuine floor-to-ceiling
glass partitions 3DETR correctly flagged "window" at high confidence, but
CLIP relabeled shelf/cabinet/curtain) this and the height-fraction check
together raised structural-mismatch coverage from 6 to 31 boxes.

## shinhan_space was rendering upside down by default — a spaces.json bug, not a code bug

The "ceiling boxes" issue above turned out to have two causes, not one.
The structural-mismatch coverage fix above was real, but there was a
second, separate bug: **`shinhan_space`'s `y_up` flag in `spaces.json` was
simply wrong** (`false`, should be `true`). The viewer defaults its
up-axis toggle from this flag (`ui/viewer/app.js`: `FLIP_Y =
!_spaceMeta.y_up`, applied to both the point cloud mesh and every
scene-graph box) — with the wrong flag, the space loaded upside down by
default, requiring a manual toggle to "+Y up" to look right. This was
pointed out directly (confirmed visually: default state shows "-Y up" and
renders inverted) and is why some ceiling fixtures visually looked like
they were "on the floor" — the *data* was always correctly oriented
(verified independently earlier by cross-referencing the scanner's camera
trajectory height against the point cloud's own Y-percentiles — the
camera sits ~0.9m above the low percentile and ~2.2m below the high one,
exactly matching a normal Y-up room), only the **display default** was
inverted. Fixed by correcting `y_up: true` for `shinhan_space` and every
variant sharing its `data_root` (`shinhan_space_p4`, `_geo`, `_geo2`,
`_geo5`, `_full`) in `spaces.json` — this is a pre-existing pipeline2b
misconfiguration, not something introduced by pipeline4, so it corrects
the default view for those older outputs too.

## Correction: the section above was wrong — shinhan_space's raw Y axis genuinely increases downward

The previous section's conclusion ("the data was always correctly oriented,
only the display default was inverted") was **disproven** by direct visual
evidence and is left in place above only as a record of the wrong turn.
The camera-trajectory-height argument used to reach it (scanner height
vs. point-cloud Y-percentiles) sounded plausible but was indirect
statistical reasoning, not a direct check — exactly the kind of shortcut
this project's own standard says not to trust.

**What actually settled it:** rendered real camera crops (`select_views()`
+ `crop_bbox_padded()`) for specific flagged nodes and looked at them.
Node #21 — bbox sitting at what the pipeline called "near the floor" —
is a **ceiling light fixture**. Nodes #67/#187 — bbox sitting at what the
pipeline called "near the ceiling" — are **a table with chairs**. This is
not a display bug; the raw `pointcloud.ply`'s own Y column increases
**downward** for this one scan, so floor_y (numerically low, P1) sits
near the true ceiling and ceil_y (numerically high, P99) sits near the
true floor. Every "boxes span floor-to-ceiling" and "ceiling residue
mislabeled as a floor object" symptom reported earlier traces back to
this, not to the trim issue or the display flag alone (though both of
those were also real, separate bugs, fixed above).

**First fix attempt failed, and why:** the obvious fix — negate
`xyz_w[:, 1]` once, globally, in `p4_detect.py` — broke
`geo_label_clip.py`'s crop selection. `cameras.json`'s camera positions
are defined in the *original*, un-negated raw convention, and
`select_views()` projects object points against those same camera
positions/rotations to pick which crops to show CLIP. Negating only the
object coordinates made that projection geometrically meaningless;
re-rendering a crop for node #187 after this "fix" showed an unrelated
floor scene instead of the expected table. Caught before it shipped by
re-running the exact same crop-rendering check that found the bug in the
first place.

**Final fix:** `xyz_w` is never touched — it stays in the same raw
convention as `cameras.json` and the viewer's `downsampled_web.ply`
everywhere. A new `y_invert` flag in `spaces.json` (set on `shinhan_space`
only) drives a `y_sign` (`-1.0` when set) used *purely internally* in
`p4_detect.py` to build the Z-up frame 3DETR expects
(`world_to_model(xyz_w, y_sign=y_sign)`) and to convert detected boxes
back (`model_box_to_world(..., y_sign=y_sign)`); the floor-band point
filter is the same sign-aware form (`(pts[:,1]-floor_y)*y_sign > 0.05`).
The output geo JSON carries the sign forward as `"y_up_sign"` so every
downstream elevation-sensitive consumer can apply the same correction
without ever needing the coordinates themselves to change:
`geo_label_clip.py`'s `shape_prior()` (the top/bottom/ceiling-gap shape
features feeding CLIP fusion) and its floor/ceiling/ceiling-light
corroboration vetoes in `classify_node()`, plus
`geo_to_scenegraph.py`'s `_reject_floor_hugging`. (The other two
geometry checks, `_reject_structural_mismatch` and
`_reject_implausible_height_fraction`, only ever compare a room-height
magnitude or `bbox_size` — both sign-invariant — so they needed no
change.) Every formula reduces to exactly its original form when
`y_up_sign` defaults to `+1.0`, so no other space is affected.

Also corrected: `spaces.json`'s `y_up` flag really was backwards, just in
the *other* direction from what the previous section concluded — since
the raw data's Y genuinely increases downward, the viewer needs
`y_up: false` (so `FLIP_Y = true` negates it for display), not `true`.
Set on `shinhan_space` and every variant sharing its `data_root`
(`shinhan_space_p4`, `_geo`, `_geo2`, `_geo5`, `_full`).

**Verified end-to-end** by re-running the full chain
(`p4_detect.py` → `geo_label_clip.py` → `geo_to_scenegraph.py`, same
`--merge-fragments --no-audit-prune --reject-bad-geometry` flags as
every other space) and re-rendering fresh crops for representative nodes
in the new output: a `table`-labeled node's crop is a real table+chairs,
a `ceiling`-labeled (structural, dropped) node's crop is genuine ceiling
texture/beams — both crops render cleanly (no garbling), confirming the
camera-projection consistency this design was built to preserve. Final
`shinhan_space_p4` counts: 155 objects carried through CLIP, 128 kept in
the graph after `--reject-bad-geometry` (top labels: `table` 32,
`office_chair` 25, `shelf` 13, `cabinet` 6) — this supersedes the
`shinhan_space` row in the results table above, which predates this fix
and still reflects the inverted reading (`shelf`/`cabinet`-dominated,
almost no tables).

`downsampled_web.ply` needed no regeneration — it was never touched by
either the bug or the fix, since `xyz_w` (what it's built from) was
always raw and untouched.

**Not fixed by this change, still open:** ceiling light fixtures are
still labeled generically `ceiling` (structural, dropped) rather than the
more specific `ceiling light` in VOCAB — this looks like a CLIP/shape-
prior tuning gap (a loosely-boxed ceiling region reads as generic
"ceiling" texture to CLIP unless the crop tightly frames a fixture), not
a symptom of the axis bug, and is a separate follow-up.

## `_reject_floor_hugging`'s thresholds were too tight — widened for chair/table-class detections only

After the axis fix above, the user still reported floor-level boxes in
`shinhan_space_p4` labeled as ordinary objects (`pallet`, `table`,
`cardboard_box`) — e.g. nodes shown at what was clearly floor level.
Rendered camera crops for four reported nodes plus every other short
(<0.35m) non-structural object in the scan (11 total): all 11 are real
junk — floor carpet/tile texture, and (one case) a scan operator's own
head — not furniture. `_reject_floor_hugging` (added earlier this file,
under "Floor/wall boxes...") already exists for exactly this pattern, but
its thresholds (`max_gap_m=0.18`, `max_height_m=0.22`, `max_det_prob=0.22`)
were too tight: these 11 verified-junk boxes sit at up to 0.26m gap,
0.30m height, 0.35 det_prob — just outside the old bounds. All 11 share
one thing: raw `det_class` "chair" or "table" — a real chair or table is
never that short, so a low-confidence chair/table guess this thin is a
strong tell for a scan artifact regardless of the exact confidence value.

**Before widening broadly, checked whether the same wider thresholds would
misfire elsewhere** — rendered crops for every factory_space_14/15 object
that would newly fall inside a widened band. Found two genuine false
positives: a wrapped stack of crates and a stack of cardboard boxes,
both real objects, both with raw `det_class="garbagebin"`. Bins
legitimately come in a huge range of real heights (a squat 0.2m bin is
completely normal), so a low-confidence "garbagebin" guess isn't the same
implausibility signal a "chair"/"table" guess this short is. **Fix**:
widened `max_gap_m`/`max_height_m`/`max_det_prob` to 0.30/0.32/0.40, but
only applied to `det_class in {"chair", "table"}` — every other class
keeps the original, tighter thresholds. Verified this avoids both false
positives while still catching the reported nodes and the rest of the
11-box pattern (re-ran shinhan_space_p4: floor-hugging prune went from 1
to 11 boxes, all four originally-reported nodes now correctly routed to
the red audit layer instead of showing as real furniture). Non-destructive
as always — recoverable via Show Removed if this turns out to be too
aggressive somewhere unverified.

## Table merging was both over- and under-merging — two separate bugs in two separate layers

Reported directly against the viewer (shinhan_space_p4 and factory spaces):
2-6 real, distinct tables sometimes collapsed into one "table unit," while
elsewhere one real table split into two never-merged pieces. The stated
requirement: only merge boxes with the same label whose footprints
actually **overlap** — nothing looser. Two independent bugs were causing
this, in the two different coarse-grouping layers this file already has:

**Bug 1 — the workstation layer was merging distinct anchors, not just
grouping items onto one.** `_build_anchor_coarse_groups()`
(`pipeline2b/scene_graph.py`) had a rule: "two anchors of the same
category (desk/table/counter/...) joined by any support/proximity edge
merge into one unit." Proximity edges come from `build_edges`'s KD-tree
search (radius ~2.3m in this scan) — far too generous a radius to mean
"these are fragments of one object." Verified directly: a shinhan
workstation group had `anchor_ids: [6, 11, 38, 59]` — four separate real
desks, several meters apart, glued into one "table unit (4 objects)"
purely because each was proximity-linked to the next. **Fix**: removed
the same-category anchor-anchor union rule entirely. Each anchor is now
always its own unit; the layer still does what it's for (chair pulled up
to a desk, item resting on a desk), just never merges two distinct desks
with each other. Verified across all 5 `_p4` spaces after the fix: zero
workstation groups have more than one anchor (previously many did).

**Bug 2 — fragment-merge's "touching" test allowed sliver-corner overlaps
to count as a real match.** `find_fragment_groups()`'s old check computed
the X-gap and Z-gap between two boxes' footprints *independently* and
required each to be `<= 0.15m`. This lets two boxes that only clip a tiny
shared corner pass: verified a real case (`5m`/`37` — shinhan) with a
0.037m x 0.195m intersection — about **2%** of either desk's own
footprint — that still counted as "touching" under the old rule, merging
two genuinely separate desks pushed edge-to-edge in a row. For contrast,
a real over-segmented pair from the same scan (`106`/`151` — same desk,
two low-confidence detector guesses) had **59%** mutual footprint
overlap, and a third fragment of that same cluster was **100%** contained
in another's footprint. **Fix**: replaced the per-axis gap test with a
genuine footprint-overlap-area requirement
(`FRAG_XZ_OVERLAP_FRAC = 0.35`, comfortably between the 2% false-positive
and the 59-100% true-positive references) — this is what "boxes that
overlap," the literal ask, means geometrically; "boxes that are merely
adjacent" no longer qualifies.

**Verified end-to-end** by re-running `geo_to_scenegraph.py` for all five
`_p4` spaces (shinhan + factory_space_13/14/15/16): confirmed zero
workstation groups have >1 anchor anywhere, and re-checked the specific
reported shinhan cluster with rendered crops both before and after —
the 2% sliver pair (5, 37) is no longer merged (each is now its own real
table, confirmed by crop), while the 59-100%-overlap cluster
(106/151/184) stays merged (also confirmed real, heavily-overlapping
detector guesses on genuinely the same desk area).

## The 0.35 area-fraction threshold above was still too strict — and, separately, an "isolated pair" needs a second gate

Reported again, more emphatically, after the fix above shipped: tables
were now **under**-merged, especially in the factory scenes — a single
long worktable was ending up as 2-4 separate final objects. Concrete
example given directly (factory_space_13): nodes 416, 88, 382, and 64
should all be one table; rendered crops for all of them and confirmed —
same green worktable, same electronics/cables on its surface, no monitor
actually present anywhere despite one fragment (64) being CLIP-labeled
"computer_monitor."

**Root cause**: the area-fraction check (`ox*oz / min(area_i, area_j)`)
conflates both axes into one number, which is wrong for a long table
sliced by the sliding-window detector into *sequential* pieces along its
length. Two such real pieces (sp13 nodes 283/88) overlap ~70% in the
SHORT (cross-section) axis — matching cross-sections exactly, as
expected — but only ~14% in the LONG (along-length) axis, since each
slice only covers its own stretch of table. The area-product buries the
strong 70% signal inside a weak combined score (~9%) and rejects a real
fragment pair.

**Fix**: switched to a PER-AXIS overlap fraction (each axis's overlap
divided by the smaller box's own extent on that axis), requiring only the
LARGER of the two axes to clear the bar (`FRAG_XZ_OVERLAP_FRAC = 0.45`).
This is exactly "cross-section matches, and there's at least partial
overlap along the length too."

**This reopened bug 2's false positive, and required a second, targeted
gate.** Re-verified nodes 5/37 (the shinhan false-positive pair from the
first fix) directly in the actual frame `find_fragment_groups` runs in
(catching my own error: an earlier check had used the wrong, pre-yaw raw
frame, which made 5/37 look safely low at ~35% — checked again in-situ
and found ~71% on their best axis, HIGHER than several genuine same-table
fragment pairs elsewhere at 54-70%). Per-axis-max cannot separate these
two cases for an isolated pair — the numbers alias. The distinguishing
feature that DOES hold up: **a lone pairwise link has no corroboration; a
3+-member component does** (several independent pairwise links agreeing
is exactly the corroboration a lone pair lacks). Real over-segmented
long-table clusters in this data always end up 3+ fragments, never
isolated 2-box components. Fix: components that reduce to exactly 2
members get a second, stricter gate — the original area-fraction check
(`FRAG_PAIR_AREA_FRAC = 0.30`), re-derived independently of the per-axis
test. 5/37 fails it cleanly (~12%); a verified genuine 2-fragment pair
elsewhere in the same scan (nodes 16/195) clears it easily (~47%).
3+-member components never hit this second gate at all.

**Verified end-to-end** after both fixes, across all 5 `_p4` spaces:
nodes 5/37 stay unmerged (confirmed via the fragment list, not just
logs); the sp13 cluster (168/276/283/325/88) merges into one 5-member
table as required; the previously-ambiguous shinhan cluster (6/11/38/59/
215, flagged as inconclusive in an earlier pass of this fix) now doesn't
merge at all under the stricter combination — resolved toward NOT
merging, which is the safer default given the user's explicit preference
for erring toward under- rather than over-merging.

**Known remaining gap, stated plainly:** nodes 16/195 (the verified
genuine pair near the 168/276/283/325/88 cluster) still don't merge WITH
that larger cluster — there's a real, small physical gap between them
(~0.29-0.34m, no bounding-box overlap at all in that direction), and
every fix in this section still requires genuine footprint intersection
as a precondition before any fraction is even computed. Bridging a true
gap (not just a weak-but-real overlap) would need reintroducing some
absolute distance tolerance, which is exactly what caused bug 2 in the
first place — not reattempted here without a way to distinguish "small
gap in one real long object" from "two adjacent separate objects with a
small gap," which this session did not find. Also investigated the
"mostly empty box" complaint (sp13 nodes 310/366) directly: checked point
fill-ratio and along-axis spatial distribution against the broader
population of ~130 real table objects in the same scan, and neither
signal cleanly separates them from legitimate (if oddly-shaped) real
fragments elsewhere in the same data — did not add a filter on evidence
this weak. These two specific boxes are still visible in the viewer
unmerged/unflagged; removing them individually via the existing
annotation-override mechanism is the more reliable path today.

## Bounding-box overlap heuristics were fundamentally the wrong tool — rewritten around real point-cloud connectivity

Reported after the fix above shipped, sharply: it now under-merged badly,
*worse* than before, especially in the factory scenes — real fragments
of long worktables were scattering into 3-4 separate final objects.
Requested directly: stop approximating from boxes — reproject each
candidate's real points into a top-down 2D view, find the actual
continuous table surface there, and only merge boxes that fall inside
the same one.

**This was the right call.** Every bbox-overlap formula tried in the
section above (per-axis gap, overlap area, per-axis-max, a stricter gate
for isolated pairs) was fitting two numbers — axis-aligned box extents —
that are only a rough proxy for the real question ("is this the same
physical surface?"). Rewrote `find_fragment_groups`
(`pipeline2b/scene_graph.py`) around real point connectivity for the two
families that actually have a flat horizontal surface to reproject
(desk/table/counter, rack/shelf/bookcase — a chair split into a seat-box
and a backrest-box doesn't have one shared "top slice," so seats and
every other label keep the old bbox method, now demoted to a fallback).
`_topdown_fragment_groups` samples each candidate's own top ~15cm of
points (90th-percentile Y, not raw max — see below), rasterizes them all
into one shared (x,z) grid, and connected-component-labels it: two
fragments merge only if their real points form one unbroken surface,
exactly what a person looking straight down at the room would judge by.

Getting this right took three more rounds, each caught by direct
verification, not assumption:

1. **Raw bbox_max is noise-sensitive, same lesson as BBOX_TRIM_Q
   elsewhere in this file.** One real fragment's top-slice, sampled from
   its raw max Y, was a single ~0.4m outlier point — the slice captured
   58 points in a 0.13x0.12m corner instead of the real ~1000-point
   tabletop, so it never reached its true neighbor. Fixed by sampling
   from each object's own 90th-percentile Y instead (10th for a
   `y_invert` space).
2. **An oversized connected component was being discarded WHOLESALE.**
   Checked directly on factory_space_13's ~130 table-family objects at
   once: a genuinely continuous, closely-furnished open floor chain-
   connects into one 32-member blob — correctly too big to be one real
   table, but the original design threw away all 32 objects in it,
   including small legitimate sub-clusters buried inside (this is
   exactly why the previous point above still failed after the raw-max
   fix: the true pair ended up trapped inside this blob and got
   discarded with everything else). Fixed by valley-splitting an
   oversized component's own points at genuine density gaps along its
   dominant axis — recursively, until every piece fits or no further
   valley can be found — reusing the same technique already proven
   elsewhere in this pipeline for tabletop detection
   (`geo_to_scenegraph._valley_split_long`), rather than inventing a new
   one.
3. **A pre-existing, unrelated pipeline-ordering bug was compounding
   both of the above.** `_reject_hollow_or_skewed` (checks whether an
   object's own points actually fill its own box) was running BEFORE
   fragment-merge, on individual pre-merge fragments — but a lone
   fragment of a real object is SUPPOSED to look incomplete on its own;
   that incompleteness is exactly why it needs merging. This was
   silently deleting real fragments before they ever got a chance to
   merge, which is what "detected less real table object parts" was
   actually reporting. Fixed by moving this one check (only this one —
   the others in `--reject-bad-geometry` weren't shown to have the same
   problem, and moving code that isn't verified broken is its own risk)
   to run right after `build_coarse_objects`, on the real, complete
   merged geometry.
4. **Fixing (1)+(2) reopened the ORIGINAL bug-2 false positive.** Once
   topdown connectivity was doing real work, it reconnected nodes 5/37 —
   two separate real desks with literally no physical gap between them
   register as one continuous point-cloud surface too, same failure mode
   as the bbox per-axis-max metric before it. The isolated-pair
   protection (a lone pair has no third fragment corroborating it, so it
   gets a stricter bbox-area cross-check) was scoped to the bbox fallback
   only; extended it to apply to ANY isolated 2-member group regardless
   of which method proposed it.

**Verified end-to-end**, all four fixes together, across all 5 `_p4`
spaces: node 5/37 stay unmerged; the sp13 cluster now correctly covers
6 members (82/88/168/276/283/325, up from the originally-reported 4);
a second real pair in the same room (118/170, confirmed by camera crop —
same tabletop) that used to be trapped inside the discarded 32-member
blob now merges into its own correct 5-member group; every workstation
group still has exactly one anchor everywhere (no regression there);
table counts across all 5 spaces went up, not down, relative to the
under-merged state that prompted this rewrite.

**Known remaining gaps, unchanged from before:** nodes 16/195 (same
scan, same table family) still don't bridge into the larger 6-member
cluster — there's a genuine small physical gap with zero point-cloud
overlap in that direction, which point-connectivity (correctly) doesn't
bridge either; and the "mostly empty box" complaint (sp13 nodes 310/366)
still has no reliable general signal distinguishing it from legitimate
fragments elsewhere — not re-investigated this round, same conclusion
holds. Both remain reachable via the existing annotation-override
mechanism.

## Reverted: back to the original bbox-gap fragment-merge

Despite the verification above, still not satisfactory — requested
directly to revert to the original merge behavior. Reverted
`find_fragment_groups`/`_frag_family`/`FRAG_MERGE_FAMILIES` in
`pipeline2b/scene_graph.py` to the exact pre-session version: same-family
union-find on a per-axis touching gap (`FRAG_XY_GAP_M = 0.15`) plus
vertical overlap (`FRAG_Z_OVERLAP_FRAC = 0.30`), wholesale rejection of
oversized components, no per-axis-max, no overlap-area gate, no isolated-
pair second check, no top-down point-cloud reprojection, no valley-split.
`_topdown_fragment_groups` and `_valley_split_mask` (added earlier this
file) are removed entirely. The call site in `geo_to_scenegraph.py` no
longer passes `y_up_sign` (the original signature never took it — the
gap/overlap test doesn't care which way is "up," only whether two ranges
intersect).

**Update: reverted further, on explicit instruction, to the literal
pre-session state.** `_build_anchor_coarse_groups`' anchor-anchor
proximity union (two distinct desks merge into one workstation) is back
too — the fix for that was in scope for this revert after all. So is
`_reject_hollow_or_skewed`'s original pre-merge position. At this point
every piece of merge-related code touched this session in
`pipeline2b/scene_graph.py` and `pipeline2b/geo_to_scenegraph.py` is
back to exactly what it was before the session started.

## hollow-or-skewed disabled entirely — a real bug in the ORIGINAL code, not something this session introduced

With everything reverted to the literal original, still reported wrong:
many real tables (e.g. factory_space_13 node #118, shown to the user as
red-audit id #200014) still missing. Traced precisely: `#118`'s true
merge partner is `#170` (confirmed earlier in this file by camera crop —
same physical tabletop), but under the ORIGINAL `find_fragment_groups`,
`#118` chain-connects through touching pairs into a **47-member**
component spanning most of the room's tables — one of 7 such oversized
components in this one room. Since that's far past `FRAG_MAX_MEMBERS=8`,
the *entire* 47-member group is discarded wholesale — this is the
`rejected N as implausibly large` line in the logs, and it is exactly as
present in the true original code as in every version tried this
session. All 47 objects, including `#118`, revert to individual,
unmerged fragments. `_reject_hollow_or_skewed` then deletes many of them
for looking geometrically "incomplete" on their own — which they
genuinely are, since each is only a fragment of a larger table, not
evidence of a bad detection.

Presented the two compounding causes plainly and asked how to handle it,
rather than silently changing the merge algorithm again. Chosen fix:
leave `find_fragment_groups`'s wholesale-rejection behavior exactly as
original (still no splitting, still discards oversized components) but
**disable `_reject_hollow_or_skewed` entirely** — it is no longer called
anywhere in `geo_to_scenegraph.py`. Ungrouped fragments that used to
vanish into the red audit layer now surface as ordinary standalone table
boxes instead (more numerous and smaller than a fully-merged table would
be, but visible and real, not silently deleted). Verified: node `#118`
now survives as a normal `table` node, and zero table-labeled entries
remain in the red audit layer across all 5 `_p4` spaces (previously
12-15 per space in the factory scenes).

**Update: two more rounds after this, converging on a middle ground.**

Round A — reported directly, with a screenshot and specific node ids
(G4, G17): the workstation layer's anchor-anchor proximity merge (the
literal original behavior, restored above) was STILL wrong — verified
`#289` and 20 other anchors were glued into one "table unit" purely by
proximity across a walkway. Also reported a precise, explicit rule this
time: "I only want to merge the boxes that are overlapping." Fixed both,
concretely:
- Removed `_build_anchor_coarse_groups`' anchor-anchor union again — an
  anchor never merges with another anchor there, only with directly-
  related non-anchor items (a chair pulled up, an item resting on it).
- Changed `find_fragment_groups`'s touching test from a gap tolerance to
  requiring genuine footprint intersection (`ox > 0 and oz > 0`, no
  minimum fraction) — the literal "boxes that are overlapping" rule.

Verified this fixed the reported cases (`#16`/`#195` now merge as their
own pair, as expected) but surfaced a new problem: in a busy room, pure
pairwise overlap STILL chains transitively across a whole row of
genuinely separate tables (verified: `#118`'s component had 45 members),
and discarding that wholesale meant NOTHING in it merged — reported
directly as "now the tables don't merge, significantly undermerged."

Round B — asked to "find a middle ground" rather than keep flipping
between two extremes. Landed on:
- **Split, don't discard, an oversized component.** Recursively break it
  at its own largest internal centroid gap (whichever of X/Z has more
  spread) — `_split_oversized`/`FRAG_SPLIT_MIN_GAP_M = 0.4`. A component
  that's one real continuous row of touching tables has no gap this wide
  and stays whole (or gets rejected if it's still too big after
  splitting); a chain that crosses a real walkway or aisle splits there
  into the actual sub-clusters. Verified: `#118`'s 45-member component
  now splits into a correct 6-member group (`#91/#118/#147/#166/#263/
  #289`) — exactly the "#118, #170, #263 should join #289's cluster"
  outcome asked for.
- **Re-added the isolated-pair area check**, since "genuine overlap, no
  minimum fraction" reopened the original false positive on its own:
  two separate real desks with no gap between them (`#5`/`#37`) DO
  overlap by the plain intersection test (~0.45m x ~0.12m) — with no
  third fragment to corroborate that one link. A component that only
  ever reduces to an isolated 2-member group (from the initial pass or
  from splitting) now needs `FRAG_PAIR_AREA_FRAC = 0.30` overlap-area to
  survive; a 3+-member component (independent overlapping pairs
  agreeing) doesn't need this extra check. Verified `#5`/`#37` fail it
  (~12%) and stay separate again.

**Where things stand:** `find_fragment_groups` requires genuine
footprint overlap (not proximity/gap tolerance) for same-family
merging; oversized chains split at real gaps instead of being discarded;
isolated pairs get one extra corroboration check. The workstation layer
never merges two distinct anchors with each other, only an anchor with
its own directly-related items. Verified across all 5 `_p4` spaces:
zero workstation groups have more than one anchor, and every specific
node-id case raised this session (`#5`/`#37` separate, `#16`/`#195`
merged, `#118`/`#170`/`#263`/`#289` merged together) resolves as
expected.

## A "person" was joining a table's workstation group — traced to a real detection, wrongly grouped

Reported directly with a concrete counter-example (`#320` and neighbors
in factory_space_13): investigated first whether this was a labeling
bug (CLIP+shape-prior mislabeling an ambiguous, partially-captured
person as furniture) — rendered crops for 6 nearby "table" nodes and all
6 showed the same person's head/shoulders next to a real table edge,
with "person" not even reaching CLIP's top-3 (its shape prior requires
height >= 0.8m; these boxes are only 0.65-0.71m tall, likely a partial
capture of someone bending or walking through the scan). Asked the user
how to fix that — but the actual answer was sharper: node `#393` in the
same room IS correctly labeled `person`, yet was still ending up grouped
with a table (`#376`) as one workstation "unit."

Traced to `_build_anchor_coarse_groups`'s support-child membership rule:
any object with a `standing_on`/`lying_on`/`hanging_on`/`standing_in`/
`lying_in` edge onto an anchor joins that anchor's unit, with **no label
restriction at all** — unlike the seat-proximity rule (scoped to
`COARSE_SEAT_LABELS`), this one accepted anything, including a person
whose box happens to register a support-tier edge onto a table (e.g.
from standing at / leaning over it). **Fix**: `_claim()` now refuses a
"person"-labeled child outright, regardless of relation type or score —
a person is never part of a table/rack "unit." Verified: node `#393`
is now standalone (`group_id: -1`), and zero `person`-labeled nodes are
members of any coarse group across all 5 `_p4` spaces (previously >0 in
several).

**Superseded almost immediately**: the "person"-only exclusion above was
rejected as the wrong shape of fix — "same should apply for other
objects as well!! same for office chairs. dont just hardcode which ones
to exclude. just only merge the objects of the same type only." Also
still under-merged (per the previous round's tradeoff), with a
concretely reiterated example: `#320` and neighbors, and workstation
group "G17" (21 anchors spanning a walkway, from the very first over-
merge report at the top of this section) should split into smaller
non-overlapping units instead of either merging wholesale or not merging
at all.

## Workstation layer rewritten: same-category anchors only, split (not discarded) when oversized

Rewrote `_build_anchor_coarse_groups` around one general principle
instead of a growing list of exceptions: a workstation group NEVER
contains anything outside the anchor's own category (desk or rack) —
not a chair, not a person, not an item resting on it, no per-label
carve-outs. Two things this removes entirely, not just for one label:

- The **seat-proximity rule** (a chair near a desk joined its unit) —
  gone. A chair only ever merges with another chair via
  `find_fragment_groups`' overlap test, never with a table.
- The **support-child rule** (anything `standing_on`/`lying_on`/etc. an
  anchor joined it, with the just-added "except person" patch) — gone
  entirely, not patched further. A person, a box resting on a table,
  anything not itself a desk/rack-category anchor never enters a
  workstation group by any path.

What's left is the ORIGINAL "two same-category anchors merge via any
support/proximity edge" rule (a row of desks is one workstation) —
restored again, since removing it entirely (the prior round) is what
caused the "significantly undermerged" complaint. This time, an
oversized result is **split, not discarded**: reusing
`_split_oversized`/`_oversized` (already built and verified for
`find_fragment_groups`' oversized fragment chains) at the same
thresholds (`FRAG_MAX_MEMBERS`, `FRAG_MAX_DIAG_M`,
`FRAG_SPLIT_MIN_GAP_M`) — recursively break the component at its
largest internal centroid gap until every piece fits or no real gap
remains. The original G17 case (21 anchors across a walkway) now splits
into several small (≤8-member) groups instead of one giant unit or
nothing at all.

**Verified across all 5 `_p4` spaces:** zero workstation groups contain
a `person`, `chair`, or any `COARSE_SEAT_LABELS` member (checked
directly, not just for the one node reported); the only same-group
label mixing left is `shelf`/`storage_rack` — the same rack FAMILY
`find_fragment_groups` already treats as one type elsewhere in this
file, not a violation of "same type only." Largest workstation group
anywhere is now 8 members (the size cap), down from 21+ before
splitting existed.

## Member-count cap was itself the last remaining under-merge cause — removed as a rejection criterion

Reported again, with a screenshot: still significantly under-merged — a
dense cluster of many small overlapping "table" boxes around node #320,
none of them merged. Direct instruction: "merge the overlapping boxes
into one big object... just break down for really large gaps."

Traced precisely (temporary gap-tracing instrumentation, removed after):
node #320 sits in the same original 45-member overlap-chain as the
#118/#170/#263/#289 cluster from two rounds ago. Splitting correctly
peels that cluster off at a real 0.405m gap — but #320's own remaining
23-member "core" has **no internal centroid gap larger than ~0.36m
anywhere** (it's one genuinely continuous mass of overlapping
detections — likely including the repeat person-artifacts found
earlier in this same area). Since it could never split down to
`FRAG_MAX_MEMBERS = 8`, the old rule ("still oversized after splitting
→ reject") threw the entire real merge away, even though every pairwise
overlap in it is genuine.

**Fix**: split `_oversized` (the "should we attempt a split" trigger,
unchanged: member count OR diagonal past `FRAG_MAX_MEMBERS`/
`FRAG_MAX_DIAG_M`) from a new `_still_too_large` (the FINAL acceptance
check after splitting has already been tried) — which checks ONLY a
much larger diagonal (`FRAG_MAX_DIAG_REJECT_M = 12.0`), never member
count. A component that can't be split at any real gap is now accepted
as one merged object regardless of how many detector fragments it's
made of; only a diagonal spanning more than 12m (clearly not one real
piece of furniture) still gets rejected. `FRAG_SPLIT_MIN_GAP_M` itself
is unchanged (0.4) — raising it to chase "really large" would have
undone the #118/#289 split from two rounds ago, since the real
separating gap there measures only 0.405m.

Applied the same fix to both places `_split_oversized` is used
(`find_fragment_groups` and `_build_anchor_coarse_groups`), so the same
diagonal-only final check applies to both the fragment-merge layer and
the workstation layer.

**Verified across all 5 `_p4` spaces:** node #320's 23-member cluster
now merges into one object; the #118/#170/#263/#289 split from two
rounds ago is unaffected (same real gap still separates them); #16/#195
still merge as their own pair; #5/#37 (shinhan) still correctly stay
separate; zero workstation groups contain a person or chair (unaffected
by this change). Fragment-merge coverage rose from 162 to 237 objects
in factory_space_13 with zero "still too large" rejections logged.

## A fourth failure mode: dense, room-spanning boxes with no structural signal at all

Separately, `shinhan_space` still had `shelf`/`cabinet`-labeled boxes
spanning 84-90% of the room's floor-to-ceiling height even after every
check above. None of the existing signals fired: raw detector class was
`desk`/`chair`/`table` (not structural), and CLIP's own top-1 pick was
confidently `shelf`/`cabinet` (not `wall`) at 0.38-0.88 fused — genuinely
different from the window-mismatch pattern. Checked the actual point
histograms directly: these are NOT a sparse noise tail skewing an
otherwise-small object (the pattern behind the other checks in this
file) — points are dense and fairly evenly spread across nearly the
entire room height, thousands per bin throughout.

**New check, `_reject_implausible_height_fraction`**: drop a box whose
height exceeds 78% of the room's own floor-to-ceiling span, regardless of
label or detector metadata — no real shelf/cabinet/table reasonably
spans that much of a room. Excludes labels whose own shape prior
(`SHAPE_PRIORS` in `geo_label_clip.py`) has no upper height bound
(`storage_rack`, `machine`, `ladder`, `pillar`, plus the structural
labels) since genuinely tall industrial racking is expected, not
suspicious, for those — verified directly: a real 87%-of-room-height
`storage_rack` in `factory_space_15` is correctly spared, while
`shinhan_space`'s tall shelves (max kept height fraction dropped from
0.90 to 0.63 after this check) are caught.

## Files

```
pipeline4/
  detr3d/            vendored 3DETR (inference-only, pure-PyTorch ops)
  p4_detect.py        stage 1: sliding-window detection -> geo JSON
  out/                <space>_p4_geo.json + _geo_points.npz
external/3detr/       full upstream repo (reference, training code, docs)
weights/3detr_scannet_masked_ep1080.pth   pretrained checkpoint (fbaipublicfiles.com)
```
