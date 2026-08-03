# Splat Analyzer

Find objects in a 3D Gaussian Splat — no manual annotation, no training required.

Give it a splat file (`.ply` or `.spz`) and a plain-English prompt like
`"chair, table, monitor"`, and it returns a 3D bounding box (position + size)
for each object it finds.

```jsonc
// in:  scene.ply  +  prompt "chair, table"
// out: interactions.json
{
  "objects": [
    { "label": "chair", "position": { "x": -4.0, "y": 1.3, "z": -2.8 },
                         "size":     { "x":  2.1, "y": 2.1, "z":  2.1 } }
  ]
}
```

## How it works

1. **Render** — synthetic camera views (RGB + depth) are rendered around the
   splat by a density-aware sampler.
2. **Detect** — the OWLv2 open-vocabulary model finds the prompt's objects in
   every frame (2D boxes).
3. **Lift to 3D** — each 2D box is back-projected into 3D using the per-pixel
   depth map.
4. **Cluster** — detections are fused across all views into one 3D box per
   object.
5. **Output** — `interactions.json` with a label, position, and size per
   object.

## Run on an NVIDIA GPU

```bash
python -m venv .venv && source .venv/bin/activate

# 1) Install torch matching your CUDA (see https://pytorch.org). Example, CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) Set your GPU arch (see table below), then install the rest:
export TORCH_CUDA_ARCH_LIST=8.9          # e.g. 8.9 for RTX 40xx / L40S
pip install -r requirements.txt

# 3) Run
python run_local.py --ply scene.ply --prompt "chair, table" --quality high
```

`gsplat` compiles CUDA extensions on install, so torch must be installed first.

## Run on a Mac (Apple Silicon)

Renders on the Apple GPU via Metal — no NVIDIA hardware needed.

```bash
./install_mac.sh                 # creates .venv, installs deps, builds the Metal renderer
source .venv/bin/activate
python run_local.py --ply scene.ply --prompt "chair, table" --quality low
```

Requirements: macOS on Apple Silicon + Xcode command-line tools
(`xcode-select --install`). The first build compiles a Metal shader (a few
minutes, one time).

Notes:
- Rendering and detection both use the Apple GPU. If you hit an
  "op not implemented for MPS" error, set `WMD_DEVICE=cpu` to force CPU
  detection.
- It's slower than a CUDA box, so start with `--quality low`.

## Output

Both modes write to `./out_<name>/` (override with `--job_dir`):

- `interactions.json` — detected objects (label, position, size)
- `frames/` — rendered RGB + depth views
- `transforms.json` — camera poses

Options: `--quality {low,medium,high}`, `--score_threshold`, `--min_votes`,
`--min_vote_frac`, `--min_peak_score`, `--max_per_label`, `--max_object_diag`,
`--max_height_z`, `--min_height_z_light`, `--cross_label_overlap_frac`,
`--n_positions`. Run `python run_local.py --help` for details.

## Detection parameters (defaults)

| Parameter | Default | Description |
|---|---|---|
| `quality` | `medium` | Camera coverage: `low` (24 views) · `medium` (90) · `high` (192) |
| `score_threshold` | `0.12` | Per-frame OWLv2 confidence cutoff |
| `min_votes` | `8` | Frames an object must appear in to be kept |
| `min_peak_score` | `0.40` | Best single-frame confidence required |

A high-quality job with the default `n_positions` (8) leaves real coverage
gaps in larger rooms; raising `--n_positions` to e.g. 64 (1536 rendered
frames total) closes that gap at the cost of longer runtime — see the parent
repository's README for the exact settings used to reproduce its results.

## Supported formats & splats

Works with virtually any standard Gaussian Splat — `.ply` or `.spz`:

- **`.ply`** — standard 3DGS (Nerfstudio, Gaussian Splatting, gsplat)
- **`.spz`** — Niantic / World Labs compressed format (v1–v3)

> **Check orientation first.** Make sure your splat is right-side up before
> running. Some exporters flip the vertical axis — if the scene loads upside
> down, re-orient it first, otherwise camera placement and detections will be
> off.

## Hardware

Developed and tested on an NVIDIA L40S (48 GB); a `high`-quality job takes
~3-5 min there. Measured peak VRAM ~7 GB (OWLv2 dominates). A local NVIDIA
box needs roughly:

- **GPU**: 8 GB VRAM minimum (12 GB comfortable)
- **RAM**: 16 GB min, 32 GB recommended
- **Disk**: ~20 GB free
- **CUDA**: 11.8 or 12.x

Set `TORCH_CUDA_ARCH_LIST` to match your GPU:

| GPU | Arch |
|---|---|
| RTX 3070/3080/3090 | `8.6` |
| RTX 4080/4090 · L40S | `8.9` |
| A100 | `8.0` |
| H100 | `9.0` |

## Project structure

```
splat_analyzer/
├── pipeline.py              # Pipeline core — OWLv2 detection, clustering, output
├── render_cameras.py        # Camera placement, depth maps, SPZ→PLY converter
├── renderers/                # Pluggable GPU renderers
│   ├── gsplat_backend.py    #   CUDA (nerfstudio gsplat)
│   └── gsplat_metal_backend.py  # Apple Metal (gsplat-mps)
├── config.py                # Shared defaults + quality presets + renderer choice
├── run_local.py             # Local CLI entry point
├── rotate_and_export.py     # Convert results into the parent pipeline's box format
├── install_mac.sh           # Apple Silicon setup
├── requirements.txt         # CUDA deps
├── requirements-mac.txt     # Apple Silicon deps
```

## Tech stack

Python · PyTorch · gsplat / gsplat-mps · OWLv2

## Acknowledgements

Built on:

- **[Boxer](https://github.com/facebookresearch/boxer)** (Meta FAIR) — the
  inspiration for this approach. Boxer lifts open-vocabulary 2D detections
  (OWLv2) into fused 3D bounding boxes from Project Aria captures; this tool
  adapts that idea to Gaussian Splats by rendering synthetic views of any
  `.ply`/`.spz` splat and lifting the 2D boxes with camera-projection
  geometry instead.
- **[OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)**
  (Google) — open-vocabulary 2D object detection.
- **[gsplat](https://github.com/nerfstudio-project/gsplat)** (Nerfstudio) and
  **[gsplat-mps](https://github.com/iffyloop/gsplat-mps)** — Gaussian Splat
  rasterization on CUDA and Apple Metal.

## License

MIT — see [LICENSE.md](LICENSE.md).
