# Storage — how the per-space files are laid out

Short answer: **the slug is the join key.** Spaces have a row in SQLite for
metadata (title, status, owner, scanpoint count). Their **files live on
disk** under deterministic paths keyed by slug. The database doesn't store
any file paths.

This file documents the layout, why it's that way, and how each row in the
DB maps to bytes on disk.

## The two roots

Two top-level directories, configurable via environment variables:

```
INFRASCAN_DATA_ROOT   default ./data    raw + processed source per space
INFRASCAN_OUT_ROOT    default ./out     pipeline outputs per space (indexable
                                        artefacts the server reads at query time)
```

These are intentionally separate so that you can keep `data/` on slower bulk
storage and `out/` on something faster (SSD) if needed. By default they sit
side by side in the repo for development.

## Layout per space

```
data/<slug>/                      ← raw + processed source
├── uploads/                      ← user's original upload(s) (.mp4, .insv, .e57…)
│   └── input.mp4
├── preflight_frames/             ← 10 thumbnails sampled by Tier 3
│   ├── frame_00.jpg                preview gallery on /spaces/<slug>/preflight
│   └── ...
├── frames/                       ← equirect panos per scanpoint, written by pipeline
│   ├── 000000.jpg
│   └── ...
├── views/                        ← 504×504 perspective JPGs sampled from frames
│   ├── 000000_pz000_y000_normal.jpg
│   └── ...
├── depth/                        ← per-view depth + camera intrinsics
│   ├── frame_0.npz               keys: depth (float32 H×W), conf (float32 H×W),
│   └── ...                       intrinsics (float32 3×3)
├── cameras.json                  ← per-view world position + R matrix (C2W)
├── intrinsics.json               ← shared K matrix
├── pointcloud.ply                ← full-res cleaned cloud (gravity-corrected)
└── poses.txt                     ← 4×4 C2W matrices, one per line per view

out/<slug>/                       ← search artefacts + browser-served assets
├── proposals.jsonl               ← one line per object proposal (bbox, view_id, mask)
├── embeddings.npy                ← float32 (N, 768) — one row per proposal
├── object_ids.npy                ← int32 (N,) — group id from cross-view dedup
├── metadata.json                 ← lookup: proposal_id → {view_id, bbox, ...}
├── index.faiss                   ← the search index (IndexFlatIP). Server mmaps this.
└── web/                          ← browser-served, auth-gated by /spaces/<slug>/asset/
    ├── downsampled_web.ply       ← coarse cloud for the 3D viewer
    ├── topdown.png               ← minimap render
    ├── bounds.json               ← topdown axis-aligned bounds
    ├── cameras.json              ← symlink → ../../data/<slug>/cameras.json
    ├── frames/                   ← symlink → ../../data/<slug>/frames
    └── views/                    ← symlink → ../../data/<slug>/views
```

## How the database row maps to disk

The `spaces` table has only what's useful for **lookups, listings, and
permission checks**:

| column        | example                | what it's for                            |
|---------------|------------------------|------------------------------------------|
| `slug`        | `"icc1"`               | the join key — finds everything on disk  |
| `title`       | `"ICC Office Bldg…"`   | shown in lists and headings              |
| `owner_id`    | uuid                   | permission check                         |
| `status`      | `"ready"`              | is the data on disk consumable yet?      |
| `n_views`     | 25056                  | shown on cards                           |
| `n_scanpoints`| 696                    | shown on cards                           |

When the server needs a file, it computes the path:

```python
from app import spaces as space_repo

# read the search index
out_dir = space_repo.out_dir(slug)            #   ./out/icc1/
faiss_idx = out_dir / "index.faiss"

# read the depth for a view
data_dir = space_repo.data_dir(slug)          #   ./data/icc1/
depth_npz = data_dir / "depth" / f"frame_{i}.npz"

# serve a browser asset to a permitted user
web = space_repo.web_dir(slug)                #   ./out/icc1/web/
# … / Data_/ downsampled_web.ply, etc.
```

No file paths in the DB. No file inventory in the DB. The disk *is* the
inventory; the DB is just metadata about it.

## Why this design

- **The pipeline writes thousands of files per space.** Tracking each one
  as a row would be operational overhead with no upside — the pipeline
  already knows the canonical naming and the server already knows where to
  look.
- **The slug is short, stable, and URL-safe.** `app.infrascan-ai.com/
  spaces/icc1/` is the same icc1 as `./data/icc1/`. No translation step.
- **`spaces.status` is the source of truth for "is it consumable yet."**
  The viewer / search / API endpoints all gate on it; they don't probe the
  filesystem to decide what's ready.
- **Per-space tear-down is trivial.** `rm -rf data/<slug> out/<slug>` plus
  `DELETE FROM spaces WHERE slug = ?` and the space is gone. Nothing
  scattered.

## What gets uploaded and what's processed

```
user                pipeline                    server
───────             ───────                     ───────
[upload]   ──►  data/<slug>/uploads/input.mp4
                                          │
                                          ▼
                Frame extraction
                                          │
                                          ▼
                data/<slug>/frames/000000.jpg
                                          │
                                          ▼
                View sampling (3 pitches × 12 yaws)
                                          │
                                          ▼
                data/<slug>/views/000000_pz000_y000_normal.jpg
                                          │
                                          ▼
                Depth + camera poses (DA3)
                                          │
                                          ▼
                data/<slug>/depth/frame_0.npz
                data/<slug>/cameras.json
                data/<slug>/poses.txt
                data/<slug>/pointcloud.ply
                                          │
                                          ▼
                Object proposals + embeddings + dedup + index
                                          │
                                          ▼
                out/<slug>/proposals.jsonl
                out/<slug>/embeddings.npy
                out/<slug>/object_ids.npy
                out/<slug>/index.faiss
                out/<slug>/metadata.json
                                          │
                                          ▼
                Web assets (downsample + topdown)
                                          │
                                          ▼
                out/<slug>/web/downsampled_web.ply
                out/<slug>/web/topdown.png
                                          │
                                          ▼
                UPDATE spaces SET status='ready'   ────►  user sees "ready"
                                                          in /spaces/
```

So when you upload `my_floor.mp4`, what's saved is:

1. `data/my-floor/uploads/my_floor.mp4` — your file, untouched
2. The pipeline runs (out of band) and writes everything else under
   `data/my-floor/` and `out/my-floor/`
3. The `spaces` row flips from `uploading` → `processing` → `ready`
4. The server can now answer queries about it

## Removing or migrating a space

```bash
# unregister but keep the files on disk
sqlite3 data/infrascan.db "DELETE FROM spaces WHERE slug = 'my-floor';"

# wipe the bytes
rm -rf data/my-floor out/my-floor

# move the bytes to a different machine
rsync -a data/my-floor/ out/my-floor/ remote:/var/infrascan/{data,out}/my-floor/
# then on remote: INSERT into spaces (slug, …) VALUES (…)
```

## Estimating disk per space

For the icc-scale spaces (≈ 700 scanpoints, ≈ 25 k views, ≈ 1 M proposals):

| artefact                | size  |
|-------------------------|-------|
| `uploads/input.mp4`     | ~ 4 GB|
| `frames/` (equirect)    | ~ 1 GB|
| `views/` (504×504 JPG)  | ~ 6 GB|
| `depth/` (.npz)         | ~ 14 GB|
| `pointcloud.ply`        | ~ 2.5 GB|
| `proposals.jsonl`       | ~ 0.8 GB|
| `embeddings.npy`        | ~ 4.0 GB|
| `index.faiss`           | ~ 4.0 GB|
| `web/downsampled_web.ply` | ~ 150 MB|
| **total per space**     | **≈ 36 GB** |

Three icc spaces ≈ 100 GB on this dev box. Plan storage accordingly.
