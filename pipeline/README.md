# infrascan-platform

Drop a 360° video (`.mp4` or `.insv`) in, get out a queryable 3D digital twin:
point cloud, per-view depth, camera poses, object index. Serves the whole
thing behind a login-gated web app.

**Onboarding hurry?** Run `scripts/onboard.sh` for a scripted setup that
takes you from an empty machine to a working local dev server in ~10 min
(+ time to download DA3 weights).

---

## Requirements

- **Linux + NVIDIA GPU** (any modern CUDA card; DA3 uses ~10 GB VRAM).
- **conda** (miniconda / miniforge — the pipeline env is provisioned as
  a conda env named `infrascan`).
- **~15 GB free disk** for the code + ML weights + one test video.
- **ffmpeg**, **git** (usually already installed).

Optional but nice:
- **cloudflared** if you want a public URL (`app.your-domain.com`).
- **tmux** for keeping the worker + server running across sessions.

---

## Quick start (scripted)

```bash
git clone --recurse-submodules https://github.com/ch-ho00/infrascan-platform.git
cd infrascan-platform
./scripts/onboard.sh
```

`onboard.sh` runs the steps below for you: sets up the conda env, fetches
DA3 model weights, initialises the local DB, creates an admin user, and
prints the URL + admin creds. When it finishes, you can start the server
+ worker and begin uploading videos.

---

## Quick start (manual, step by step)

If you'd rather run each step yourself:

### 1 · Get the code

```bash
git clone --recurse-submodules https://github.com/ch-ho00/infrascan-platform.git
cd infrascan-platform
```

The `--recurse-submodules` matters — the viewer UI lives in a submodule
(`ui/legacy-viewer` → K-AILab/3d-object-tagging).

### 2 · Set up the conda env

```bash
# Create + activate
conda env create -n infrascan -f environment.yml
conda activate infrascan
pip install -e .
```

### 3 · Fetch the DA3 model weights

The pipeline needs ~6.7 GB of ML checkpoints that don't live in git
(they exceed GitHub's file-size limit). See
[`pipeline/da3_streaming/README.md`](pipeline/da3_streaming/README.md).
Short version:

```bash
# ask a teammate for the current canonical location, then e.g.
rsync -av chan@dgx-kail:/path/to/da3_weights/ \
  pipeline/da3_streaming/weights/
```

### 4 · Initialise the local database + admin user

```bash
bash scripts/bootstrap_dev.sh
# → creates ./data/infrascan.db
# → creates admin user  admin@infrascan.local / infrascan-admin
```

Add `--with-legacy-icc` if you want the built-in `icc1/2/3` demo spaces
(needs the legacy on-disk artefacts symlinked in — dev-box only, safe to
skip).

### 5 · Run the app + the worker

Two long-running processes, one for the web app and one for the offline
pipeline. Put them in a tmux session so they survive disconnects:

```bash
# Terminal 1 — web app
bash scripts/run_dev.sh
# → uvicorn on http://localhost:8070

# Terminal 2 — pipeline worker (picks up any 'processing' space in the DB)
python -m scripts.worker
```

Open http://localhost:8070 and log in with the admin creds from step 4.

### 6 · Drop your first video

Two ways:

**Via the web UI** — click *Upload* on the my-spaces page, pick your
`.mp4` or `.insv`, watch the progress bar. The worker runs the whole
pipeline automatically:

```
video
  → stitch (skipped for .mp4, dfisheye→equirect for .insv)
  → frames        extract every 2nd video frame
  → views         perspective sample at 3 pitches × 12 yaws per frame
  → da3           Depth Anything v3: dense depth + camera poses + pcd
  → propose       object proposals per view
  → embed         DINOv2 embeddings
  → match         within-scan-point dedup (SuperPoint + LightGlue)
  → backproject   2D detections → 3D positions
  → merge         cross-scan-point dedup
  → index         FAISS index
  → topdown       floor-plan render
  → downsample    voxel-downsample cloud for the browser
```

When it's done the space shows up as *ready* and the viewer opens.

**Or programmatically** — drop the file into
`data/<slug>/uploads/input_equirect.mp4`, register the space with
`status=processing`, and the worker picks it up:

```bash
python -m scripts.register_space \
  --slug my-first-scan --title "My first scan" \
  --owner-email admin@infrascan.local --status processing \
  --n-views 0 --n-scanpoints 0

mkdir -p data/my-first-scan/uploads
cp ~/captures/my-scan.mp4 data/my-first-scan/uploads/input_equirect.mp4
```

The worker polls the DB every ~10 s.

---

## Repo layout

```
app/            FastAPI app (login, /spaces, /search, /upload, viewer, admin)
shared/         CSS design tokens + base components
ui/
  templates/    Jinja templates
  static/       app-side JS/CSS
  legacy-viewer/  ← git submodule (K-AILab/3d-object-tagging) — the 3D viewer
pipeline/       Offline batch pipeline stages (see below)
  da3_streaming/  DA3 model + inference (weights NOT tracked, see its README)
scripts/        CLI: bootstrap, worker, register/remove spaces, migrations
infra/          docker + terraform + cloudflared config templates
docs/           ERD, architecture, migration notes
tests/          pytest — smoke + auth
```

Pipeline stages live one per file in `pipeline/`:

- `_00_stitch_insv.py` — `.insv` → equirect `.mp4` (ffmpeg dfisheye)
- `00_video_to_img.py` — video → frames (every_n=2)
- `00a_sample_views.py` — frames → perspective views (3-pitch × 12-yaw)
- `00b_da3_streaming.py` — views → depth NPZs + `cameras.json` + `pointcloud.ply`
- `01_propose.py` — FastSAM proposals
- `02_embed.py` — DINOv2 embeddings
- `02b_match_views.py` — within-scanpoint dedup (SuperPoint + LightGlue)
- `03_backproject.py` — 2D → 3D positions
- `03b_merge_groups.py` — cross-scanpoint dedup
- `04_index.py` — FAISS build
- `gen_topdown.py` — floor plan
- `clean_and_downsample.py` — cloud for browser

---

## Data layout (per space)

Once a space finishes processing, its files land in these matching-shape
directories:

```
data/<slug>/
  uploads/input_equirect.mp4     the input video
  frames/000000.jpg              equirect frames
  views/000000_pz000_y000_normal.jpg   perspective views (5328 per scan typical)
  depth/frame_0.npz              {image, depth, conf, intrinsics} per view
  cameras.json                   {id, pos, R, pano, xy, frame, pitch, yaw} per view
  intrinsics.json                {fx, fy, cx, cy, matrix_K, width, height}
  pointcloud.ply                 → symlink to da3 combined cloud

out/<slug>/
  proposals.jsonl                object proposals
  embeddings.npy                 DINOv2 embeddings
  index.faiss                    FAISS index for click-to-find-similar
  web/downsampled_web.ply        cloud for the browser
  web/topdown.png                floor-plan render
```

The `data/<slug>/{cameras.json, intrinsics.json, pointcloud.ply, views/, frames/, depth/}`
subset matches the intern data-handoff format (see the
[`scan13-16` bundle](docs/ARCHITECTURE.md#intern-data-format)).

---

## Configuration

Environment-driven. Defaults work for local dev; production reads
`.env`.

| Var | Default | Meaning |
|---|---|---|
| `INFRASCAN_DB_PATH` | `./data/infrascan.db` | SQLite path |
| `INFRASCAN_DATA_ROOT` | `./data` | Per-space data dir |
| `INFRASCAN_OUT_ROOT` | `./out` | Per-space index dir |
| `INFRASCAN_COOKIE_DOMAIN` | (blank) | `.infrascan-ai.com` in prod |
| `INFRASCAN_SECRET_KEY` | `dev-key-replace-in-prod` | Required in prod (signs session cookies) |
| `INFRASCAN_FFMPEG` / `INFRASCAN_FFPROBE` | conda env binaries | Override to use system ffmpeg |

---

## Further reading

- [`pipeline/da3_streaming/README.md`](pipeline/da3_streaming/README.md) — DA3 setup + weight download.
- [`docs/USER_MODEL.md`](docs/USER_MODEL.md) — users · spaces · roles · sharing.
- [`docs/ERD.md`](docs/ERD.md) — the DB entity-relationship diagram.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — DNS, processes, deployment.
- [`docs/MIGRATION.md`](docs/MIGRATION.md) — coming from the legacy `infrascan` tree.

## Troubleshooting

- **Push says HTTP 500** — you're probably trying to push the DA3 weights.
  They're `.gitignored`; make sure `pipeline/da3_streaming/weights/` isn't
  staged.
- **`ui/legacy-viewer/` is empty after clone** — you forgot `--recurse-submodules`.
  Fix: `git submodule update --init --recursive`.
- **Worker sits at `da3 · 0.0 · starting…`** — DA3 needs a GPU + weights.
  Check `nvidia-smi` and `pipeline/da3_streaming/weights/model.safetensors` exists.
- **`app.infrascan-ai.com` 502** — the platform-srv process died. Restart it
  with `bash scripts/run_dev.sh` (and cf-tunnel if you're using one).
