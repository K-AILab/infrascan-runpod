# Pipeline

Offline batch jobs that turn a raw 360° upload into:
- a navigable digital twin (data/<slug>/)
- a search index (out/<slug>/)

These never serve traffic. They write files and update `spaces.status` in
the platform database.

## Stages (run in order)

| # | Script                  | What it does                                       |
|---|-------------------------|----------------------------------------------------|
| 0 | `00b_gen_da3.py`        | Monocular depth + camera poses                     |
| 1 | `01_propose.py`         | Automatic object proposals (per-view bbox+mask)    |
| 2 | `02_embed.py`           | Visual embeddings per proposal                     |
| 2b| `02b_match_views.py`    | Within-scanpoint dedup via feature matching        |
| 3 | `03_backproject.py`     | Per-row world position via depth + DBSCAN cleanup  |
| 3b| `03b_merge_groups.py`   | Cross-scanpoint merge by embedding + 3D distance   |
| 4 | `04_index.py`           | Build the FAISS search index                       |
|   | `gen_topdown.py`        | Floor-plan PNG + bounds.json                       |
|   | `downsample_ply.py`     | Browser-ready downsampled point cloud              |

## Convenience: `add_space`

For the typical 360° video flow:

```bash
python -m pipeline.add_space \
    --slug my-floor \
    --source ~/captures/my_floor.mp4 \
    --title "My Floor"
```

That:
1. Saves the upload under `data/<slug>/uploads/`
2. Runs stages 0 → 4 + topdown + downsample
3. INSERTs / updates the `spaces` row
4. Status flips: uploading → processing → ready

## Model weights

The pipeline reads model weights from `INFRASCAN_PIPELINE_MODELS` (default
`./external`). First run downloads them; cached after that.

See `docs/STORAGE.md` for the on-disk layout this pipeline writes.
