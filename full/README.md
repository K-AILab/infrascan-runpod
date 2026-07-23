# Full pipeline on RunPod Serverless — Stages A / B / C

Goal: the **infrascan-platform UI** (upload a video from your laptop) with the
**heavy GPU work done on RunPod Serverless**, running the *whole* pipeline
(DA3 depth+poses → FastSAM proposals → DINOv2/CLIP embeddings → backprojected
**object point cloud** → merge → index), not just stage 0.

```
[laptop] → infrascan UI (always-on, CPU)
             on "process": POST video → RunPod Serverless GPU endpoint (this image)
                                          runs full pipeline, uploads result.zip to bucket
             ← result_url ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ┘
```

> **Status: v1, not yet live-tested.** Deploy needs *your* RunPod account,
> *your* storage bucket, and an amd64 build (RunPod builds it). We validate the
> pipeline logic locally first (see “Local validation”), then iterate on real
> deploy errors — the handler names the failing stage + stderr on any failure.

---

## Stage A — the GPU worker (this folder)
- `Dockerfile` — clones `infrascan-platform@fix/da3-serpentine-pose-order`, installs
  the 5-model stack (DA3, FastSAM, DINOv2, CLIP, open3d/faiss), bakes FastSAM weights.
- `handler.py` — creates a headless space, runs stitch→frames→views then the runner
  stages 00b→04, zips the outputs, uploads to the bucket, returns the URL.
- `storage.py` — S3/R2 upload/download.

**Deploy (same GitHub flow as the POC, but a GPU endpoint):**
1. Commit `full/` and push.
2. RunPod → Serverless → New Endpoint → import this repo, **Dockerfile path `/full/Dockerfile`**.
3. Pick a **GPU** worker (≥24 GB VRAM recommended; DA3 GIANT + FastSAM + DINOv2).
   Container disk ≥ 30 GB. Attach a **Network Volume** mounted at `/runpod-volume`
   (so the multi-GB DA3 download caches once).
4. Add the storage env vars (Stage B) as endpoint **secrets**.

## Stage B — storage (env/secrets)
Set on the endpoint (Cloudflare R2 shown; AWS S3: omit `S3_ENDPOINT_URL`):
```
S3_ENDPOINT_URL = https://<accountid>.r2.cloudflarestorage.com
S3_BUCKET       = infrascan-results
S3_ACCESS_KEY   = <key>
S3_SECRET_KEY   = <secret>
S3_REGION       = auto
S3_PUBLIC_BASE  = https://<your-r2-public-domain>   # optional; else presigned URL
```
`handler.py` uploads `results/<slug>.zip` and returns a fetchable URL.

## Stage C — wire the infrascan UI to the endpoint (in a FORK, not the CEO's repo)
The app’s pipeline currently runs locally. To offload to serverless, in a **fork**
of infrascan-platform, replace the GPU stage dispatch (where `start_processing` /
the runner kicks off) with a call to the endpoint:
```python
# fork: app/routes/upload.py  (start_processing) — sketch
import requests, time
def dispatch_to_runpod(slug, video_url):
    ep = os.environ["RUNPOD_ENDPOINT"]; key = os.environ["RUNPOD_API_KEY"]
    h = {"Authorization": f"Bearer {key}"}
    jid = requests.post(f"https://api.runpod.ai/v2/{ep}/run", headers=h,
        json={"input": {"video_url": video_url, "slug": slug}}).json()["id"]
    while True:
        s = requests.get(f"https://api.runpod.ai/v2/{ep}/status/{jid}", headers=h).json()
        if s["status"] == "COMPLETED": return s["output"]      # {result_url, ...}
        if s["status"] == "FAILED":    raise RuntimeError(s)
        time.sleep(5)
```
Then the UI downloads `result_url`, unzips into the space’s data dir, and the
rest of the app (viewer, search index) works unchanged. Keep the CEO’s repo
untouched — do this on a fork.

## Local validation (do this first — needs the GPU free)
Before trusting the serverless flow, run the *same* pipeline via the local app on
this GB10 box (the app is already up on :8090) to confirm the exact sequence and
that it produces the point cloud + index:
```
INFRASCAN_DB_PATH=... INFRASCAN_DATA_ROOT=... \
  python -m pipeline.runner --slug <slug>     # after upload+preflight+process
```
Whatever the local run needs (owner_id, exact pre-DA3 layout) is then mirrored 1:1
in `handler.py`.

## Cost note
GPU serverless bills per-second of execution (DA3+FastSAM+DINOv2 over many views
is minutes per scan). Min workers 0 = no idle cost; the UI stays cheap/always-on.
