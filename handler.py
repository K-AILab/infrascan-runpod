"""Stage A — full-pipeline RunPod serverless GPU worker.

Runs the ENTIRE infrascan video->data pipeline (not just stage 0): the CEO's
platform is cloned into the image (fix/da3-serpentine-pose-order branch), and
this handler drives it headlessly the same way the web app does on upload:

    create space -> place video -> stitch -> frames -> views
        -> 00b_gen_da3 (depth+poses)  [GPU]
        -> 01_propose (FastSAM)        [GPU]
        -> 02_embed (DINOv2)           [GPU]
        -> 02b_match_views
        -> 03_backproject (object point cloud)
        -> 03b_merge_groups
        -> 04_index
    -> zip {da3 npz, cameras.json, pointcloud, index} -> upload to bucket -> URL

Input JSON:
    {"input": {"video_url": "...", "slug": "scan1", "capture_type": "insta360",
               "every_n": 100}}
Output JSON:
    {"slug": "...", "result_url": "https://.../scan1.zip",
     "stages": {"00b_gen_da3": "ok", ...}, "num_views": N, "num_points": M}
  or {"error": "...", "failed_stage": "...", "stderr": "..."}

NOTE: this is v1 — the exact per-space layout for the pre-DA3 steps is validated
against a local run of the platform before trusting it. Every stage is a
subprocess with captured stderr, so the first failing stage names itself.
"""
import json, os, glob, shutil, subprocess, sys, tempfile, traceback, urllib.request, uuid
from pathlib import Path

import runpod

# ---- where the CEO's platform lives in the image (cloned by the Dockerfile) ----
PLATFORM = os.environ.get("INFRASCAN_PLATFORM_DIR", "/app/infrascan-platform")
# Per-run working data lives on the (optionally mounted) volume, else /workspace.
WORKROOT = os.environ.get("INFRASCAN_WORKROOT", "/workspace/runs")

# The proven entrypoint is `python -m pipeline.runner --slug <slug>`, which runs
# 00b_gen_da3 -> 01_propose -> 02_embed -> 02b_match_views -> 03_backproject
# -> 03b_merge_groups -> 04_index -> gen_topdown -> downsample_ply (through the
# point cloud). We call that directly rather than re-implementing the stage list.
PRE_STAGES = ["_00_stitch_insv", "00_video_to_img", "00a_sample_views"]


def _run(cmd, cwd, env, stage):
    """Run one stage as a subprocess; raise with captured stderr on failure."""
    print(f"[stage {stage}] $ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    tail = (r.stderr or "")[-3000:]
    print((r.stdout or "")[-2000:], flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"stage {stage} exited {r.returncode}\n{tail}")
    return tail


def handler(job):
    inp = job.get("input", {}) or {}
    video_url = inp.get("video_url")
    if not video_url:
        return {"error": "provide input.video_url"}
    slug = (inp.get("slug") or f"scan-{uuid.uuid4().hex[:8]}").lower()
    capture_type = inp.get("capture_type", "insta360")
    every_n = int(inp.get("every_n", 100))

    # Isolate this run's data/DB under the work root; point the platform config at it.
    run_root = Path(WORKROOT) / slug
    run_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["INFRASCAN_DB_PATH"] = str(run_root / "infrascan.db")
    env["INFRASCAN_DATA_ROOT"] = str(run_root / "data")
    env["INFRASCAN_OUT_ROOT"] = str(run_root / "out")
    env["PYTHONPATH"] = PLATFORM + os.pathsep + env.get("PYTHONPATH", "")
    py = sys.executable
    stages_status = {}

    try:
        # 1) bootstrap the platform DB + a space row (mirrors app.spaces.create_space)
        sys.path.insert(0, PLATFORM)
        for k in ("INFRASCAN_DB_PATH", "INFRASCAN_DATA_ROOT", "INFRASCAN_OUT_ROOT"):
            os.environ[k] = env[k]
        from app import config as cfg
        from app import spaces as space_repo
        from app.db import init as db_init, get_conn
        from app.auth import create_user
        cfg.ensure_dirs(); db_init()
        if not space_repo.by_slug(slug):
            # spaces.owner_id is a FK to users(id) (TEXT) — need a real user first.
            row = get_conn().execute("SELECT id FROM users LIMIT 1").fetchone()
            owner_id = row[0] if row else create_user(
                email=f"{slug}@worker.local", name="worker", password="worker-bootstrap-pw", role="admin")
            space_repo.create_space(slug=slug, title=slug, owner_id=owner_id, status="processing")
        data_dir = Path(space_repo.data_dir(slug))
        (data_dir / "uploads").mkdir(parents=True, exist_ok=True)

        # 2) download the video
        ext = os.path.splitext(video_url.split("?")[0])[1].lower() or ".mp4"
        vid = data_dir / "uploads" / f"input{ext}"
        print(f"[dl] {video_url} -> {vid}", flush=True)
        urllib.request.urlretrieve(video_url, str(vid))

        # 3) pre-DA3: stitch -> frames -> views  (writes into the space's data dir)
        P = Path(PLATFORM) / "pipeline"
        eq = data_dir / "equirect.mp4"
        frames = data_dir / "frames"; views = data_dir / "views"
        _run([py, str(P / "_00_stitch_insv.py"), "--input", str(vid), "--output", str(eq)],
             PLATFORM, env, "_00_stitch_insv"); stages_status["_00_stitch_insv"] = "ok"
        _run([py, str(P / "00_video_to_img.py"), "--video", str(eq),
              "--output_dir", str(frames), "--every_n", str(every_n)],
             PLATFORM, env, "00_video_to_img"); stages_status["00_video_to_img"] = "ok"
        _run([py, str(P / "00a_sample_views.py"), "--input_dir", str(frames),
              "--output_dir", str(views)], PLATFORM, env, "00a_sample_views")
        stages_status["00a_sample_views"] = "ok"

        # 4) the proven entrypoint: runs 00b -> ... -> downsample_ply (through pointcloud)
        _run([py, "-m", "pipeline.runner", "--slug", slug], PLATFORM, env, "pipeline.runner")
        stages_status["pipeline.runner"] = "ok"

        # 5) collect outputs, zip, upload
        n_views = len(glob.glob(str(views / "*.jpg")))
        pcds = glob.glob(str(data_dir / "**" / "*.ply"), recursive=True)
        archive = Path(WORKROOT) / f"{slug}.zip"
        shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=str(data_dir))

        import storage  # Stage B (sits next to this file)
        result_url = storage.upload(archive, key=f"results/{slug}.zip")

        return {
            "slug": slug, "result_url": result_url,
            "num_views": n_views, "num_pointclouds": len(pcds),
            "pointclouds": [os.path.basename(p) for p in pcds],
            "stages": stages_status,
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "failed_stage": next((s for s in (PRE_STAGES + ["pipeline.runner"])
                                  if s not in stages_status), "?"),
            "stages_ok": stages_status,
            "trace": traceback.format_exc()[-1500:],
        }


runpod.serverless.start({"handler": handler})

# build trigger 65dfca5
