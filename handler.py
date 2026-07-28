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
PRE_STAGES = ["_00_stitch_insv", "00_video_to_img", "00a_sample_views", "00b_da3_streaming"]


def _run(cmd, cwd, env, stage):
    """Run one stage as a subprocess; raise with captured stderr on failure."""
    print(f"[stage {stage}] $ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    tail = (r.stderr or "")[-3000:]
    print((r.stdout or "")[-2000:], flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"stage {stage} exited {r.returncode}\n{tail}")
    return tail


# da3_streaming loads three weight files by RELATIVE path (./weights/...) with
# cwd=da3_streaming/. They are ~6.6 GB total and gitignored, so they are NOT in
# the image. Cache them on the persistent network volume and symlink them in, so
# the giant DA3 checkpoint downloads only ONCE across all cold starts.
DA3_WEIGHTS = {
    "config.json":
        "https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/main/config.json",
    "model.safetensors":
        "https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/main/model.safetensors",
    "dino_salad.ckpt":
        "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",
}


def _ensure_da3_weights():
    da3_dir = Path(PLATFORM) / "pipeline" / "da3_streaming"
    wdir = da3_dir / "weights"
    cache = Path(os.environ.get("DA3_WEIGHTS_DIR",
                                str(Path(WORKROOT).parent / "da3_weights")))
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in DA3_WEIGHTS.items():
        dst = cache / name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        print(f"[weights] downloading {name} (first cold start only) ...", flush=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, str(tmp))
        tmp.replace(dst)
        print(f"[weights] {name} -> {dst} ({dst.stat().st_size/1e6:.1f} MB)", flush=True)
    wdir.mkdir(parents=True, exist_ok=True)
    for name in DA3_WEIGHTS:
        link = wdir / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(cache / name)
    print(f"[weights] linked {list(DA3_WEIGHTS)} into {wdir}", flush=True)


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
    # WORKROOT is a persistent volume keyed by slug, so a re-scan would otherwise inherit the
    # previous run's DA3 chunk files (_da3_streaming/pcd/*_pcd.ply). merge_ply_files globs those,
    # mixing stale + fresh chunks into a corrupt pointcloud.ply whose header count != its body
    # (crashes training / silently drops points). Start every job from a clean dir. The DA3
    # weights live OUTSIDE run_root (WORKROOT.parent/da3_weights), so they are not touched.
    shutil.rmtree(run_root, ignore_errors=True)
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
        # 3 pitches (0, +30, -30) like the original pipeline: DA3 poses all of them
        # (richer point cloud from up/down coverage) and the perspective viewer can
        # look up/down. Gaussian TRAINING stays single-pitch — build_hires_dataset.py
        # (train branch) filters to pz000 via its --pz 0 default, so only the eye-level
        # crops feed splatfacto. Encoded in filenames as pz000/pz030/pz330.
        _run([py, str(P / "00a_sample_views.py"), "--input_dir", str(frames),
              "--output_dir", str(views), "--pitches", "-30", "0", "30"],
             PLATFORM, env, "00a_sample_views")
        stages_status["00a_sample_views"] = "ok"

        # 3b) DA3 streaming: estimate camera POSES (+depth) from the views -> cameras.json.
        #     A fresh video has no poses; the runner's 00b_gen_da3 requires cameras.json,
        #     so this must run first. Ensure the ~6.6GB DA3+SALAD weights are on the
        #     volume + linked in before running it.
        _ensure_da3_weights()
        _run([py, str(P / "00b_da3_streaming.py"), "--space", slug],
             PLATFORM, env, "00b_da3_streaming"); stages_status["00b_da3_streaming"] = "ok"

        # 4) the proven entrypoint: runs 00b -> ... -> downsample_ply (through pointcloud)
        _run([py, "-m", "pipeline.runner", "--slug", slug], PLATFORM, env, "pipeline.runner")
        stages_status["pipeline.runner"] = "ok"

        # 5) upload the UNPACKED dataset straight to S3 under scans/<slug>/ so the
        #    home server stores NOTHING — the tri-viewer streams each file from S3.
        #    Layout the viewer expects:
        #      scans/<slug>/frames/*.jpg          (panoramas)
        #      scans/<slug>/views/*.jpg           (perspective; served as panos/ too)
        #      scans/<slug>/cameras.json, intrinsics.json
        #      scans/<slug>/depth/frame_<i>.npz   (per-view depth)
        #      scans/<slug>/pointcloud.ply        (for Phase-2 splat training)
        import storage
        s3c = storage._client()
        bucket = os.environ["S3_BUCKET"]
        prefix = f"scans/{slug}"

        def _put(local: Path, key: str):
            s3c.upload_file(str(local), bucket, key)

        nfiles = 0
        for sub in ("frames", "views"):
            for p in sorted((data_dir / sub).glob("*")):
                if p.is_file():
                    _put(p, f"{prefix}/{sub}/{p.name}"); nfiles += 1
        for f in ("cameras.json", "intrinsics.json", "pointcloud.ply"):
            if (data_dir / f).exists():
                _put(data_dir / f, f"{prefix}/{f}"); nfiles += 1
        ro = data_dir / "_da3_streaming" / "results_output"
        if ro.is_dir():
            for npz in sorted(ro.glob("frame_*.npz")):
                _put(npz, f"{prefix}/depth/{npz.name}"); nfiles += 1

        n_views = len(glob.glob(str(views / "*.jpg")))
        n_panos = len(glob.glob(str(frames / "*.jpg")))
        print(f"[s3] uploaded {nfiles} files to s3://{bucket}/{prefix}/", flush=True)

        return {
            "slug": slug, "scan_prefix": prefix + "/",
            "num_views": n_views, "num_panos": n_panos,
            "num_files": nfiles, "stages": stages_status,
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
