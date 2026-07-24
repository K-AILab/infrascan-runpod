"""RunPod TRAINING endpoint — depth-supervised splatfacto (abai's recipe).

Runs AFTER the pipeline job. Input: {"slug": "<space>"}. It pulls the scan from
S3 (scans/<slug>/: frames, views, cameras.json, intrinsics.json, pointcloud.ply),
then replicates abai's exact chain:

  make_transforms.py       cameras.json -> nerfstudio transforms.json (504)
  build_hires_dataset.py   re-render perspective crops at 1024 (reuse poses)
  reproject_scanner_depth  OUR pointcloud.ply -> per-view depth  (depth_scanner/)
  generate_person_masks    YOLO person masks -> masks/  (+ wire mask_path)
  [5-iter dry run]         measure THIS scene's DEPTH_SCALE (dataparser scale)
  ns_depthsup.py           splatfacto 30k + EdgeAwareLogL1 depth loss (+masks)
  ns_export_gs.py          checkpoint -> splat.ply
  ply2ksplat.mjs           splat.ply -> splat.ksplat

Then uploads scans/<slug>/splat.ksplat to S3. The home-server worker flips
has3d=true so the tri-viewer's 3D tab un-greys and streams the splat from S3.

Output: {"slug", "has3d": true, "splat_key", "depth_scale", "iters"} or {"error"}.
"""
import os, sys, json, glob, shutil, subprocess, traceback
from pathlib import Path

import runpod
import boto3

BUCKET = os.environ["S3_BUCKET"]
s3 = boto3.client(
    "s3",
    region_name=os.environ.get("S3_REGION", "us-east-1"),
    aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
    aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
)

VENDOR = Path("/app/train/vendor")
WORKROOT = Path(os.environ.get("TRAIN_WORKROOT", "/runpod-volume/train"))
PY = sys.executable

# abai's proven splatfacto recipe (shinhan-pz000-hires-30k / factory13 depthsup)
MODEL_ARGS = [
    "--pipeline.model.num-downscales", "0",
    "--pipeline.model.camera-optimizer.mode", "SO3xR3",
    "--pipeline.model.rasterize-mode", "antialiased",
    "--pipeline.model.densify-grad-thresh", "0.0004",
    "--pipeline.model.cull-alpha-thresh", "0.005",
    "--pipeline.model.cull-scale-thresh", "0.5",
    "--pipeline.model.densify-size-thresh", "0.01",
    "--pipeline.model.split-screen-size", "0.05",
    "--pipeline.model.cull-screen-size", "0.15",
    "--pipeline.model.stop-screen-size-at", "4000",
    "--pipeline.model.stop-split-at", "15000",
    "--pipeline.model.warmup-length", "500",
    "--pipeline.model.refine-every", "100",
    "--pipeline.model.reset-alpha-every", "30",
    "--pipeline.model.use-absgrad", "True",
    "--pipeline.model.use-scale-regularization", "True",
    "--pipeline.model.use-bilateral-grid", "True",
    "--pipeline.model.background-color", "random",
    "--pipeline.model.sh-degree", "3",
    "--pipeline.model.sh-degree-interval", "1000",
    "--vis", "tensorboard",
    "--viewer.quit-on-train-completion", "True",
]


def data_args(data_dir: str):
    return ["nerfstudio-data", "--data", data_dir,
            "--orientation-method", "up", "--center-method", "poses",
            "--auto-scale-poses", "True", "--train-split-fraction", "0.9"]


def _sh(cmd, env=None, cwd=None):
    print("[train] $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env, cwd=cwd)


def _dl_scan(slug: str, dst: Path):
    """Download scans/<slug>/ from S3 into a 3d-data-style capture dir."""
    dst.mkdir(parents=True, exist_ok=True)
    prefix = f"scans/{slug}/"
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            rel = o["Key"][len(prefix):]
            if not rel:
                continue
            local = dst / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, o["Key"], str(local))
            n += 1
    if n == 0:
        raise RuntimeError(f"no objects under s3://{BUCKET}/{prefix}")
    print(f"[train] downloaded {n} files -> {dst}", flush=True)


def _add_mask_paths(ds: Path):
    """nerfstudio uses per-frame mask_path; generate_person_masks writes masks/<stem>.png
    but doesn't edit transforms.json. Wire each frame to its mask if present."""
    tj_path = ds / "transforms.json"
    tj = json.loads(tj_path.read_text())
    masks = ds / "masks"
    wired = 0
    for fr in tj.get("frames", []):
        stem = Path(fr["file_path"]).stem
        m = masks / f"{stem}.png"
        if m.exists():
            fr["mask_path"] = f"masks/{stem}.png"
            wired += 1
    tj_path.write_text(json.dumps(tj))
    print(f"[train] wired {wired} mask_path entries", flush=True)


def _measure_depth_scale(ds: Path, probe_dir: Path) -> float:
    """5-iter dry run → read the dataparser's auto-scale-poses factor (per-scene).
    Reusing another scene's scale is the exact bug abai hit (PSNR plateau)."""
    _sh(["ns-train", "splatfacto", "--output-dir", str(probe_dir),
         "--experiment-name", "scaleprobe", "--max-num-iterations", "5",
         "--vis", "tensorboard", "--viewer.quit-on-train-completion", "True"]
        + data_args(str(ds)))
    dpt = sorted(probe_dir.glob("**/dataparser_transforms.json"))
    if not dpt:
        raise RuntimeError("dataparser_transforms.json not found after dry run")
    scale = float(json.loads(dpt[-1].read_text())["scale"])
    print(f"[train] measured DEPTH_SCALE = {scale}", flush=True)
    return scale


def handler(job):
    inp = job.get("input", {}) or {}
    slug = inp.get("slug")
    if not slug:
        return {"error": "provide input.slug"}
    iters = int(inp.get("iters", 30000))
    use_masks = bool(inp.get("masks", True))

    root = WORKROOT / slug
    if root.exists():
        shutil.rmtree(root)
    src = root / "src"; ds504 = root / "ds504"; ds = root / "ds1024"
    out = root / "out"; export = root / "export"

    try:
        _dl_scan(slug, src)

        # 1) cameras.json -> transforms.json (504)  2) re-render at 1024
        _sh([PY, VENDOR / "make_transforms.py", "--src", src, "--out", ds504])
        _sh([PY, VENDOR / "build_hires_dataset.py", "--src", src,
             "--ref-data", ds504, "--out", ds, "--res", "1024"])

        # 3) reproject OUR pointcloud -> per-view depth
        _sh([PY, VENDOR / "reproject_scanner_depth.py", "--data", ds,
             "--pointcloud", src / "pointcloud.ply", "--out", ds / "depth_scanner"])

        # 4) YOLO person masks (+ wire mask_path)
        if use_masks:
            _sh([PY, VENDOR / "generate_person_masks.py", "--data", ds])
            _add_mask_paths(ds)

        # 5) per-scene DEPTH_SCALE (dry run)
        scale = _measure_depth_scale(ds, root / "scaleprobe")

        # 6) depth-supervised splatfacto (abai's recipe)
        env = os.environ.copy()
        env["DEPTH_DIR"] = str(ds / "depth_scanner")
        env["DEPTH_W"] = "0.1"
        env["DEPTH_START_ITER"] = "500"
        env["DEPTH_SCALE"] = str(scale)
        _sh([PY, VENDOR / "ns_depthsup.py", "splatfacto",
             "--output-dir", out, "--experiment-name", slug,
             "--max-num-iterations", str(iters), "--steps-per-save", "2000"]
            + MODEL_ARGS + data_args(str(ds)), env=env)

        # 7) export checkpoint -> splat.ply
        cfg = sorted(out.glob("**/config.yml"))
        if not cfg:
            raise RuntimeError("no config.yml after training")
        _sh([PY, VENDOR / "ns_export_gs.py", "--load-config", cfg[-1],
             "--output-dir", export])

        # 8) splat.ply -> splat.ksplat
        ply = export / "splat.ply"
        ksplat = root / "splat.ksplat"
        _sh(["node", VENDOR / "node" / "ply2ksplat.mjs", ply, ksplat, "3"])

        # 9) upload to S3 so the tri-viewer's 3D tab can stream it
        key = f"scans/{slug}/splat.ksplat"
        s3.upload_file(str(ksplat), BUCKET, key)
        print(f"[train] uploaded {key}", flush=True)

        return {"slug": slug, "has3d": True, "splat_key": key,
                "depth_scale": scale, "iters": iters}
    except subprocess.CalledProcessError as e:
        return {"error": f"stage failed rc={e.returncode}: {e.cmd[:3]}",
                "trace": traceback.format_exc()[-1500:]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-1500:]}
    finally:
        shutil.rmtree(root, ignore_errors=True)


runpod.serverless.start({"handler": handler})
