"""Scene-graph stage (Approach A: OWLv2-on-splat) run INLINE after training.

This is the standalone scenegraph endpoint's pipeline, adapted to run as a
continuation of the TRAIN job: the train handler already has the trained splat
and the whole scan on local disk, so this reads those LOCAL files (no S3
download) and only uploads the result. It runs the vendored
`scene-graph-pipeline` Approach A end to end across the two venvs, then writes:

  scans/<slug>/scene_graph.json                  # viewer-facing (SPLAT frame)
  scans/<slug>/scenegraph/<slug>_final.json      # full Approach-A output
  scans/<slug>/scenegraph/<slug>_transform.json  # splat<->pointcloud transform

Invoked by train/handler.py as a subprocess under venv-main:
  /opt/venv-main/bin/python run_scenegraph.py --slug S --splat .../splat.ply \
     --pointcloud .../pointcloud.ply --cameras .../cameras.json \
     --intrinsics .../intrinsics.json --views .../views --workdir .../sg \
     --bucket <bucket>

Exits 0 on success (prints an [scenegraph] summary), non-zero on failure. The
train handler treats a non-zero exit as non-fatal (the splat is already in S3).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import boto3

# This file lives at /app/train/scenegraph/run_scenegraph.py inside the image.
APP = Path(__file__).resolve().parent
REPO = APP / "vendor" / "scene-graph-pipeline"
SPLAT_ANALYZER = REPO / "external" / "splat_analyzer"

# Two isolated interpreters (the pipeline's two envs must not be merged).
PY_SPLAT = os.environ.get("PY_SPLAT", "/opt/venv-splat/bin/python")
PY_MAIN = os.environ.get("PY_MAIN", "/opt/venv-main/bin/python")

# Open-vocabulary label set for OWLv2 (README's tuned list). Overridable per run.
DEFAULT_PROMPT = (
    "chair, table, desk, workbench, shelf, storage rack, cabinet, cardboard box, "
    "pallet, cart, machine, trash bin, whiteboard, bench, ladder, light, window, "
    "door, fire extinguisher, printer, plant, monitor"
)


def _sh(cmd, cwd=None, env=None):
    """Run one stage as a subprocess; raise with captured stderr on failure."""
    print("[scenegraph] $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                       capture_output=True, text=True)
    if r.stdout:
        print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"stage {Path(str(cmd[1])).name} exited {r.returncode}\n"
                           + (r.stderr or "")[-3000:])
    return r.stdout


def _link_or_copy(src: Path, dst: Path) -> None:
    """Stage a local input under the pipeline's data_root. Symlink to avoid
    duplicating large files (pointcloud/views); fall back to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _dump_debug(work: Path, slug: str, bucket: str) -> None:
    """Best-effort: upload the pipeline's intermediate box files + a counts summary
    to scans/<slug>/scenegraph/debug/ so we can see WHERE boxes drop to zero
    (detection vs. gaussian-count filter vs. refit)."""
    if not bucket:
        return
    try:
        s3 = boto3.client(
            "s3", region_name=os.environ.get("S3_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY"))
    except Exception:
        return

    def count(p: Path):
        try:
            d = json.loads(p.read_text())
            if isinstance(d, list):
                return len(d)
            if isinstance(d, dict):
                for k in ("boxes", "objects", "detections", "interactions", "nodes"):
                    if isinstance(d.get(k), list):
                        return len(d[k])
                return {k: (len(v) if isinstance(v, list) else "?") for k, v in d.items()}
        except Exception as e:
            return f"unreadable: {e}"

    files = {
        "interactions.json": work / "owl" / "interactions.json",
        "scene_boxes.json": work / "scene_boxes.json",
        "scene_boxes_filtered.json": work / "scene_boxes_filtered.json",
        "scene_boxes_refit.json": work / "scene_boxes_refit.json",
    }
    counts = {}
    for name, p in files.items():
        if p.exists():
            counts[name] = count(p)
            try:
                s3.upload_file(str(p), bucket, f"scans/{slug}/scenegraph/debug/{name}")
            except Exception:
                pass
        else:
            counts[name] = "MISSING"
    try:
        s3.put_object(Bucket=bucket, Key=f"scans/{slug}/scenegraph/debug/counts.json",
                      Body=json.dumps(counts, indent=2).encode())
    except Exception:
        pass
    print("[scenegraph] debug counts: " + json.dumps(counts), flush=True)


def _bbox_diag(ply_path: Path) -> float:
    """Diagonal of a point cloud / splat's xyz bounding box (points only, cheap).
    open3d reads x,y,z and ignores the splat's SH/opacity columns."""
    import open3d as o3d
    import numpy as np
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise RuntimeError(f"no points read from {ply_path}")
    return float(np.linalg.norm(pts.max(0) - pts.min(0)))


def _register_space(slug: str, data_root: Path) -> None:
    """Write REPO/spaces.json with just this slug (fresh container per job)."""
    cfg = {"spaces": {slug: {
        "title": slug,
        "data_root": str(data_root.relative_to(REPO)),
        "out_dir": f"out/fastsam_{slug}",
        "y_up": True,
        "n_views": 0,
        "n_scanpoints": 0,
    }}}
    (REPO / "spaces.json").write_text(json.dumps(cfg, indent=2))


def run(args) -> dict:
    # Point the OWLv2/CLIP loads at the BAKED cache (the Dockerfile prefetched
    # models here). Must override the train handler's HF_HOME=/train/hf, which we
    # inherit — child processes (PY_SPLAT/PY_MAIN) pick these up at spawn time.
    os.environ["HF_HOME"] = os.environ.get("SG_HF_HOME", "/opt/sg-models/hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HOME"]
    os.environ["TORCH_HOME"] = os.environ.get("SG_TORCH_HOME", "/opt/sg-models/torch")
    # (models are baked at this cache; transformers/open_clip use it without a
    #  network call. Not forcing HF_HUB_OFFLINE so a cache miss can still fetch.)

    slug = args.slug
    work = Path(args.workdir).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    data_root = REPO / "data" / slug          # _paths resolves data_root under REPO

    try:
        # ---- 1. stage LOCAL inputs (no S3 download) ---------------------------
        views_src = Path(args.views)
        if not views_src.is_dir() or not any(views_src.iterdir()):
            return {"error": f"no views under {views_src} — Approach A needs real "
                             "per-view photos for the CLIP relabel step"}
        splat_ply = Path(args.splat).resolve()
        pointcloud_ply = data_root / "pointcloud.ply"
        _link_or_copy(Path(args.pointcloud).resolve(), pointcloud_ply)
        _link_or_copy(Path(args.cameras).resolve(), data_root / "cameras.json")
        _link_or_copy(Path(args.intrinsics).resolve(), data_root / "intrinsics.json")
        _link_or_copy(views_src.resolve(), data_root / "views")
        _register_space(slug, data_root)
        print(f"[scenegraph] staged {slug} from local train dir", flush=True)

        boxes = work / "scene_boxes.json"
        boxes_filt = work / "scene_boxes_filtered.json"
        boxes_refit = work / "scene_boxes_refit.json"
        transform = work / f"{slug}_transform.json"
        geo = work / f"{slug}_geo.json"
        pc_json = work / f"{slug}_pointcloud.json"
        relabel_geo = work / f"{slug}_relabel_geo.json"
        final = work / f"{slug}_final.json"
        scene_graph = work / "scene_graph.json"

        # ---- 2. detect on the splat (GPU) ------------------------------------
        _sh([PY_SPLAT, "run_local.py", "--ply", splat_ply, "--prompt", args.prompt,
             "--quality", args.quality, "--n_positions", args.n_positions,
             "--score_threshold", args.score_threshold,
             "--min_vote_frac", args.min_vote_frac,
             "--min_peak_score", args.min_peak_score,
             "--max_per_label", args.max_per_label,
             "--max_object_diag", args.max_object_diag, "--job_dir", work / "owl"],
            cwd=SPLAT_ANALYZER)
        _sh([PY_SPLAT, "rotate_and_export.py",
             "--interactions", work / "owl" / "interactions.json",
             "--out", boxes, "--yaw-deg", 0],
            cwd=SPLAT_ANALYZER)

        # ---- 3. refine box sizes against the gaussians (CPU) -----------------
        _sh([PY_MAIN, REPO / "pipeline9" / "closed_surface_flux.py",
             "--ply", splat_ply, "--boxes", boxes,
             "--min-gaussians", args.min_gaussians, "--filtered-out", boxes_filt],
            cwd=REPO)
        _sh([PY_MAIN, REPO / "pipeline9" / "refit_box_extent.py",
             "--ply", splat_ply, "--boxes", boxes_filt, "--out", boxes_refit],
            cwd=REPO)

        # ---- 4. align splat<->pointcloud, transform boxes onto the cloud -----
        rough = _bbox_diag(pointcloud_ply) / max(_bbox_diag(splat_ply), 1e-9)
        print(f"[scenegraph] rough scale-to-meters guess = {rough:.4f}", flush=True)
        _sh([PY_MAIN, REPO / "pipeline9" / "align_splat_to_pointcloud.py",
             "--splat-ply", splat_ply, "--pointcloud-ply", pointcloud_ply,
             "--scale-to-meters-guess", rough, "--out", transform], cwd=REPO)
        scale = float(json.loads(transform.read_text())["true_scale_to_meters"])
        print(f"[scenegraph] true scale-to-meters = {scale}", flush=True)

        _sh([PY_MAIN, REPO / "pipeline9" / "export_scene_graph_for_point_viewer.py",
             "--ply", splat_ply, "--yaw-deg", 0, "--scale-to-meters", scale,
             "--boxes", boxes_refit, "--space", slug, "--out", geo], cwd=REPO)
        _sh([PY_MAIN, REPO / "pipeline9" / "apply_scenegraph_to_pointcloud.py",
             "--scene-graph", geo, "--transform", transform,
             "--source-scale-to-meters", scale, "--space", slug,
             "--pointcloud-ply", pointcloud_ply, "--out", pc_json], cwd=REPO)

        # ---- 5. CLIP relabel from the real photos (GPU) ----------------------
        _sh([PY_MAIN, REPO / "pipeline9" / "relabel_with_clip.py",
             "--scene-graph", pc_json, "--pointcloud-ply", pointcloud_ply,
             "--space", slug, "--geo-json-out", relabel_geo, "--out", final],
            cwd=REPO)

        # ---- 6. build the viewer scene graph (splat-native boxes + labels) ---
        _sh([PY_MAIN, APP / "export_viewer_scenegraph.py",
             "--refit-boxes", boxes_refit, "--final", final,
             "--slug", slug, "--out", scene_graph], cwd=REPO)

        # ---- 7. upload -------------------------------------------------------
        sg = json.loads(scene_graph.read_text())
        s3 = boto3.client(
            "s3",
            region_name=os.environ.get("S3_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
        )
        bucket = args.bucket or os.environ["S3_BUCKET"]
        key = f"scans/{slug}/scene_graph.json"
        s3.upload_file(str(scene_graph), bucket, key)
        s3.upload_file(str(final), bucket, f"scans/{slug}/scenegraph/{slug}_final.json")
        s3.upload_file(str(transform), bucket,
                       f"scans/{slug}/scenegraph/{slug}_transform.json")
        print(f"[scenegraph] uploaded {key} "
              f"({len(sg['nodes'])} nodes, {len(sg['edges'])} edges)", flush=True)

        return {"slug": slug, "scene_graph_key": key,
                "n_nodes": len(sg["nodes"]), "n_edges": len(sg["edges"]),
                "labels": sg["labels"], "scale_to_meters": scale}

    finally:
        # dump intermediate box files + counts to S3 BEFORE cleaning scratch, so
        # we can see where boxes vanish (only when the run didn't fully succeed —
        # a good run's counts are implicit in the final scene_graph.json).
        try:
            _dump_debug(work, slug, args.bucket or os.environ.get("S3_BUCKET"))
        except Exception:
            pass
        # clean the run's scratch AND the per-slug staging we wrote under REPO
        shutil.rmtree(work, ignore_errors=True)
        # data_root uses symlinks into the train dir — unlink them, don't follow
        for name in ("pointcloud.ply", "cameras.json", "intrinsics.json", "views"):
            p = data_root / name
            if p.is_symlink() or p.exists():
                (p.unlink() if p.is_symlink() or p.is_file()
                 else shutil.rmtree(p, ignore_errors=True))
        shutil.rmtree(data_root, ignore_errors=True)
        (REPO / "spaces.json").unlink(missing_ok=True)
        shutil.rmtree(REPO / "out", ignore_errors=True)


def _dump_error_s3(args, text: str) -> None:
    """Best-effort: write the failure to scans/<slug>/scenegraph_error.txt so it
    can be inspected without RunPod run-logs (which have no public API)."""
    try:
        s3 = boto3.client(
            "s3", region_name=os.environ.get("S3_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY"))
        bucket = args.bucket or os.environ["S3_BUCKET"]
        s3.put_object(Bucket=bucket, Key=f"scans/{args.slug}/scenegraph_error.txt",
                      Body=text.encode("utf-8", "replace"))
        print(f"[scenegraph] wrote error to scans/{args.slug}/scenegraph_error.txt",
              flush=True)
    except Exception as e:
        print(f"[scenegraph] could not upload error dump: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--splat", required=True, help="trained splat.ply (local)")
    ap.add_argument("--pointcloud", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--views", required=True, help="dir of per-view photos")
    ap.add_argument("--workdir", required=True, help="scratch dir for this run")
    ap.add_argument("--bucket", default=os.environ.get("S3_BUCKET"))
    # tuning (defaults match the pipeline's validated run)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--quality", default="high")
    ap.add_argument("--n_positions", default="64")
    ap.add_argument("--score_threshold", default="0.10")
    ap.add_argument("--min_vote_frac", default="0.026")
    ap.add_argument("--min_peak_score", default="0.35")
    ap.add_argument("--max_per_label", default="80")
    ap.add_argument("--max_object_diag", default="0.5")
    ap.add_argument("--min_gaussians", default="50")
    args = ap.parse_args()

    try:
        result = run(args)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[scenegraph] FAILED: {type(e).__name__}: {e}", flush=True)
        print(tb[-2500:], flush=True)
        _dump_error_s3(args, f"{type(e).__name__}: {e}\n\n{tb}")
        sys.exit(1)

    if result.get("error"):
        print(f"[scenegraph] ERROR: {result['error']}", flush=True)
        _dump_error_s3(args, str(result["error"]))
        sys.exit(1)
    print("[scenegraph] RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
