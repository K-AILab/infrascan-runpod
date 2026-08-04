"""Scene-graph stage driver — runs the UPSTREAM pipeline, unmodified.

This is OUR thin glue around `k-ailab/scene-graph-pipeline` (vendored at
`vendor/scene-graph-pipeline`, commit e4d9409). It does NOT reimplement any of
the pipeline's logic: it stages our S3-sourced scan into the layout the
pipeline expects, registers the space, then runs the creator's own
`pipeline9/run_space_pipeline.py` end to end (detection → flux → refit → ICP
align → CLIP → support prior → top-down table detection → grounding → SAM mask
refit → 2nd CLIP → render-verify → harmonise → export). Finally it converts the
pipeline's own output into the `scene_graph.json` schema our viewer already
consumes, and uploads it.

Invoked by train/handler.py as a subprocess under the combined venv:
  $PY_SG run_scenegraph.py --slug S --splat .../splat.ply \
     --pointcloud .../pointcloud.ply --cameras .../cameras.json \
     --intrinsics .../intrinsics.json --views .../views --workdir .../sg \
     --bucket <bucket>

Deviations from a stock upstream checkout are documented in NOTES.md. The two
that matter here:
  * run_space_pipeline.py's per-space SPACES table is fed at runtime via the
    SG_SPACES_JSON shim (the single edit made to that file) instead of being
    hand-edited — every constant we pass is either derived from the data or a
    documented default.
  * yaw_deg defaults to 0 (detection on the splat as captured). Upstream
    detects on a pre-derotated splat using a per-space yaw it measures by hand;
    there is no committed auto-yaw tool, so we do not invent one. A measured
    yaw can be supplied with --yaw-deg once known.

Exits 0 on success (prints `[scenegraph] RESULT {...}`), non-zero on failure.
The train handler treats a non-zero exit as non-fatal (the splat is already in
S3).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path

import boto3

# This file lives at /app/train/scenegraph/run_scenegraph.py inside the image.
APP = Path(__file__).resolve().parent
REPO = APP / "vendor" / "scene-graph-pipeline"          # vendored upstream root
RUN_SPACE = REPO / "pipeline9" / "run_space_pipeline.py"
OUT = REPO / "pipeline9" / "out"                         # where the pipeline writes

# One combined interpreter, matching upstream's run_space_pipeline.py (which
# drives every stage — including the OWLv2 detector — with `sys.executable`).
PY = os.environ.get("PY_SG", sys.executable)

_LOG = []   # accumulates the pipeline's stdout/stderr for the S3 debug log


def _link_or_copy(src: Path, dst: Path) -> None:
    """Stage a local input under the pipeline's data_root. Symlink to avoid
    duplicating large files (splat/pointcloud/views); fall back to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)


def _register_space(slug: str, data_root: Path, n_views: int) -> None:
    """Write REPO/spaces.json with just this slug so the pipeline's `_paths`
    accessor can resolve data/<slug>/{views,cameras.json,intrinsics.json,
    pointcloud.ply} for the CLIP-relabel stage. Fresh container per job, so a
    single-entry file is fine."""
    cfg = {"spaces": {slug: {
        "title": slug,
        "data_root": str(data_root.relative_to(REPO)),
        "out_dir": f"out/fastsam_{slug}",
        "y_up": False,          # our splats are Z-up; the pipeline's scale guess
                                # only seeds ICP, which recovers the true scale.
        "n_views": n_views,
        "n_scanpoints": 0,
    }}}
    (REPO / "spaces.json").write_text(json.dumps(cfg, indent=2))


def _write_space_config(slug: str, splat: Path, args) -> Path:
    """Build the run_space_pipeline SPACES entry for this slug and write it where
    the SG_SPACES_JSON shim will pick it up. Every value is either derived from
    the data or a documented default (see NOTES.md)."""
    entry = {slug: {
        # yaw=0 -> detect on the splat as captured; derot == splat (no pre-rotate).
        "yaw_deg": float(args.yaw_deg),
        "splat": str(splat.relative_to(REPO)),
        "derot": str(splat.relative_to(REPO)),
        # None -> run_space_pipeline uses the ICP-measured true scale.
        "viewer_scale": None,
        # our .ksplat is built from the original (un-derotated) splat.
        "tri_frame": "original",
        "tri_scene": slug,
        # --space labels for the export/apply stages; slug is registered above.
        "pc_space": slug,
        "splat_space": slug,
        "surface_label": args.surface_label,
        "max_long_m": float(args.max_long_m),
        "drop_labels": args.drop_labels,
    }}
    p = REPO / f"_sg_spaces_{slug}.json"
    p.write_text(json.dumps(entry, indent=2))
    return p


def _pipeline_env(spaces_json: Path) -> dict:
    env = os.environ.copy()
    env["SG_SPACES_JSON"] = str(spaces_json)
    # Point OWLv2/CLIP/SAM at the baked cache; override the train handler's
    # HF_HOME (=/train/hf) which we inherit.
    env["HF_HOME"] = os.environ.get("SG_HF_HOME", "/opt/sg-models/hf")
    env["HUGGINGFACE_HUB_CACHE"] = env["HF_HOME"]
    env["TORCH_HOME"] = os.environ.get("SG_TORCH_HOME", "/opt/sg-models/torch")
    # Defensive PYTHONPATH so the pipeline's cross-package imports (scene_graph,
    # _paths, geo_bounds) resolve regardless of cwd. The scripts also self-insert
    # these, so this is belt-and-suspenders.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO), str(REPO / "pipeline"), str(REPO / "pipeline2b"),
         str(REPO / "pipeline9"), env.get("PYTHONPATH", "")])
    return env


def _convert_to_viewer(slug: str) -> dict:
    """Map the pipeline's own output to the scene_graph.json schema our viewer
    reads (id, label, center[x,y,z], size[x,y,z] in the SPLAT-NATIVE frame; edges
    src/dst/relation). Geometry comes from <slug>_boxes_final.json (native splat
    frame, == our .ksplat frame). Edges come from <slug>_geo_true.json, whose
    node ids are the same box indices (export sets id = list index)."""
    boxes = json.loads((OUT / f"{slug}_boxes_final.json").read_text())["boxes"]
    nodes = [{
        "id": i,
        "label": b["label"],
        "center": [round(float(v), 4) for v in b["center"]],
        "size": [round(abs(float(v)), 4) for v in b["size"]],
    } for i, b in enumerate(boxes)]

    edges = []
    geo_path = OUT / f"{slug}_geo_true.json"
    if geo_path.exists():
        geo = json.loads(geo_path.read_text())
        n = len(nodes)
        for e in geo.get("edges", []):
            if 0 <= e.get("src", -1) < n and 0 <= e.get("dst", -1) < n:
                edges.append({"src": e["src"], "dst": e["dst"],
                              "relation": e.get("relation", "near")})

    return {
        "slug": slug,
        "coord_frame": "splat",
        "up_axis": "z",
        "version": 2,
        "nodes": nodes,
        "edges": edges,
        "labels": dict(Counter(n["label"] for n in nodes)),
    }


def _s3():
    return boto3.client(
        "s3", region_name=os.environ.get("S3_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY"))


def run(args) -> dict:
    slug = args.slug
    data_root = REPO / "data" / slug
    if OUT.exists():
        # a fresh container has none, but only_scenegraph re-runs may not — clear
        # any stale per-slug outputs so we never read a previous run's boxes.
        for f in OUT.glob(f"{slug}_*"):
            f.unlink()

    try:
        # ---- 1. stage LOCAL inputs into the pipeline's data_root -------------
        views_src = Path(args.views)
        if not views_src.is_dir() or not any(views_src.iterdir()):
            return {"error": f"no views under {views_src} — the CLIP relabel "
                             "stage needs the space's real per-view photos"}
        splat = data_root / "splat.ply"
        _link_or_copy(Path(args.splat).resolve(), splat)
        _link_or_copy(Path(args.pointcloud).resolve(), data_root / "pointcloud.ply")
        _link_or_copy(Path(args.cameras).resolve(), data_root / "cameras.json")
        _link_or_copy(Path(args.intrinsics).resolve(), data_root / "intrinsics.json")
        _link_or_copy(views_src.resolve(), data_root / "views")
        n_views = sum(1 for _ in (data_root / "views").iterdir())
        _register_space(slug, data_root, n_views)
        spaces_json = _write_space_config(slug, splat, args)
        print(f"[scenegraph] staged {slug}: {n_views} views, yaw={args.yaw_deg}", flush=True)

        # ---- 2. run the UPSTREAM pipeline end to end ------------------------
        cmd = [PY, str(RUN_SPACE), "--space", slug, "--no-install"]
        if args.skip_masks:
            cmd.append("--skip-masks")
        if args.skip_verify:
            cmd.append("--skip-verify")
        print("[scenegraph] $ " + " ".join(cmd), flush=True)
        # Stream the pipeline's output line-by-line so the worker log shows live
        # progress (a ~30-min run would otherwise be silent), while still keeping
        # every line in _LOG for the S3 pipeline.log dump.
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=_pipeline_env(spaces_json),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            _LOG.append(line.rstrip("\n"))
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"run_space_pipeline exited {rc}\n"
                               + "\n".join(_LOG[-40:]))

        # ---- 3. convert to our viewer schema + upload ----------------------
        sg = _convert_to_viewer(slug)
        s3 = _s3()
        bucket = args.bucket or os.environ["S3_BUCKET"]
        key = f"scans/{slug}/scene_graph.json"
        s3.put_object(Bucket=bucket, Key=key,
                      Body=json.dumps(sg).encode(), ContentType="application/json")
        # clear any stale error dump from a previous failed run of this slug
        try:
            s3.delete_object(Bucket=bucket, Key=f"scans/{slug}/scenegraph_error.txt")
        except Exception:
            pass
        # keep the pipeline's own outputs alongside for provenance/debugging
        for tag in ("boxes_final", "geo_true", "splat_to_pc_transform"):
            p = OUT / f"{slug}_{tag}.json"
            if p.exists():
                s3.upload_file(str(p), bucket, f"scans/{slug}/scenegraph/{slug}_{tag}.json")
        print(f"[scenegraph] uploaded {key} "
              f"({len(sg['nodes'])} nodes, {len(sg['edges'])} edges)", flush=True)
        return {"slug": slug, "scene_graph_key": key,
                "n_nodes": len(sg["nodes"]), "n_edges": len(sg["edges"]),
                "labels": sg["labels"]}

    finally:
        # ship the full pipeline log to S3 for diagnosis, then clean scratch.
        try:
            s3 = _s3()
            bucket = args.bucket or os.environ.get("S3_BUCKET")
            if bucket:
                s3.put_object(Bucket=bucket,
                              Key=f"scans/{slug}/scenegraph/debug/pipeline.log",
                              Body=("\n".join(_LOG)).encode("utf-8", "replace"))
        except Exception:
            pass
        shutil.rmtree(data_root, ignore_errors=True)
        (REPO / "spaces.json").unlink(missing_ok=True)
        (REPO / f"_sg_spaces_{slug}.json").unlink(missing_ok=True)
        for sub in ("out", "external/splat_analyzer"):
            for junk in (REPO / sub).glob(f"*{slug}*"):
                (junk.unlink() if junk.is_file()
                 else shutil.rmtree(junk, ignore_errors=True))


def _dump_error_s3(args, text: str) -> None:
    """Best-effort: write the failure to scans/<slug>/scenegraph_error.txt so it
    can be inspected without RunPod run-logs (which have no public API)."""
    try:
        bucket = args.bucket or os.environ["S3_BUCKET"]
        _s3().put_object(Bucket=bucket,
                         Key=f"scans/{args.slug}/scenegraph_error.txt",
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
    ap.add_argument("--workdir", required=True, help="scratch dir (reserved)")
    ap.add_argument("--bucket", default=os.environ.get("S3_BUCKET"))
    # per-space knobs (defaults documented in NOTES.md)
    ap.add_argument("--yaw-deg", default="0",
                    help="room yaw in the splat frame; 0 = detect as captured")
    ap.add_argument("--surface-label", default="table",
                    help="work-surface class for the top-down detector "
                         "(office=table, factory=workbench)")
    ap.add_argument("--max-long-m", default="2.1",
                    help="longest plausible single work-surface in metres "
                         "(office desk ~2.1, factory bench ~7.0)")
    ap.add_argument("--drop-labels", default="",
                    help="comma-separated classes to drop (unreliable in a space)")
    ap.add_argument("--skip-masks", action="store_true",
                    help="skip the SAM mask-refit stage")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the render-back IoU annotation stage")
    args = ap.parse_args()

    try:
        result = run(args)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[scenegraph] FAILED: {type(e).__name__}: {e}", flush=True)
        print(tb[-2500:], flush=True)
        _dump_error_s3(args, f"{type(e).__name__}: {e}\n\n{tb}\n\n"
                             "--- pipeline log tail ---\n" + "\n".join(_LOG)[-4000:])
        sys.exit(1)

    if result.get("error"):
        print(f"[scenegraph] ERROR: {result['error']}", flush=True)
        _dump_error_s3(args, str(result["error"]))
        sys.exit(1)
    print("[scenegraph] RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
