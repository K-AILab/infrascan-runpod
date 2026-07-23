"""Worker daemon — picks up spaces with status='processing' and runs the
reconstruction pipeline through to status='ready' (or 'failed').

Run:
    python -m scripts.worker

What it does, one space at a time (serial — single GPU):
    0.  Stitch .insv → 8K equirect mp4 (skipped if input already .mp4)
    1.  Extract frames from the mp4
    2.  Sample 12 yaws × 1 pitch perspective views per frame
    3.  Run DA3 to estimate depth + camera poses + point cloud
    4.  FastSAM object proposals per view
    5.  DINOv2 embeddings per proposal
    6.  Within-scanpoint LightGlue dedup
    7.  Cross-scanpoint backproject + embedding merge
    8.  Build FAISS index
    9.  Generate topdown floor-plan
    10. Downsample point cloud for the browser viewer
    11. Set status='ready' on the row

Failure handling: each stage runs as a subprocess. Non-zero exit →
`failure_stage` + `failure_reason` written; `status='failed'`.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

# Allow the worker to import the app package
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import config, spaces as space_repo
from app.db import init as db_init, tx, get_conn
from app.validation import FAILURE_HINTS


POLL_INTERVAL = 10   # seconds between idle polls
PY = sys.executable


ProgressParser = Callable[[str], Optional[Tuple[float, str]]]


@dataclass
class Stage:
    key: str                   # short name used in DB / logs
    pretty: str                # what the UI shows
    cmd: Callable[[str, Path, Path], list[str]]
    can_skip: Callable[[str, Path, Path], bool] = lambda *_: False
    parser: Optional[ProgressParser] = None
    # Optional filesystem-based fallback parser: returns (pct, text) using
    # the (slug, data, out) directories — useful when subprocess output is
    # quiet but files are landing on disk.
    fs_progress: Optional[Callable[[str, Path, Path], Optional[Tuple[float, str]]]] = None


# ─── Stage definitions ───────────────────────────────────────────────────
def _exists(p: Path) -> bool:
    return p.exists()


def stage_stitch_cmd(slug: str, data: Path, out: Path) -> list[str]:
    # Find the original upload
    src = next(iter(sorted((data / "uploads").iterdir())))
    target = data / "uploads" / "input_equirect.mp4"
    return [PY, str(REPO / "pipeline" / "_00_stitch_insv.py"),
            "--input", str(src), "--output", str(target)]


def stage_stitch_skip(slug: str, data: Path, out: Path) -> bool:
    return (data / "uploads" / "input_equirect.mp4").exists()


def stage_frames_cmd(slug: str, data: Path, out: Path) -> list[str]:
    src = data / "uploads" / "input_equirect.mp4"
    return [PY, str(REPO / "pipeline" / "00_video_to_img.py"),
            "--video", str(src),
            "--output_dir", str(data / "frames"),
            "--every_n", "2"]


def stage_frames_skip(slug: str, data: Path, out: Path) -> bool:
    d = data / "frames"
    return d.exists() and any(d.iterdir())


def stage_views_cmd(slug: str, data: Path, out: Path) -> list[str]:
    # 3-pitch sampling matches the canonical production config recorded in
    # data/experiments/EXPERIMENT_LOG.md ("504 + chunk_size=120 + 3-pitch").
    # Was single-pitch previously — caused missing ceiling/floor coverage and
    # blocked meaningful replicate testing against canonical outputs.
    return [PY, str(REPO / "pipeline" / "00a_sample_views.py"),
            "--input_dir", str(data / "frames"),
            "--output_dir", str(data / "views"),
            "--fov", "90",
            "--out_size", "504",
            "--pitches", "-30", "0", "30"]


def stage_views_skip(slug: str, data: Path, out: Path) -> bool:
    d = data / "views"
    return d.exists() and any(d.iterdir())


def stage_da3_cmd(slug: str, data: Path, out: Path) -> list[str]:
    # Use the streaming variant: jointly estimates depth + camera poses from
    # the perspective views (the intern's 00b_gen_da3.py assumes cameras.json
    # already exists from an upstream LiDAR step).
    return [PY, str(REPO / "pipeline" / "00b_da3_streaming.py"), "--space", slug, "--resume"]


def stage_da3_skip(slug: str, data: Path, out: Path) -> bool:
    cameras = data / "cameras.json"
    depth = data / "depth"
    return cameras.exists() and depth.exists() and any(depth.iterdir())


def stage_simple(script: str, can_skip_path: str) -> tuple[Callable, Callable]:
    """Helper for the homogeneous intern stages (--space <slug> + a final file)."""
    def cmd(slug, data, out):
        return [PY, str(REPO / "pipeline" / script), "--space", slug]
    def skip(slug, data, out):
        return (out / can_skip_path).exists()
    return cmd, skip


# ─── Per-stage progress parsers ───────────────────────────────────────────
_RE_FFMPEG_TIME = re.compile(r"time=(\d+):(\d+):([\d.]+)")
_RE_DA3 = re.compile(r"\[Progress\]:\s*(\d+)\s*/\s*(\d+)")
_RE_PROPOSE = re.compile(r"\[propose\]\s+(\d+)/(\d+)")
_RE_EMBED   = re.compile(r"\[embed\]\s+(\d+)/(\d+)")
_RE_MATCH   = re.compile(r"\[match\]\s+within-sp\s+(\d+)/(\d+)")
_RE_MERGE   = re.compile(r"\[merge\]\s+cosine chunk\s+([\d,]+)\s*/\s*([\d,]+)")


def parser_ffmpeg(line: str):
    m = _RE_FFMPEG_TIME.search(line)
    if not m:
        return None
    h, mn, s = m.group(1), m.group(2), m.group(3)
    secs = int(h) * 3600 + int(mn) * 60 + float(s)
    # We don't know the total duration here without ffprobing; surface
    # the timestamp as text and use a sentinel pct that nudges forward.
    return (min(0.99, secs / 60.0), f"encoded {h}:{mn}:{s}")


def parser_simple(rgx: re.Pattern):
    def _p(line: str):
        m = rgx.search(line)
        if not m:
            return None
        a, b = int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
        return (a / max(b, 1), f"{a:,} / {b:,}")
    return _p


def fs_count_progress(rel: str, expected: int):
    """Reports progress by counting files at data/<slug>/<rel>/ vs `expected`."""
    def _p(slug, data, out):
        d = data / rel
        if not d.exists():
            return None
        n = sum(1 for _ in d.iterdir())
        if not n:
            return None
        return (min(1.0, n / max(expected, 1)), f"{n:,} of ~{expected:,}")
    return _p


def _stage(key, pretty, cmd, skip, parser=None, fs=None):
    return Stage(key=key, pretty=pretty, cmd=cmd, can_skip=skip, parser=parser, fs_progress=fs)


_propose_cmd, _propose_skip = stage_simple("01_propose.py",       "proposals.jsonl")
_embed_cmd,   _embed_skip   = stage_simple("02_embed.py",         "embeddings.npy")
_match_cmd,   _match_skip   = stage_simple("02b_match_views.py",  "object_ids.npy")
_back_cmd,    _back_skip    = stage_simple("03_backproject.py",   "world_positions.json")
_merge_cmd,   _merge_skip   = stage_simple("03b_merge_groups.py", "merged_groups.json")
_index_cmd,   _index_skip   = stage_simple("04_index.py",         "index.faiss")


PIPELINE: list[Stage] = [
    _stage("stitch",     "Stitching dual-fisheye → equirect", stage_stitch_cmd, stage_stitch_skip, parser=parser_ffmpeg),
    _stage("frames",     "Extracting frames from video",      stage_frames_cmd, stage_frames_skip,
           fs=fs_count_progress("frames", expected=500)),
    _stage("views",      "Sampling perspective views",        stage_views_cmd,  stage_views_skip,
           fs=fs_count_progress("views",  expected=6000)),
    _stage("da3",        "Estimating depth + camera poses",   stage_da3_cmd,    stage_da3_skip,
           parser=parser_simple(_RE_DA3)),
    _stage("propose",    "Proposing objects per view",        _propose_cmd, _propose_skip, parser=parser_simple(_RE_PROPOSE)),
    _stage("embed",      "Computing visual embeddings",       _embed_cmd,   _embed_skip,   parser=parser_simple(_RE_EMBED)),
    _stage("match",      "Within-scanpoint dedup",            _match_cmd,   _match_skip,   parser=parser_simple(_RE_MATCH)),
    _stage("backproject","Backprojecting objects to 3D",      _back_cmd,    _back_skip),
    _stage("merge",      "Cross-scanpoint object merge",      _merge_cmd,   _merge_skip,   parser=parser_simple(_RE_MERGE)),
    _stage("index",      "Building search index",             _index_cmd,   _index_skip),
    _stage("topdown",    "Rendering floor-plan",
           lambda s,d,o:[PY, str(REPO/"pipeline"/"gen_topdown.py"), "--space", s],
           lambda s,d,o:(o/"web/topdown.png").exists()),
    _stage("downsample", "Downsampling point cloud for web",
           lambda s,d,o:[PY, str(REPO/"pipeline"/"downsample_ply.py"), s],
           lambda s,d,o:(o/"web/downsampled_web.ply").exists()),
]


# ─── DB helpers ──────────────────────────────────────────────────────────
def _set(slug: str, **fields) -> None:
    sets = ", ".join(f"{k} = ?" for k in fields)
    sets += ", updated_at = datetime('now')"
    with tx() as conn:
        conn.execute(f"UPDATE spaces SET {sets} WHERE slug = ?",
                     (*fields.values(), slug))


def _pick_one() -> Optional[str]:
    """Grab the oldest space waiting for processing. Returns its slug or None."""
    row = get_conn().execute(
        "SELECT slug FROM spaces WHERE status = 'processing' ORDER BY updated_at LIMIT 1"
    ).fetchone()
    return row["slug"] if row else None


def _count_outputs(data_dir: Path) -> tuple[int, int]:
    """Return (n_views, n_scanpoints) by inspecting disk."""
    import json as _json
    views_dir = data_dir / "views"
    n_views = sum(1 for _ in views_dir.iterdir()) if views_dir.exists() else 0
    cam_path = data_dir / "cameras.json"
    n_sps = 0
    if cam_path.exists():
        try:
            cams = _json.loads(cam_path.read_text())
            n_sps = len({c.get("frame") for c in cams if isinstance(c.get("frame"), int)})
        except Exception:
            n_sps = 0
    return n_views, n_sps


# ─── Per-stage runner with live progress ─────────────────────────────────
DB_FLUSH_INTERVAL = 2.0  # seconds — throttle DB writes


def _flush(slug: str, last_pct: float, last_text: str) -> None:
    _set(slug, stage_pct=round(last_pct, 4), stage_text=last_text[:200])


def _fs_watcher(slug: str, stage: Stage, data: Path, out: Path, stop_flag: dict):
    """Periodically nudge progress from filesystem state when no stdout regex."""
    while not stop_flag.get("stop"):
        try:
            r = stage.fs_progress(slug, data, out) if stage.fs_progress else None
            if r:
                pct, text = r
                _flush(slug, pct, text)
        except Exception:
            pass
        time.sleep(1.5)


def run_stage(slug: str, stage: Stage, data: Path, out: Path,
              idx: int, total: int) -> None:
    if stage.can_skip(slug, data, out):
        print(f"[worker] {slug} · {stage.key} already done — skipping")
        _set(slug,
             stage=stage.key, stage_idx=idx, stage_total=total,
             stage_pct=1.0, stage_text="(skipped)")
        return
    cmd = stage.cmd(slug, data, out)
    print(f"[worker] {slug} · {stage.key}: {' '.join(cmd)}", flush=True)

    _set(slug,
         stage=stage.key, stage_idx=idx, stage_total=total,
         stage_pct=0.0, stage_text="starting…",
         failure_stage=stage.key)   # also keep failure_stage for legacy

    # File-system fallback watcher
    import threading
    stop = {"stop": False}
    watcher = None
    if stage.fs_progress:
        watcher = threading.Thread(target=_fs_watcher,
                                   args=(slug, stage, data, out, stop),
                                   daemon=True)
        watcher.start()

    # Stream subprocess stdout, parse progress where we can
    proc = subprocess.Popen(cmd, cwd=str(REPO),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    last_flush = 0.0
    last_pct = 0.0
    last_text = "starting…"
    try:
        for line in proc.stdout:
            sys.stdout.write(line)  # echo to tmux log
            sys.stdout.flush()
            if stage.parser:
                r = stage.parser(line)
                if r:
                    last_pct, last_text = r
                    now = time.time()
                    if now - last_flush >= DB_FLUSH_INTERVAL:
                        _flush(slug, last_pct, last_text)
                        last_flush = now
        rc = proc.wait()
    finally:
        stop["stop"] = True

    if rc != 0:
        hint = FAILURE_HINTS.get(stage.key) or \
               f"The {stage.pretty.lower()} step exited with code {rc}."
        raise StageFailed(stage.key, hint)

    _set(slug, stage_pct=1.0, stage_text="done")


class StageFailed(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.message = message


# ─── Worker loop ─────────────────────────────────────────────────────────
def process_one(slug: str) -> None:
    data = space_repo.data_dir(slug)
    out = space_repo.out_dir(slug)
    out.mkdir(parents=True, exist_ok=True)
    (out / "web").mkdir(parents=True, exist_ok=True)

    print(f"\n[worker] === {slug} ===  data={data}  out={out}")
    started = time.time()
    total = len(PIPELINE)
    try:
        for idx, stage in enumerate(PIPELINE, start=1):
            run_stage(slug, stage, data, out, idx=idx, total=total)
        # Compute view + scanpoint counts from disk so the UI shows real numbers.
        n_views, n_sps = _count_outputs(data)
        _set(slug, status="ready", failure_stage=None, failure_reason=None,
             stage="done", stage_idx=total, stage_total=total,
             stage_pct=1.0, stage_text="ready",
             n_views=n_views, n_scanpoints=n_sps)
        print(f"[worker] ✓ {slug} ready in {(time.time()-started)/60:.0f} min "
              f"· {n_views} views · {n_sps} scanpoints")
    except StageFailed as e:
        print(f"[worker] ✗ {slug} failed at {e.stage}: {e.message}", file=sys.stderr)
        _set(slug, status="failed", failure_stage=e.stage, failure_reason=e.message)
    except Exception as e:                      # noqa: BLE001
        msg = f"Worker crashed: {type(e).__name__}: {e}"
        print(f"[worker] ✗ {slug} {msg}", file=sys.stderr)
        _set(slug, status="failed", failure_stage="worker", failure_reason=msg)


def main() -> None:
    config.ensure_dirs()
    db_init()

    print(f"[worker] started — polling every {POLL_INTERVAL}s for processing rows")

    # Friendly shutdown on SIGTERM/SIGINT
    stop = False
    def _shutdown(_signum, _frame):
        nonlocal stop
        stop = True
        print("[worker] shutdown requested, will exit after current space")
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while not stop:
        slug = _pick_one()
        if slug:
            process_one(slug)
        else:
            time.sleep(POLL_INTERVAL)
    print("[worker] bye")


if __name__ == "__main__":
    main()
