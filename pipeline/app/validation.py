"""Four-tier upload validation.

Tier 1 — pre-upload schemas (browser checks against these)
Tier 2 — ffprobe at receive time (container + codec + dims)
Tier 3 — background preflight: sample frames, score the scan
Tier 4 — pipeline stage reporter (records failure_stage / failure_reason)

All tiers default to SOFT WARNINGS. Only fundamental container / codec
failures hard-block — the rest are advisory and shown alongside a
visual preview before the capturer commits to processing.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from . import config


# ── Tier 1 — accept rules (mirrored in the browser as data attrs) ─────────
ACCEPT_RULES: dict[str, dict] = {
    "insta360": {"exts": [".insv", ".insp"],         "min_mb": 30,  "max_gb": 50},
    "video":    {"exts": [".mp4", ".mov", ".m4v"],   "min_mb": 30,  "max_gb": 50},
    "frames":   {"exts": [".zip", ".tar", ".tar.gz"], "min_mb": 100, "max_gb": 50},
    "lidar":    {"exts": [".e57"],                   "min_mb": 100, "max_gb": 200},
}


def tier1_check(filename: str, size_bytes: int, capture_type: str) -> Optional[str]:
    """Return None if OK, else a user-facing error message."""
    rule = ACCEPT_RULES.get(capture_type)
    if not rule:
        return f"Unknown capture type: {capture_type}"
    ext = "".join(Path(filename).suffixes[-2:]) if filename.endswith(".tar.gz") \
          else Path(filename).suffix.lower()
    if ext.lower() not in rule["exts"]:
        return f"Expected one of {rule['exts']}, got {ext!r}"
    if size_bytes < rule["min_mb"] * 1024 * 1024:
        return f"File too small (<{rule['min_mb']} MB) — looks like a placeholder, not a real scan."
    if size_bytes > rule["max_gb"] * 1024 * 1024 * 1024:
        return f"File too large (>{rule['max_gb']} GB) — please split or compress."
    return None


# ── Tier 2 — ffprobe at receive time ──────────────────────────────────────
@dataclass
class CheckResult:
    name: str
    severity: str           # 'ok' | 'warn' | 'fail'
    message: str
    detail: dict = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        return self.severity == "ok"


@dataclass
class PreflightReport:
    grade: str                       # 'pass' | 'warn' | 'fail'
    checks: list[CheckResult] = field(default_factory=list)
    summary: str = ""
    est_scanpoints: Optional[int] = None
    est_views: Optional[int] = None
    est_processing_minutes: Optional[int] = None
    sampled_frames: list[str] = field(default_factory=list)   # relative paths

    def to_dict(self) -> dict:
        return {
            "grade": self.grade,
            "checks": [asdict(c) for c in self.checks],
            "summary": self.summary,
            "est_scanpoints": self.est_scanpoints,
            "est_views": self.est_views,
            "est_processing_minutes": self.est_processing_minutes,
            "sampled_frames": self.sampled_frames,
        }


def _ffprobe_bin() -> str:
    return os.environ.get("INFRASCAN_FFPROBE", shutil.which("ffprobe") or "ffprobe")


def _ffmpeg_bin() -> str:
    return os.environ.get("INFRASCAN_FFMPEG", shutil.which("ffmpeg") or "ffmpeg")


def _ffprobe(path: Path) -> dict:
    """Run ffprobe; return a normalised dict. Raises if ffprobe itself errors."""
    cmd = [
        _ffprobe_bin(),
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
    return json.loads(out.stdout)


def _rate_to_float(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/", 1)
        return float(num) / max(float(den), 1.0)
    return float(rate)


def tier2_check(file_path: Path, capture_type: str) -> PreflightReport:
    """Hard-blocking container checks. Returns grade='fail' for any hard block."""
    report = PreflightReport(grade="pass")

    if capture_type == "frames":
        # Archive — defer to Tier 3 (we extract and look at one frame).
        report.checks.append(CheckResult(
            name="container",
            severity="ok",
            message="Archive accepted — we'll look at its contents next.",
        ))
        return report

    if capture_type == "lidar":
        # E57 — minimum sanity check by magic bytes, full support is later.
        report.grade = "warn"
        report.checks.append(CheckResult(
            name="container",
            severity="warn",
            message="LiDAR (.e57) support is in preview — processing is not enabled yet.",
        ))
        return report

    try:
        info = _ffprobe(file_path)
    except subprocess.CalledProcessError as e:
        report.grade = "fail"
        report.checks.append(CheckResult(
            name="container",
            severity="fail",
            message="We couldn't read this file as a video. The file may be corrupted.",
            detail={"stderr": (e.stderr or "")[:500]},
        ))
        return report
    except subprocess.TimeoutExpired:
        report.grade = "fail"
        report.checks.append(CheckResult(
            name="container",
            severity="fail",
            message="The file took too long to probe — likely corrupted or truncated.",
        ))
        return report

    video_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        report.grade = "fail"
        report.checks.append(CheckResult(
            name="streams",
            severity="fail",
            message="No video stream found inside the file.",
        ))
        return report

    # Codec check
    bad_codec = next((s["codec_name"] for s in video_streams
                      if s.get("codec_name") not in ("h264", "hevc", "h265")), None)
    if bad_codec:
        report.grade = "fail"
        report.checks.append(CheckResult(
            name="codec",
            severity="fail",
            message=f"Unsupported video codec: {bad_codec}. We accept H.264 and HEVC.",
            detail={"codec": bad_codec},
        ))
        return report

    # Dual-stream expectation for .insv
    if capture_type == "insta360":
        if len(video_streams) != 2:
            report.grade = "warn"
            report.checks.append(CheckResult(
                name="streams",
                severity="warn",
                message=(f"Expected 2 video streams (front + back fisheye) in an Insta360 "
                         f".insv, got {len(video_streams)}. We'll try anyway."),
            ))
        else:
            report.checks.append(CheckResult(
                name="streams",
                severity="ok",
                message="Found front + back fisheye streams.",
            ))

    # Equirect aspect ratio for stitched video
    if capture_type == "video":
        s0 = video_streams[0]
        w, h = int(s0.get("width", 0)), int(s0.get("height", 0))
        aspect = w / max(h, 1)
        if abs(aspect - 2.0) > 0.04:
            report.grade = "warn"
            report.checks.append(CheckResult(
                name="projection",
                severity="warn",
                message=(f"This video is {w}×{h} (aspect {aspect:.2f}) — equirect 360° "
                         f"video is usually 2:1. Did you upload a regular video by mistake?"),
                detail={"width": w, "height": h, "aspect": aspect},
            ))
        else:
            report.checks.append(CheckResult(
                name="projection",
                severity="ok",
                message=f"Equirect-shaped video ({w}×{h}).",
            ))

    # Resolution sanity
    s0 = video_streams[0]
    w, h = int(s0.get("width", 0)), int(s0.get("height", 0))
    per_stream_min = 1920 if capture_type == "insta360" else 3840
    if w < per_stream_min:
        report.grade = "warn" if report.grade == "pass" else report.grade
        report.checks.append(CheckResult(
            name="resolution",
            severity="warn",
            message=(f"Video is {w}px wide — at this resolution the digital twin will be "
                     f"low detail. We recommend at least {per_stream_min}px wide."),
        ))
    else:
        report.checks.append(CheckResult(
            name="resolution",
            severity="ok",
            message=f"Resolution looks good ({w}×{h}).",
        ))

    # Duration + estimated scanpoints
    duration = 0.0
    try:
        duration = float(info.get("format", {}).get("duration", 0.0))
    except Exception:
        pass
    fps = _rate_to_float(s0.get("r_frame_rate", "0/0"))
    every_n_default = 2  # matches the pipeline default
    est_frames = int((duration * fps) / every_n_default) if duration and fps else 0
    est_scanpoints = est_frames
    est_views = est_scanpoints * 36  # 12 yaws × 3 pitches

    report.est_scanpoints = est_scanpoints
    report.est_views = est_views
    report.est_processing_minutes = max(30, int(est_scanpoints * 0.6))  # rough heuristic

    if duration < 20:
        report.grade = "warn" if report.grade == "pass" else report.grade
        report.checks.append(CheckResult(
            name="duration",
            severity="warn",
            message=(f"Recording is only {duration:.0f} s — fewer than 10 scanpoints will "
                     f"land in the index. Walk for at least 20 s for a usable twin."),
            detail={"duration_seconds": duration},
        ))
    elif duration > 30 * 60:
        report.grade = "warn" if report.grade == "pass" else report.grade
        report.checks.append(CheckResult(
            name="duration",
            severity="warn",
            message=f"Recording is {duration/60:.0f} min — processing will take many hours.",
            detail={"duration_seconds": duration},
        ))
    else:
        report.checks.append(CheckResult(
            name="duration",
            severity="ok",
            message=f"Duration: {duration:.0f} s · ~{est_scanpoints} scanpoints expected.",
            detail={"duration_seconds": duration, "fps": fps},
        ))

    if fps < 24:
        report.grade = "warn" if report.grade == "pass" else report.grade
        report.checks.append(CheckResult(
            name="frame_rate",
            severity="warn",
            message=f"Frame rate is {fps:.1f} fps. Below 24 fps the walk-through feels choppy.",
        ))

    report.summary = _summarize(report)
    return report


# ── Tier 3 — sampled-frame preflight ──────────────────────────────────────
def _luma_mean(jpg_path: Path) -> float:
    """Mean luminance 0-255. Tiny dep — uses PIL only."""
    from PIL import Image
    with Image.open(jpg_path) as im:
        gs = im.convert("L")
        # downsample for speed
        gs.thumbnail((128, 128))
        px = gs.getdata()
        if not px:
            return 0.0
        return sum(px) / len(px)


def _sample_frames(file_path: Path, out_dir: Path, n: int = 10) -> list[Path]:
    """Extract n frames spaced evenly through the video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Get duration first.
    info = _ffprobe(file_path)
    duration = float(info.get("format", {}).get("duration", 0.0))
    if duration <= 0:
        return []
    timestamps = [duration * (i + 0.5) / n for i in range(n)]
    out_paths = []
    for i, ts in enumerate(timestamps):
        out = out_dir / f"frame_{i:02d}.jpg"
        cmd = [
            _ffmpeg_bin(), "-y", "-v", "error",
            "-ss", f"{ts:.3f}",
            "-i", str(file_path),
            "-vframes", "1",
            "-vf", "scale=640:-1",
            str(out),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            if out.exists():
                out_paths.append(out)
        except Exception:
            pass
    return out_paths


def tier3_preflight(file_path: Path, out_dir: Path, capture_type: str,
                    base_report: Optional[PreflightReport] = None) -> PreflightReport:
    """Sample frames, score the scan visually, attach a preview gallery."""
    report = base_report or PreflightReport(grade="pass")

    frames = _sample_frames(file_path, out_dir / "preflight_frames", n=10)
    if not frames:
        report.checks.append(CheckResult(
            name="preview",
            severity="warn",
            message="Couldn't extract preview frames — pipeline may also struggle to read this.",
        ))
        report.grade = "warn"
        report.summary = _summarize(report)
        return report

    # Brightness — too dark?
    lumas = [_luma_mean(f) for f in frames]
    median_luma = sorted(lumas)[len(lumas) // 2]
    if median_luma < 24:
        report.grade = "warn"
        report.checks.append(CheckResult(
            name="brightness",
            severity="warn",
            message=(f"Recording is very dark (median brightness {median_luma:.0f}/255). "
                     f"Was the lens cap on, or was the scene unlit?"),
            detail={"median_luma": median_luma},
        ))
    elif median_luma > 230:
        report.grade = "warn"
        report.checks.append(CheckResult(
            name="brightness",
            severity="warn",
            message=(f"Recording is very bright (median brightness {median_luma:.0f}/255). "
                     f"Was the sensor saturated?"),
        ))
    else:
        report.checks.append(CheckResult(
            name="brightness",
            severity="ok",
            message=f"Scene brightness ok (median {median_luma:.0f}/255).",
        ))

    # Frame-to-frame variation: capturer actually moved?
    deltas = []
    for a, b in zip(lumas, lumas[1:]):
        deltas.append(abs(a - b))
    mean_delta = sum(deltas) / max(len(deltas), 1)
    if mean_delta < 2.0:
        report.grade = "warn"
        report.checks.append(CheckResult(
            name="motion",
            severity="warn",
            message=("Looks like the camera barely moved — depth needs parallax. "
                     "If you stood still, the twin will be a single viewpoint."),
            detail={"mean_inter_frame_delta": mean_delta},
        ))
    else:
        report.checks.append(CheckResult(
            name="motion",
            severity="ok",
            message="The scene changes frame-to-frame — capturer moved as expected.",
        ))

    # Loop detection (very cheap proxy): is the last frame's brightness near the first's?
    if len(lumas) >= 6:
        first, last = lumas[0], lumas[-1]
        if abs(first - last) < 3.0 and median_luma > 30:
            report.checks.append(CheckResult(
                name="loop",
                severity="warn",
                message=("Start and end of the recording look similar — you may have "
                         "walked in a loop. That's fine, but the back half won't add new space."),
            ))
            # Don't downgrade to warn for this alone — it's informative.

    # Record paths relative to the SPACE root, so the asset route resolves them.
    if (out_dir / "preflight_frames").exists():
        report.sampled_frames = [f"preflight_frames/{f.name}" for f in frames]

    report.summary = _summarize(report)
    return report


def _summarize(report: PreflightReport) -> str:
    warns = sum(1 for c in report.checks if c.severity == "warn")
    fails = sum(1 for c in report.checks if c.severity == "fail")
    if fails:
        return f"{fails} hard issue{'' if fails == 1 else 's'} — fix and re-upload."
    if warns:
        return f"{warns} thing{'' if warns == 1 else 's'} worth checking, but you can process anyway."
    return "Looks good. Ready to process."


# ── Tier 4 — pipeline stage reporter ──────────────────────────────────────
# Called from pipeline runner. Updates spaces.failure_stage / failure_reason.
def record_stage_failure(slug: str, stage: str, reason: str) -> None:
    from .db import tx
    with tx() as conn:
        conn.execute(
            """UPDATE spaces
                  SET status = 'failed',
                      failure_stage  = ?,
                      failure_reason = ?,
                      updated_at = datetime('now')
                WHERE slug = ?""",
            (stage, reason, slug),
        )


def record_stage_ok(slug: str, _stage: str) -> None:
    # Stage success is implicit — we just touch updated_at.
    from .db import tx
    with tx() as conn:
        conn.execute("UPDATE spaces SET updated_at = datetime('now') WHERE slug = ?", (slug,))


# Mapping for common failure modes — used by pipeline runner to translate
# raw exit codes / stderr into capturer-friendly messages.
FAILURE_HINTS = {
    "00b_gen_da3": "We couldn't reconstruct depth from this video. The scene may have too "
                   "little visual texture (e.g. mostly blank walls or a single solid colour).",
    "01_propose":  "We couldn't propose any objects from the views. The scene may be too "
                   "uniform or low-contrast.",
    "02_embed":    "We could see objects but couldn't compute embeddings for them. The "
                   "compute backend may have run out of memory.",
    "03_backproject":
                   "Per-object depth couldn't be sampled — most candidate objects landed on "
                   "reflective surfaces (glass, mirrors) where depth is unreliable.",
    "04_index":    "We built embeddings but couldn't write the search index.",
}
