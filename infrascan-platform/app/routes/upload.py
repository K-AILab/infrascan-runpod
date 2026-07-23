"""Upload routes — capturer file drop + multi-tier validation.

Flow:
    1. POST /upload      file lands on disk;
                         Tier 1 (already enforced in browser, re-checked here);
                         Tier 2 (ffprobe) runs synchronously (fast);
                         BackgroundTasks kicks Tier 3 (frame sampling).
    2. GET  /spaces/<slug>/preflight   poll for Tier 3 result.
    3. POST /spaces/<slug>/process     capturer accepts warnings; pipeline starts.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .. import config, spaces as space_repo, validation
from ..auth import require_user
from ..db import tx


REPO = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(REPO / "ui" / "templates"))

router = APIRouter(tags=["upload"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024 * 1024  # 50 GB; matches Tier 1 max_gb


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request, user=Depends(require_user)):
    return templates.TemplateResponse(request, "upload.html", { "user": user,
            "accept_rules": validation.ACCEPT_RULES,
        },
    )


@router.post("/upload")
async def upload_post(
    request: Request,
    bg: BackgroundTasks,
    user=Depends(require_user),
    title: str = Form(...),
    slug: str = Form(...),
    capture_type: str = Form("video"),
    file: UploadFile = File(...),
):
    slug = slug.strip().lower()
    if not slug or not slug.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "Slug must be alphanumeric (- and _ allowed).")
    if space_repo.by_slug(slug):
        raise HTTPException(409, "A space with that link already exists.")
    if capture_type not in validation.ACCEPT_RULES:
        raise HTTPException(400, "Unknown capture type.")

    # Persist file
    target_dir = space_repo.data_dir(slug) / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (file.filename or "input.bin")

    written = 0
    with target.open("wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                target.unlink(missing_ok=True)
                raise HTTPException(413, "File too large.")
            f.write(chunk)

    # ── Tier 1 (re-check what the browser claimed) ────────────────────────
    err = validation.tier1_check(target.name, written, capture_type)
    if err:
        target.unlink(missing_ok=True)
        raise HTTPException(400, err)

    # ── Tier 2 (ffprobe; synchronous, ~200 ms) ────────────────────────────
    report = validation.tier2_check(target, capture_type)
    if report.grade == "fail":
        # Hard block — keep the file for debugging but don't create the space.
        return _render_upload_with_report(request, user, capture_type, title, slug, report)

    # Create the space row in `preflight` state; Tier 3 will fill in the rest.
    space_repo.create_space(slug=slug, title=title, owner_id=user["id"], status="preflight")
    with tx() as conn:
        conn.execute("UPDATE spaces SET capture_type = ? WHERE slug = ?", (capture_type, slug))
        conn.execute("UPDATE spaces SET preflight_json = ? WHERE slug = ?",
                     (json.dumps(report.to_dict()), slug))

    # ── Tier 3 in the background ──────────────────────────────────────────
    bg.add_task(
        _run_tier3,
        slug=slug,
        file_path=str(target),
        capture_type=capture_type,
    )

    return RedirectResponse(f"/spaces/{slug}/preflight", status_code=303)


# ── Tier 3 background task ────────────────────────────────────────────────
def _run_tier3(slug: str, file_path: str, capture_type: str) -> None:
    from .. import validation
    out_dir = space_repo.data_dir(slug)
    base_report_json = None
    row = space_repo.by_slug(slug)
    if row and row["preflight_json"]:
        base_report_json = row["preflight_json"]
    base_report = None
    if base_report_json:
        d = json.loads(base_report_json)
        base_report = validation.PreflightReport(
            grade=d["grade"],
            checks=[validation.CheckResult(**c) for c in d["checks"]],
            summary=d.get("summary", ""),
            est_scanpoints=d.get("est_scanpoints"),
            est_views=d.get("est_views"),
            est_processing_minutes=d.get("est_processing_minutes"),
            sampled_frames=d.get("sampled_frames", []),
        )

    try:
        report = validation.tier3_preflight(
            Path(file_path), out_dir, capture_type, base_report=base_report,
        )
    except Exception as e:
        report = base_report or validation.PreflightReport(grade="warn")
        report.checks.append(validation.CheckResult(
            name="preflight",
            severity="warn",
            message=f"Preflight checks couldn't run cleanly ({type(e).__name__}); proceed at your own risk.",
        ))
        if report.grade == "pass":
            report.grade = "warn"

    with tx() as conn:
        conn.execute(
            "UPDATE spaces SET status='preflight_done', preflight_grade=?, preflight_json=?, updated_at=datetime('now') WHERE slug=?",
            (report.grade, json.dumps(report.to_dict()), slug),
        )


# ── Preflight result + accept-to-process ──────────────────────────────────
@router.get("/spaces/{slug}/preflight", response_class=HTMLResponse)
async def preflight_view(slug: str, request: Request, user=Depends(require_user)):
    row = space_repo.by_slug(slug)
    if not row:
        raise HTTPException(404, "Space not found.")
    if not space_repo.can_view(row, user["id"], user["role"]):
        raise HTTPException(403, "Forbidden.")
    report = None
    if row["preflight_json"]:
        report = json.loads(row["preflight_json"])
    return templates.TemplateResponse(request, "preflight.html", { "user": user, "space": dict(row), "report": report},
    )


@router.post("/spaces/{slug}/process")
async def start_processing(slug: str, user=Depends(require_user)):
    row = space_repo.by_slug(slug)
    if not row:
        raise HTTPException(404, "Space not found.")
    if not space_repo.can_view(row, user["id"], user["role"]):
        raise HTTPException(403, "Forbidden.")
    if row["status"] not in ("preflight_done", "preflight", "failed"):
        raise HTTPException(409, f"Space is currently {row['status']}; processing already in flight.")
    with tx() as conn:
        conn.execute(
            "UPDATE spaces SET status='processing', failure_stage=NULL, failure_reason=NULL, updated_at=datetime('now') WHERE slug=?",
            (slug,),
        )
    # The actual pipeline runner is a separate process; it polls `processing` rows.
    return RedirectResponse(f"/spaces/{slug}", status_code=303)


@router.get("/api/upload/{slug}/status")
async def upload_status(slug: str, user=Depends(require_user)):
    row = space_repo.by_slug(slug)
    if not row:
        raise HTTPException(404, "Space not found.")
    if not space_repo.can_view(row, user["id"], user["role"]):
        raise HTTPException(403, "Forbidden.")
    return {
        "slug": slug,
        "status": row["status"],
        "preflight_grade": row["preflight_grade"],
        "updated_at": row["updated_at"],
        "stage":         row["stage"],
        "stage_idx":     row["stage_idx"],
        "stage_total":   row["stage_total"],
        "stage_pct":     row["stage_pct"],
        "stage_text":    row["stage_text"],
    }


# ── Helpers ───────────────────────────────────────────────────────────────
def _render_upload_with_report(request, user, capture_type, title, slug, report):
    return templates.TemplateResponse(request, "upload.html", { "user": user,
            "accept_rules": validation.ACCEPT_RULES,
            "tier2_report": report.to_dict(),
            "form": {"capture_type": capture_type, "title": title, "slug": slug},
        },
        status_code=400,
    )
