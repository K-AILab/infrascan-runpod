"""Admin routes — gallery, users, all spaces, logs."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .. import spaces as space_repo
from ..auth import (
    create_user, get_user, get_user_by_email,
    require_admin,
)
from ..db import get_conn, tx


REPO = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(REPO / "ui" / "templates"))

router = APIRouter(prefix="", tags=["admin"])


@router.get("/gallery", response_class=HTMLResponse)
async def gallery(request: Request, user=Depends(require_admin)):
    rows = space_repo.all_spaces()
    return templates.TemplateResponse(request, "admin-gallery.html", { "user": user, "spaces": [dict(r) for r in rows]},
    )


@router.get("/users", response_class=HTMLResponse)
async def users_page(request: Request, user=Depends(require_admin)):
    rows = get_conn().execute(
        """SELECT u.*, COUNT(s.id) AS owned_spaces FROM users u
              LEFT JOIN spaces s ON s.owner_id = u.id
              GROUP BY u.id
              ORDER BY u.created_at DESC"""
    ).fetchall()
    return templates.TemplateResponse(request, "admin-users.html", { "user": user, "users": [dict(r) for r in rows]},
    )


@router.post("/users")
async def create_user_post(
    request: Request,
    email: str = Form(...),
    name: str = Form(...),
    role: str = Form("capturer"),
    password: str = Form(...),
    user=Depends(require_admin),
):
    if get_user_by_email(email):
        raise HTTPException(409, "Email already in use.")
    create_user(email=email, name=name, password=password, role=role)
    return RedirectResponse("/users", status_code=303)


@router.get("/spaces", response_class=HTMLResponse)
async def all_spaces_page(request: Request, user=Depends(require_admin)):
    rows = space_repo.all_spaces()
    # Owner emails
    owners = {r["id"]: dict(r) for r in get_conn().execute("SELECT * FROM users").fetchall()}
    enriched = []
    for s in rows:
        s = dict(s)
        owner = owners.get(s["owner_id"]) or {}
        s["owner_email"] = owner.get("email", "")
        s["owner_name"] = owner.get("name", "")
        enriched.append(s)
    return templates.TemplateResponse(request, "admin-spaces.html", { "user": user, "spaces": enriched},
    )


@router.post("/spaces/{slug}/reassign")
async def reassign_space(
    slug: str,
    new_owner_email: str = Form(...),
    user=Depends(require_admin),
):
    row = space_repo.by_slug(slug)
    if not row:
        raise HTTPException(404, "Space not found.")
    new_owner = get_user_by_email(new_owner_email)
    if not new_owner:
        raise HTTPException(404, "User not found.")
    with tx() as conn:
        conn.execute("UPDATE spaces SET owner_id = ?, updated_at = datetime('now') WHERE slug = ?",
                     (new_owner["id"], slug))
    return RedirectResponse("/spaces", status_code=303)


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request, user=Depends(require_admin)):
    # In production this tails systemd journals. Stub for v0.
    return templates.TemplateResponse(request, "admin-logs.html", { "user": user, "log_lines": ["(log tailing not yet wired)"]},
    )
