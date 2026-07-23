"""Per-space routes: my-spaces list, space detail, 3D viewer, named-object API."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import config, spaces as space_repo
from ..auth import require_user, session_user
from ..db import get_conn, new_id, tx


REPO = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(REPO / "ui" / "templates"))

router = APIRouter(tags=["spaces"])

# Mirrors the keys in scripts/worker.py's PIPELINE list — used to render
# user-friendly stage labels on the detail page.
STAGE_PRETTY = {
    "stitch":      "Stitching dual-fisheye to a 360° video",
    "frames":      "Extracting frames from the video",
    "views":       "Sampling perspective views",
    "da3":         "Estimating depth + camera poses",
    "propose":     "Proposing objects per view",
    "embed":       "Computing visual embeddings",
    "match":       "Within-scanpoint dedup",
    "backproject": "Backprojecting objects to 3D",
    "merge":       "Cross-scanpoint object merge",
    "index":       "Building search index",
    "topdown":     "Rendering floor-plan",
    "downsample":  "Downsampling point cloud for the viewer",
    "done":        "Done",
}


# ── HTML ──────────────────────────────────────────────────────────────────
@router.get("/spaces", response_class=HTMLResponse)
async def my_spaces(request: Request, user=Depends(require_user)):
    rows = space_repo.visible_to(user["id"])
    spaces = []
    for r in rows:
        d = dict(r)
        d["thumbnail_url"] = space_repo.thumbnail_url(d["slug"])
        spaces.append(d)
    return templates.TemplateResponse(request, "spaces.html", {"user": user, "spaces": spaces})


@router.get("/spaces/{slug}")
async def space_detail(slug: str, request: Request, user=Depends(require_user)):
    row = space_repo.by_slug(slug)
    if not row:
        raise HTTPException(404, "Space not found")
    if not space_repo.can_view(row, user["id"], user["role"]):
        raise HTTPException(403, "Forbidden")

    # Legacy viewer (find-similar.js / app.js) does a synchronous GET to
    # ${API_BASE} = /spaces/<slug> at module-load time, expecting JSON with
    # a y_up flag. Browser navigation requests text/html. Branch on Accept.
    accept = request.headers.get("accept", "").lower()
    if "text/html" not in accept:
        return JSONResponse(_public_space(row))

    named = list(get_conn().execute(
        "SELECT id, name, slug FROM named_objects WHERE space_id = ? ORDER BY created_at DESC",
        (row["id"],),
    ).fetchall())

    space = dict(row)
    space["thumbnail_url"] = space_repo.thumbnail_url(space["slug"])
    # Decode preflight_json for inline summary on the detail page.
    preflight = None
    if row["preflight_json"]:
        try:
            preflight = json.loads(row["preflight_json"])
        except Exception:
            preflight = None
    return templates.TemplateResponse(request, "space-detail.html", {
            "user": user,
            "space": space,
            "named_objects": [dict(n) for n in named],
            "preflight": preflight,
            "stage_pretty": STAGE_PRETTY,
        },
    )


@router.get("/spaces/{slug}/viewer/", response_class=HTMLResponse)
async def space_viewer(slug: str, request: Request, user=Depends(require_user)):
    row = space_repo.by_slug(slug)
    if not row:
        raise HTTPException(404, "Space not found")
    if not space_repo.can_view(row, user["id"], user["role"]):
        raise HTTPException(403, "Forbidden")
    # The viewer only makes sense when the digital twin has been reconstructed.
    # For any earlier state, bounce back to the space detail page where the
    # user can see what's next.
    if row["status"] != "ready":
        return RedirectResponse(f"/spaces/{slug}?viewer_unavailable=1", status_code=303)
    return templates.TemplateResponse(request, "viewer.html", { "user": user, "space": dict(row)},
    )


@router.get("/api/spaces/{slug}/assets")
async def api_list_assets(slug: str, user=Depends(require_user)) -> dict:
    """Sizes (bytes) of the browser-served assets — used by viewer.js to render
    a real progress bar for the heavy ones (cf-tunnel strips Content-Length on
    chunked responses, so xhr.lengthComputable is unreliable)."""
    row = space_repo.by_slug(slug)
    if not row:
        raise HTTPException(404, "Space not found")
    if not space_repo.can_view(row, user["id"], user["role"]):
        raise HTTPException(403, "Forbidden")
    web = space_repo.web_dir(slug)
    out = {}
    for name in ("downsampled_web.ply", "cameras.json", "topdown.png", "bounds.json"):
        f = web / name
        if f.exists():
            try:
                out[name] = f.stat().st_size
            except OSError:
                pass
    return out


# ── Per-space static asset passthrough ────────────────────────────────────
# Serves /spaces/<slug>/asset/<path> from out/<slug>/web/<path>, gated by auth.
@router.get("/spaces/{slug}/asset/{path:path}")
async def space_asset(slug: str, path: str, user=Depends(require_user)):
    row = space_repo.by_slug(slug)
    if not row:
        raise HTTPException(404, "Space not found")
    if not space_repo.can_view(row, user["id"], user["role"]):
        raise HTTPException(403, "Forbidden")
    # Guard against path traversal
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(400, "Bad path")
    target = space_repo.resolve_asset_path(slug, path)
    if not target:
        raise HTTPException(404, "Asset not found")
    return FileResponse(target, headers={"Cache-Control": "no-store, max-age=0"})


# ── JSON API ──────────────────────────────────────────────────────────────
@router.get("/api/spaces")
async def api_list_spaces(request: Request, user=Depends(require_user)) -> dict:
    rows = space_repo.visible_to(user["id"])
    return {"spaces": [_public_space(r) for r in rows]}


@router.get("/api/spaces/{slug}")
async def api_get_space(slug: str, user=Depends(require_user)) -> dict:
    row = space_repo.by_slug(slug)
    if not row:
        raise HTTPException(404, "Space not found")
    if not space_repo.can_view(row, user["id"], user["role"]):
        raise HTTPException(403, "Forbidden")
    return _public_space(row)




# ── Named objects ─────────────────────────────────────────────────────────
_slug_re = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    s = _slug_re.sub("-", name.strip().lower()).strip("-")
    return s or "object"


@router.post("/api/spaces/{slug}/save")
async def api_save_named_object(
    slug: str,
    payload: dict,
    user=Depends(require_user),
) -> dict:
    """Save the visual prompt under `name`. The embedding + anchor come from the
    last /query call — the client passes them in.
    """
    row = space_repo.by_slug(slug)
    if not row:
        raise HTTPException(404, "Space not found")
    if not space_repo.can_view(row, user["id"], user["role"]):
        raise HTTPException(403, "Forbidden")

    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Name required")
    anchor_proposal_id = int(payload.get("anchor_proposal_id", 0))
    anchor_view_id = int(payload.get("anchor_view_id", 0))
    anchor_bbox = payload.get("anchor_bbox") or [0, 0, 0, 0]
    embedding_b64 = payload.get("embedding_b64")
    if not embedding_b64:
        raise HTTPException(400, "Embedding required")

    import base64
    blob = base64.b64decode(embedding_b64)

    obj_id = new_id()
    obj_slug = _slugify(name)
    with tx() as conn:
        # Ensure unique (space_id, slug)
        existing = conn.execute(
            "SELECT 1 FROM named_objects WHERE space_id = ? AND slug = ?",
            (row["id"], obj_slug),
        ).fetchone()
        if existing:
            raise HTTPException(409, "A name with that slug already exists in this space")
        conn.execute(
            """INSERT INTO named_objects(
                  id, space_id, name, slug, prompt_embedding,
                  anchor_proposal_id, anchor_view_id, anchor_bbox, created_by)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                obj_id, row["id"], name, obj_slug, blob,
                anchor_proposal_id, anchor_view_id, json.dumps(anchor_bbox), user["id"],
            ),
        )
    return {"id": obj_id, "slug": obj_slug, "name": name}


@router.get("/api/spaces/{slug}/named-objects")
async def api_list_named(slug: str, user=Depends(require_user)) -> dict:
    row = space_repo.by_slug(slug)
    if not row:
        raise HTTPException(404, "Space not found")
    if not space_repo.can_view(row, user["id"], user["role"]):
        raise HTTPException(403, "Forbidden")
    rows = get_conn().execute(
        "SELECT id, name, slug, created_at FROM named_objects WHERE space_id = ? ORDER BY name",
        (row["id"],),
    ).fetchall()
    return {"named_objects": [dict(r) for r in rows]}


# ── helpers ───────────────────────────────────────────────────────────────
def _public_space(row) -> dict:
    return {
        "slug": row["slug"],
        "title": row["title"],
        "status": row["status"],
        "n_views": row["n_views"],
        "n_scanpoints": row["n_scanpoints"],
        "updated_at": row["updated_at"],
        "y_up": bool(row["y_up"]),   # legacy viewer reads this
    }
