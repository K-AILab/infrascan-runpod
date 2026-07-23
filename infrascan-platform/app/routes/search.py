"""Search routes — cross-space + per-space query.

The query implementation is left as a small adapter around the per-space
search index. The platform stores the index on disk under
out/<slug>/index.faiss, builds it with the offline pipeline, and queries it
at request time. faiss-cpu is the run-time dependency.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Dict, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .. import config, spaces as space_repo
from ..auth import require_user
from ..db import get_conn


REPO = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(REPO / "ui" / "templates"))

router = APIRouter(tags=["search"])


# ── Index cache (one IndexFlatIP per space, loaded lazily) ────────────────
_indices: Dict[str, "faiss.Index"] = {}
_meta: Dict[str, dict] = {}
_lock = Lock()


def _load_index(slug: str):
    import faiss  # local — only when a query happens
    if slug in _indices:
        return _indices[slug], _meta[slug]
    with _lock:
        if slug in _indices:
            return _indices[slug], _meta[slug]
        idx_path = space_repo.out_dir(slug) / "index.faiss"
        meta_path = space_repo.out_dir(slug) / "metadata.json"
        if not idx_path.exists():
            raise HTTPException(503, "Search index for this space is not built yet.")
        idx = faiss.read_index(str(idx_path))
        meta = {}
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text())
        _indices[slug] = idx
        _meta[slug] = meta
        return idx, meta


# ── HTML ──────────────────────────────────────────────────────────────────
@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, user=Depends(require_user)):
    visible = space_repo.visible_to(user["id"])
    return templates.TemplateResponse(request, "search.html", { "user": user, "spaces": [dict(r) for r in visible]},
    )


# ── JSON ──────────────────────────────────────────────────────────────────
@router.post("/api/spaces/{slug}/query")
async def api_query(
    slug: str,
    payload: dict,
    user=Depends(require_user),
) -> dict:
    """Top-K nearest matches inside one space.

    Payload:
      {
        "embedding_b64": "...",     # 768-d float32 vector (base64)
        "top_k": 40                  # optional
      }
    """
    row = space_repo.by_slug(slug)
    if not row:
        raise HTTPException(404, "Space not found")
    if not space_repo.can_view(row, user["id"], user["role"]):
        raise HTTPException(403, "Forbidden")

    embedding_b64 = payload.get("embedding_b64")
    if not embedding_b64:
        raise HTTPException(400, "Embedding required")
    import base64
    arr = np.frombuffer(base64.b64decode(embedding_b64), dtype=np.float32)
    if arr.size == 0 or arr.size % 768 != 0:
        raise HTTPException(400, "Embedding must be 768-d float32")
    q = arr.reshape(1, -1).astype(np.float32)

    top_k = int(payload.get("top_k", 40))
    top_k = max(1, min(200, top_k))

    idx, meta = _load_index(slug)
    scores, indices = idx.search(q, top_k)

    hits = []
    for score, i in zip(scores[0].tolist(), indices[0].tolist()):
        if i < 0:
            continue
        item = {"index": int(i), "score": float(score)}
        # Map to proposal metadata if available
        if isinstance(meta.get("proposals"), list) and 0 <= i < len(meta["proposals"]):
            item.update(meta["proposals"][i])
        hits.append(item)

    return {"hits": hits, "took_ms": None}
