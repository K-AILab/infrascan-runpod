"""Chunked upload — bypasses Cloudflare's 100 MB single-request cap.

Protocol (small + custom, not full tus.io):

  POST /upload/start                ─ open a session
      JSON in : {filename, total_bytes, capture_type, title, slug}
      JSON out: {session_id, chunk_size}

  POST /upload/chunk/<session_id>   ─ stream one chunk
      query   : ?offset=<bytes>
      body    : raw bytes (≤ chunk_size)
      JSON out: {received, total_bytes}

  POST /upload/finalize/<session_id>
                                    ─ verify, run Tier 1+2, start Tier 3
      JSON out: {redirect: "/spaces/<slug>/preflight"}

Sessions live in memory + on disk (data/<slug>/uploads/.parts/). Re-uploading
the same chunk is idempotent. Sessions auto-expire after 24 h.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Dict

from fastapi import (
    APIRouter, BackgroundTasks, Depends, HTTPException, Request,
)

from .. import spaces as space_repo, validation
from ..auth import require_user
from ..db import tx
from .upload import _run_tier3   # reuse


router = APIRouter(tags=["upload-chunked"])

CHUNK_SIZE = 50 * 1024 * 1024     # 50 MB — safely under CF's 100 MB free-tier cap
SESSION_TTL = 24 * 3600           # 24 h

_sessions: Dict[str, dict] = {}
_lock = Lock()


# ── 1. start ──────────────────────────────────────────────────────────────
@router.post("/upload/start")
async def upload_start(request: Request, user=Depends(require_user)) -> dict:
    payload = await request.json()
    filename = (payload.get("filename") or "").strip()
    total = int(payload.get("total_bytes", 0))
    capture_type = payload.get("capture_type") or "video"
    title = (payload.get("title") or "").strip()
    slug = (payload.get("slug") or "").strip().lower()

    if not filename or total <= 0:
        raise HTTPException(400, "filename and total_bytes required.")
    if not slug or not slug.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "Slug must be alphanumeric (- and _ allowed).")
    if space_repo.by_slug(slug):
        raise HTTPException(409, "A space with that link already exists.")
    if capture_type not in validation.ACCEPT_RULES:
        raise HTTPException(400, "Unknown capture type.")

    err = validation.tier1_check(filename, total, capture_type)
    if err:
        raise HTTPException(400, err)

    sid = uuid.uuid4().hex
    target_dir = space_repo.data_dir(slug) / "uploads"
    parts_dir = target_dir / ".parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    with _lock:
        _sessions[sid] = {
            "user_id": user["id"],
            "filename": filename,
            "total": total,
            "capture_type": capture_type,
            "title": title,
            "slug": slug,
            "parts_dir": str(parts_dir),
            "target": str(target_dir / filename),
            "created": time.time(),
        }
    return {"session_id": sid, "chunk_size": CHUNK_SIZE, "total_bytes": total}


# ── 2. chunk ──────────────────────────────────────────────────────────────
@router.post("/upload/chunk/{session_id}")
async def upload_chunk(
    session_id: str,
    request: Request,
    offset: int = 0,
    user=Depends(require_user),
) -> dict:
    sess = _sessions.get(session_id)
    if not sess or sess["user_id"] != user["id"]:
        raise HTTPException(404, "Unknown upload session.")

    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty chunk body.")
    if offset < 0 or offset + len(body) > sess["total"]:
        raise HTTPException(416, "Chunk out of range.")
    if len(body) > CHUNK_SIZE * 1.1:   # 10% slack
        raise HTTPException(413, "Chunk larger than negotiated chunk_size.")

    # idempotent: each chunk is written to its own indexed file.
    idx = offset // CHUNK_SIZE
    part = Path(sess["parts_dir"]) / f"{idx:06d}.part"
    part.write_bytes(body)

    # touch session
    sess["last_seen"] = time.time()
    received = _bytes_received(sess)
    return {"received": received, "total_bytes": sess["total"]}


# ── 3. finalize ───────────────────────────────────────────────────────────
@router.post("/upload/finalize/{session_id}")
async def upload_finalize(
    session_id: str,
    bg: BackgroundTasks,
    user=Depends(require_user),
) -> dict:
    sess = _sessions.get(session_id)
    if not sess or sess["user_id"] != user["id"]:
        raise HTTPException(404, "Unknown upload session.")

    received = _bytes_received(sess)
    if received != sess["total"]:
        raise HTTPException(409, f"Missing chunks: have {received}, expected {sess['total']}.")

    # Stitch parts → final file
    target = Path(sess["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = Path(sess["parts_dir"])
    parts = sorted(parts_dir.glob("*.part"))
    with target.open("wb") as out:
        for p in parts:
            out.write(p.read_bytes())
    # Cleanup parts
    for p in parts:
        p.unlink(missing_ok=True)
    try:
        parts_dir.rmdir()
    except OSError:
        pass

    # Tier 1 re-check
    err = validation.tier1_check(target.name, target.stat().st_size, sess["capture_type"])
    if err:
        target.unlink(missing_ok=True)
        raise HTTPException(400, err)

    # Tier 2 (synchronous)
    report = validation.tier2_check(target, sess["capture_type"])
    if report.grade == "fail":
        msgs = [c.message for c in report.checks if c.severity == "fail"]
        raise HTTPException(400, "; ".join(msgs) or "Upload failed Tier 2 checks.")

    # Create the space row
    space_repo.create_space(slug=sess["slug"], title=sess["title"], owner_id=user["id"], status="preflight")
    with tx() as conn:
        conn.execute("UPDATE spaces SET capture_type = ? WHERE slug = ?",
                     (sess["capture_type"], sess["slug"]))
        conn.execute("UPDATE spaces SET preflight_json = ? WHERE slug = ?",
                     (json.dumps(report.to_dict()), sess["slug"]))

    # Tier 3 in background
    bg.add_task(_run_tier3, slug=sess["slug"], file_path=str(target), capture_type=sess["capture_type"])

    # Forget the session
    with _lock:
        _sessions.pop(session_id, None)

    return {"redirect": f"/spaces/{sess['slug']}/preflight"}


# ── helpers ───────────────────────────────────────────────────────────────
def _bytes_received(sess: dict) -> int:
    parts_dir = Path(sess["parts_dir"])
    if not parts_dir.exists():
        return 0
    return sum(p.stat().st_size for p in parts_dir.glob("*.part"))


# Periodic session sweep (called by a background task in main)
def sweep_expired() -> int:
    now = time.time()
    dead = [sid for sid, s in _sessions.items() if now - s.get("created", now) > SESSION_TTL]
    for sid in dead:
        _sessions.pop(sid, None)
    return len(dead)
