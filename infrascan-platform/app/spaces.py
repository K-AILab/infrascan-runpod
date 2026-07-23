"""Space data access — what the server reads/writes about Space rows.

Lookup helpers + permission checks. Per-space artifacts (the search index,
the proposals.jsonl, the topdown.png, the downsampled cloud) live on disk
under OUT_ROOT/<slug>/ and DATA_ROOT/<slug>/.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from . import config
from .db import get_conn, new_id, tx


# ── Disk layout per space ─────────────────────────────────────────────────
def data_dir(slug: str) -> Path:
    """Source data (views/, frames/, depth/, pointcloud.ply, cameras.json…)."""
    return config.DATA_ROOT / slug


def out_dir(slug: str) -> Path:
    """Pipeline output (proposals.jsonl, embeddings.npy, index.faiss, topdown.png…)."""
    return config.OUT_ROOT / slug


def web_dir(slug: str) -> Path:
    """Browser-served assets (downsampled_web.ply, cameras.json symlinks)."""
    return out_dir(slug) / "web"


def thumbnail_url(slug: str) -> str | None:
    """Pick a real thumbnail for the space — topdown if processed, else a
    preflight frame, else None (caller renders a gradient placeholder)."""
    topdown = web_dir(slug) / "topdown.png"
    if topdown.exists():
        return f"/spaces/{slug}/asset/topdown.png"
    pf_dir = data_dir(slug) / "preflight_frames"
    if pf_dir.exists():
        frames = sorted(p.name for p in pf_dir.glob("frame_*.jpg"))
        if frames:
            return f"/spaces/{slug}/asset/preflight_frames/{frames[0]}"
    return None


def resolve_asset_path(slug: str, path: str) -> Path | None:
    """Map an /asset/<path> request to a real file under the space's tree.
    Returns the path if it exists and is a file, else None.

    Lookup order:
        1. out/<slug>/web/<path>            (the canonical browser-served dir)
        2. data/<slug>/preflight_frames/... (when path starts with preflight_frames/)
    """
    web = web_dir(slug)
    candidate = web / path
    if candidate.exists() and candidate.is_file():
        return candidate
    if path.startswith("preflight_frames/"):
        candidate = data_dir(slug) / path
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


# ── Row lookups ───────────────────────────────────────────────────────────
def by_slug(slug: str) -> Optional[sqlite3.Row]:
    return get_conn().execute("SELECT * FROM spaces WHERE slug = ?", (slug,)).fetchone()


def visible_to(user_id: str) -> list[sqlite3.Row]:
    """Spaces this user owns or is a member of."""
    return list(get_conn().execute(
        """
        SELECT DISTINCT s.* FROM spaces s
          LEFT JOIN memberships m ON m.space_id = s.id AND m.user_id = ?
          WHERE s.owner_id = ? OR m.user_id = ?
          ORDER BY s.updated_at DESC
        """,
        (user_id, user_id, user_id),
    ).fetchall())


def all_spaces() -> list[sqlite3.Row]:
    """Every space, regardless of owner. Admin-only."""
    return list(get_conn().execute(
        "SELECT * FROM spaces ORDER BY updated_at DESC"
    ).fetchall())


# ── Mutations ─────────────────────────────────────────────────────────────
def create_space(
    slug: str,
    title: str,
    owner_id: str,
    *,
    status: str = "uploading",
    y_up: bool = True,
) -> str:
    sid = new_id()
    with tx() as conn:
        conn.execute(
            """INSERT INTO spaces(id, slug, title, owner_id, status, y_up)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (sid, slug, title, owner_id, status, 1 if y_up else 0),
        )
    return sid


def update_status(slug: str, status: str, *, n_views: int = None, n_scanpoints: int = None) -> None:
    fields = ["status = ?", "updated_at = datetime('now')"]
    vals: list = [status]
    if n_views is not None:
        fields.append("n_views = ?"); vals.append(int(n_views))
    if n_scanpoints is not None:
        fields.append("n_scanpoints = ?"); vals.append(int(n_scanpoints))
    vals.append(slug)
    with tx() as conn:
        conn.execute(f"UPDATE spaces SET {', '.join(fields)} WHERE slug = ?", vals)


# ── Permission helpers ────────────────────────────────────────────────────
def can_view(space: sqlite3.Row, user_id: str, user_role: str) -> bool:
    if user_role == "admin":
        return True
    if space["owner_id"] == user_id:
        return True
    row = get_conn().execute(
        "SELECT 1 FROM memberships WHERE space_id = ? AND user_id = ?",
        (space["id"], user_id),
    ).fetchone()
    return bool(row)


def can_edit(space: sqlite3.Row, user_id: str, user_role: str) -> bool:
    if user_role == "admin":
        return True
    if space["owner_id"] == user_id:
        return True
    row = get_conn().execute(
        "SELECT role_in_space FROM memberships WHERE space_id = ? AND user_id = ?",
        (space["id"], user_id),
    ).fetchone()
    return bool(row and row["role_in_space"] == "editor")
