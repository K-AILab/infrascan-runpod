"""Per-space paths — same API the intern's pipeline scripts expect, but
backed by the platform's SQLite DB instead of a spaces.json file.

The intern's scripts call `from _paths import space, space_choices` and
expect a dict with keys: data_root, views, cameras, intrinsics, da3,
pointcloud, out_dir, y_up, n_views, n_scanpoints. We provide exactly
that, sourced from the platform's `app.spaces` module.
"""
from __future__ import annotations

import sys
from pathlib import Path


# Allow pipeline scripts to import the platform app even when invoked
# as `python pipeline/01_propose.py --space …` from the repo root.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app import config as _cfg     # noqa: E402
from app import spaces as _srep     # noqa: E402
from app.db import get_conn, init as _db_init  # noqa: E402


def _ensure_db_ready() -> None:
    _cfg.ensure_dirs()
    _db_init()


def space_choices() -> list[str]:
    _ensure_db_ready()
    return [r["slug"] for r in get_conn().execute(
        "SELECT slug FROM spaces ORDER BY slug"
    ).fetchall()]


def space(name: str) -> dict:
    _ensure_db_ready()
    row = _srep.by_slug(name)
    if not row:
        raise KeyError(
            f"Space '{name}' not in the database. "
            f"Register it via the platform's upload flow or "
            f"scripts/register_space.py."
        )
    data_root = _srep.data_dir(name)
    out_dir_ = _srep.out_dir(name)
    return {
        "name":         row["slug"],
        "title":        row["title"],
        "data_root":    data_root,
        "views":        data_root / "views",
        "cameras":      data_root / "cameras.json",
        "intrinsics":   data_root / "intrinsics.json",
        "da3":          data_root / "depth",
        "pointcloud":   data_root / "pointcloud.ply",
        "out_dir":      out_dir_,
        "y_up":         bool(row["y_up"]),
        "n_views":      int(row["n_views"]),
        "n_scanpoints": int(row["n_scanpoints"]),
    }


SPACE_CHOICES = space_choices()


def out_dir(name: str) -> Path:
    return space(name)["out_dir"]
