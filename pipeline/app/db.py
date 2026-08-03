"""SQLite database — schema, migrations, and a tiny connection helper.

This file is the single source of truth for the on-disk schema. The
mapping to entities is documented in `docs/ERD.md`.
"""
from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import config


SCHEMA_VERSION = 3

INITIAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('capturer','manager','admin')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at    TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at      TEXT NOT NULL,
    last_seen_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

CREATE TABLE IF NOT EXISTS spaces (
    id              TEXT PRIMARY KEY,
    slug            TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    owner_id        TEXT NOT NULL REFERENCES users(id),
    status          TEXT NOT NULL CHECK (status IN
                       ('uploading','preflight','preflight_done','processing','ready','failed')),
    capture_type    TEXT NOT NULL DEFAULT 'video'
                       CHECK (capture_type IN ('insta360','video','frames','lidar')),
    n_views         INTEGER NOT NULL DEFAULT 0,
    n_scanpoints    INTEGER NOT NULL DEFAULT 0,
    y_up            INTEGER NOT NULL DEFAULT 1,

    -- Preflight outcomes (Tier 2 + Tier 3)
    preflight_grade TEXT,                -- 'pass' | 'warn' | 'fail'
    preflight_json  TEXT,                -- JSON: {checks: [...], summary, est_*}

    -- Pipeline failure (Tier 4)
    failure_stage   TEXT,                -- e.g. '02_embed', '00b_gen_da3'
    failure_reason  TEXT,                -- user-friendly explanation

    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_spaces_owner ON spaces(owner_id);

CREATE TABLE IF NOT EXISTS memberships (
    space_id        TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL REFERENCES users(id)  ON DELETE CASCADE,
    role_in_space   TEXT NOT NULL CHECK (role_in_space IN ('viewer','editor')),
    added_at        TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (space_id, user_id)
);

CREATE TABLE IF NOT EXISTS named_objects (
    id                        TEXT PRIMARY KEY,
    space_id                  TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    name                      TEXT NOT NULL,
    slug                      TEXT NOT NULL,
    prompt_embedding          BLOB NOT NULL,        -- float32 × 768
    anchor_proposal_id        INTEGER NOT NULL,
    anchor_view_id            INTEGER NOT NULL,
    anchor_bbox               TEXT NOT NULL,        -- JSON [x1,y1,x2,y2]
    created_by                TEXT NOT NULL REFERENCES users(id),
    created_at                TEXT NOT NULL DEFAULT (datetime('now')),
    cached_member_proposal_ids TEXT,                 -- JSON int array, denormalized
    UNIQUE (space_id, slug)
);
CREATE INDEX IF NOT EXISTS idx_named_space ON named_objects(space_id);
"""


# Threading: sqlite3 connections aren't thread-safe; use one per thread.
_local = threading.local()


def _connect() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def get_conn() -> sqlite3.Connection:
    """Per-thread connection. The server keeps long-lived workers; one per worker."""
    c = getattr(_local, "conn", None)
    if c is None:
        c = _connect()
        _local.conn = c
    return c


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    """Atomic transaction. Rollbacks on exception."""
    conn = get_conn()
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if not row:
        return 0
    r = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return int(r["v"] or 0)


MIGRATIONS: dict[int, str] = {
    # v1 → v2: capture_type + preflight + failure columns on spaces.
    2: """
    ALTER TABLE spaces ADD COLUMN capture_type    TEXT NOT NULL DEFAULT 'video';
    ALTER TABLE spaces ADD COLUMN preflight_grade TEXT;
    ALTER TABLE spaces ADD COLUMN preflight_json  TEXT;
    ALTER TABLE spaces ADD COLUMN failure_stage   TEXT;
    ALTER TABLE spaces ADD COLUMN failure_reason  TEXT;
    """,
    # v2 → v3: live processing progress (worker writes; UI polls).
    3: """
    ALTER TABLE spaces ADD COLUMN stage         TEXT;
    ALTER TABLE spaces ADD COLUMN stage_idx     INTEGER;
    ALTER TABLE spaces ADD COLUMN stage_total   INTEGER;
    ALTER TABLE spaces ADD COLUMN stage_pct     REAL;
    ALTER TABLE spaces ADD COLUMN stage_text    TEXT;
    """,
}


def init() -> None:
    """Create the schema if it doesn't exist; migrate if it does."""
    conn = get_conn()
    v = current_version(conn)
    if v == 0:
        # Fresh: write the full target schema in one go.
        conn.executescript(INITIAL_SCHEMA)
        conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
        print(f"[db] initialised schema v{SCHEMA_VERSION} at {config.DB_PATH}")
        return

    # Incremental migrations from v → SCHEMA_VERSION.
    while v < SCHEMA_VERSION:
        v += 1
        sql = MIGRATIONS.get(v)
        if not sql:
            raise RuntimeError(f"no migration for v{v}")
        conn.executescript(sql)
        conn.execute("INSERT INTO schema_version(version) VALUES (?)", (v,))
        print(f"[db] migrated to v{v}")


def new_id() -> str:
    """Compact id (no dashes)."""
    return uuid.uuid4().hex


# ── CLI entry: python -m app.db init ──────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "init":
        init()
    else:
        print("usage: python -m app.db init")
        sys.exit(1)
