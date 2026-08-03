"""Authentication: password hashing + session cookies.

- Passwords: argon2 (fast, memory-hard, modern).
- Sessions: opaque UUID stored in `sessions` table; cookie carries the id only,
  signed with itsdangerous so the value can't be forged into a different id.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import HTTPException, Request, status
from itsdangerous import BadSignature, URLSafeSerializer

from . import config
from .db import get_conn, new_id, tx


_ph = PasswordHasher()
_signer = URLSafeSerializer(config.SECRET_KEY, salt="infrascan-session")


# ── Password ──────────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    if not plain or len(plain) < 8:
        raise ValueError("password must be at least 8 characters")
    return _ph.hash(plain)


def verify_password(stored_hash: str, plain: str) -> bool:
    try:
        _ph.verify(stored_hash, plain)
        return True
    except VerifyMismatchError:
        return False


# ── Users ─────────────────────────────────────────────────────────────────
def create_user(email: str, name: str, password: str, role: str = "capturer") -> str:
    if role not in ("capturer", "manager", "admin"):
        raise ValueError(f"invalid role: {role}")
    uid = new_id()
    with tx() as conn:
        conn.execute(
            "INSERT INTO users(id, email, name, password_hash, role) VALUES (?,?,?,?,?)",
            (uid, email.strip().lower(), name.strip(), hash_password(password), role),
        )
    return uid


def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    return get_conn().execute(
        "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
    ).fetchone()


def get_user(user_id: str) -> Optional[sqlite3.Row]:
    return get_conn().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


# ── Sessions ──────────────────────────────────────────────────────────────
def create_session(user_id: str) -> tuple[str, dt.datetime]:
    """Create a session row. Returns (session_id, expires_at)."""
    sid = new_id()
    expires = dt.datetime.utcnow() + dt.timedelta(days=config.SESSION_TTL_DAYS)
    with tx() as conn:
        conn.execute(
            "INSERT INTO sessions(id, user_id, expires_at) VALUES (?,?,?)",
            (sid, user_id, expires.isoformat(timespec="seconds")),
        )
    return sid, expires


def revoke_session(session_id: str) -> None:
    with tx() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def cookie_payload(session_id: str) -> str:
    """Return a signed cookie value that resolves back to session_id."""
    return _signer.dumps({"sid": session_id})


def parse_cookie(value: str) -> Optional[str]:
    """Reverse of cookie_payload. Returns session_id or None if tampered."""
    try:
        data = _signer.loads(value)
        return data.get("sid")
    except (BadSignature, ValueError, TypeError):
        return None


def session_user(request: Request) -> Optional[sqlite3.Row]:
    """Return the logged-in user row or None. No exception."""
    raw = request.cookies.get(config.COOKIE_NAME)
    if not raw:
        return None
    sid = parse_cookie(raw)
    if not sid:
        return None
    row = get_conn().execute(
        """SELECT u.* FROM sessions s
              JOIN users u ON u.id = s.user_id
              WHERE s.id = ? AND s.expires_at > datetime('now')""",
        (sid,),
    ).fetchone()
    if row:
        # touch
        get_conn().execute(
            "UPDATE sessions SET last_seen_at = datetime('now') WHERE id = ?", (sid,)
        )
    return row


def require_user(request: Request) -> sqlite3.Row:
    """FastAPI dependency — 401s if not logged in."""
    user = session_user(request)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not signed in")
    return user


def require_admin(request: Request) -> sqlite3.Row:
    user = require_user(request)
    if user["role"] != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    return user
