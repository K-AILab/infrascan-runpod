"""Smoke tests for auth + sessions."""
from __future__ import annotations

import os
import tempfile

import pytest

from app import config


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch, tmp_path):
    """Point config at a fresh sqlite DB for each test."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(config, "OUT_ROOT", tmp_path / "out")
    from app import db, auth
    # Reset per-thread cache
    if hasattr(db._local, "conn"):
        try:
            db._local.conn.close()
        except Exception:
            pass
        delattr(db._local, "conn")
    db.init()
    yield
    if hasattr(db._local, "conn"):
        db._local.conn.close()
        delattr(db._local, "conn")


def test_create_and_login_user():
    from app.auth import create_user, get_user_by_email, verify_password

    uid = create_user("ada@example.com", "Ada", "correct-horse-battery-staple", role="admin")
    assert uid

    row = get_user_by_email("ada@example.com")
    assert row is not None
    assert row["role"] == "admin"
    assert verify_password(row["password_hash"], "correct-horse-battery-staple")
    assert not verify_password(row["password_hash"], "wrong")


def test_session_roundtrip():
    from app.auth import create_user, create_session, cookie_payload, parse_cookie

    uid = create_user("alan@example.com", "Alan", "good-enough-password")
    sid, _ = create_session(uid)
    cookie = cookie_payload(sid)
    assert parse_cookie(cookie) == sid
    # Tampered cookie returns None
    assert parse_cookie(cookie[:-1] + "X") is None
