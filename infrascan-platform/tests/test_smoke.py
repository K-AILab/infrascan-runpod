"""Smoke test: the FastAPI app boots and `/healthz` answers."""
from __future__ import annotations

import os
import pytest


def test_healthz(monkeypatch, tmp_path):
    monkeypatch.setenv("INFRASCAN_DB_PATH", str(tmp_path / "smoke.db"))
    monkeypatch.setenv("INFRASCAN_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("INFRASCAN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("INFRASCAN_SECRET_KEY", "test-key")

    # Reload config + app fresh
    import importlib
    import app.config as cfg
    importlib.reload(cfg)
    import app.main as mainmod
    importlib.reload(mainmod)

    from fastapi.testclient import TestClient
    client = TestClient(mainmod.app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}
