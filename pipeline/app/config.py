"""Runtime configuration. Single source of truth, read once at startup."""
from __future__ import annotations

import os
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


# Storage roots
DB_PATH = Path(os.environ.get("INFRASCAN_DB_PATH", "./data/infrascan.db")).resolve()
DATA_ROOT = Path(os.environ.get("INFRASCAN_DATA_ROOT", "./data")).resolve()
OUT_ROOT = Path(os.environ.get("INFRASCAN_OUT_ROOT", "./out")).resolve()

# Auth
SECRET_KEY = os.environ.get("INFRASCAN_SECRET_KEY", "dev-only-do-not-use-in-prod")
COOKIE_DOMAIN = os.environ.get("INFRASCAN_COOKIE_DOMAIN", "") or None
SESSION_TTL_DAYS = int(os.environ.get("INFRASCAN_SESSION_TTL_DAYS", "30"))
COOKIE_NAME = "infrascan_session"
COOKIE_SECURE = _bool("INFRASCAN_COOKIE_SECURE", default=not COOKIE_DOMAIN is None)

# Pipeline (only used when running batch jobs)
PIPELINE_MODELS = Path(os.environ.get("INFRASCAN_PIPELINE_MODELS", "./external")).resolve()


def ensure_dirs() -> None:
    """Create the data + out roots if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def is_production() -> bool:
    return SECRET_KEY != "dev-only-do-not-use-in-prod"
