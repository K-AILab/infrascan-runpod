"""Admin FastAPI app — internal surface.

Mounts:
  /                       gallery — every space + every experiment
  /users                  user management
  /spaces                 cross-user space management
  /logs                   recent server logs (tail)
  /shared/*               design tokens
  /static/*               admin static assets

Run:  uvicorn app.main_admin:app --port 8051
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request, Depends
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config, db
from .auth import session_user, require_admin
from .routes import admin as admin_routes
from .routes import auth as auth_routes


REPO = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(REPO / "ui" / "templates"))


def make_admin_app() -> FastAPI:
    config.ensure_dirs()
    db.init()

    app = FastAPI(title="InfraScan Admin", version="0.1.0")

    app.mount("/shared", StaticFiles(directory=str(REPO / "shared")), name="shared")
    app.mount("/static", StaticFiles(directory=str(REPO / "ui" / "static")), name="static")

    app.include_router(auth_routes.router)  # login/logout shared
    app.include_router(admin_routes.router)

    @app.get("/")
    async def root(request: Request):
        user = session_user(request)
        if not user:
            return RedirectResponse("/login?next=/", status_code=302)
        if user["role"] != "admin":
            return RedirectResponse("https://app.infrascan-ai.com/", status_code=302)
        return RedirectResponse("/gallery", status_code=302)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    return app


app = make_admin_app()
