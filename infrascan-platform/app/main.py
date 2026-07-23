"""User-facing FastAPI app.

Mounts:
  /                       login or my-spaces (depending on auth)
  /login                  sign-in
  /signup                 (invite-only in v1; renders 404 unless enabled)
  /logout                 POST
  /spaces                 list user's spaces
  /spaces/<slug>          space detail
  /spaces/<slug>/viewer   3D walk-through
  /search                 cross-space visual search
  /upload                 capturer upload flow
  /api/spaces             JSON list
  /api/spaces/<slug>/query  visual search
  /api/spaces/<slug>/save   save a named object
  /shared/*               design tokens
  /static/*               app static assets

Run:  uvicorn app.main:app --port 8050
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config, db
from .auth import session_user
from .routes import auth as auth_routes
from .routes import spaces as spaces_routes
from .routes import search as search_routes
from .routes import upload as upload_routes
from .routes import upload_chunked as upload_chunked_routes
from .routes import find_similar as find_similar_routes


REPO = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(REPO / "ui" / "templates"))


def make_app() -> FastAPI:
    config.ensure_dirs()
    db.init()

    app = FastAPI(title="InfraScan", version="0.1.0")

    # Static. Wrap so /static/* sends no-cache headers — Cloudflare's default
    # 4 h edge cache was serving stale viewer.js builds.
    from starlette.types import ASGIApp, Receive, Scope, Send
    class _NoCacheStatic:
        def __init__(self, app: ASGIApp): self.app = app
        async def __call__(self, scope: Scope, receive: Receive, send: Send):
            async def _send(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers = [(k, v) for (k, v) in headers if k.lower() != b"cache-control"]
                    headers.append((b"cache-control", b"no-store, max-age=0"))
                    message["headers"] = headers
                await send(message)
            await self.app(scope, receive, _send)
    app.mount("/shared", _NoCacheStatic(StaticFiles(directory=str(REPO / "shared"))), name="shared")
    app.mount("/static", _NoCacheStatic(StaticFiles(directory=str(REPO / "ui" / "static"))), name="static")
    # ui/legacy-viewer is a git submodule of K-AILab/3d-object-tagging
    # (branch handoff/infrascan-platform). Viewer source files live at
    # ui/viewer/ inside that repo, so mount that subdirectory — keeps every
    # existing /legacy-viewer/<file>.js URL working with no template churn.
    app.mount("/legacy-viewer", _NoCacheStatic(StaticFiles(directory=str(REPO / "ui" / "legacy-viewer" / "ui" / "viewer"))), name="legacy_viewer")

    # Routes
    app.include_router(auth_routes.router)
    app.include_router(spaces_routes.router)
    app.include_router(search_routes.router)
    app.include_router(upload_routes.router)
    app.include_router(upload_chunked_routes.router)
    app.include_router(find_similar_routes.router)

    @app.get("/")
    async def root(request: Request):
        user = session_user(request)
        if not user:
            return RedirectResponse("/login", status_code=302)
        return RedirectResponse("/spaces", status_code=302)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    return app


app = make_app()
