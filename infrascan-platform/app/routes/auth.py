"""Auth routes — login, logout. (Sign-up is invite-only in v1.)"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .. import config
from ..auth import (
    cookie_payload,
    create_session,
    get_user_by_email,
    parse_cookie,
    revoke_session,
    session_user,
    verify_password,
)


REPO = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(REPO / "ui" / "templates"))

router = APIRouter(tags=["auth"])


@router.get("/login", response_class=HTMLResponse)
async def login_get(request: Request, next: str = "/", error: str | None = None):
    if session_user(request):
        return RedirectResponse(next, status_code=302)
    return templates.TemplateResponse(request, "login.html", { "next": next, "error": error})


@router.post("/login")
async def login_post(
    request: Request,
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
):
    user = get_user_by_email(email)
    if not user or not verify_password(user["password_hash"], password):
        # Don't leak which one failed.
        return templates.TemplateResponse(request, "login.html", { "next": next, "error": "Incorrect email or password."},
            status_code=401,
        )
    sid, expires = create_session(user["id"])
    resp = RedirectResponse(next, status_code=303)
    resp.set_cookie(
        config.COOKIE_NAME,
        cookie_payload(sid),
        httponly=True,
        secure=config.COOKIE_SECURE,
        samesite="lax",
        domain=config.COOKIE_DOMAIN,
        expires=int(expires.timestamp()),
        path="/",
    )
    return resp


@router.post("/logout")
async def logout(request: Request):
    raw = request.cookies.get(config.COOKIE_NAME)
    if raw:
        sid = parse_cookie(raw)
        if sid:
            revoke_session(sid)
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(config.COOKIE_NAME, domain=config.COOKIE_DOMAIN, path="/")
    return resp
