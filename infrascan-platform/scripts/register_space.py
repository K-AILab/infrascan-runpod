"""Register an existing on-disk space into the database.

Useful when the pipeline has produced data/<slug>/ + out/<slug>/ externally
and you want the platform to know about it.

Usage:
    python -m scripts.register_space --slug icc1 \\
        --title "ICC Office — Scan 1" \\
        --owner-email you@example.com \\
        --status ready \\
        --n-views 25056 --n-scanpoints 696
"""
from __future__ import annotations

import argparse
import sys

from app.auth import get_user_by_email
from app.db import init, get_conn, tx
from app.spaces import by_slug, create_space, update_status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--owner-email", required=True)
    ap.add_argument("--status", choices=("uploading", "processing", "ready", "failed"), default="ready")
    ap.add_argument("--n-views", type=int, default=0)
    ap.add_argument("--n-scanpoints", type=int, default=0)
    ap.add_argument("--y-up", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
    args = ap.parse_args()

    init()
    owner = get_user_by_email(args.owner_email)
    if not owner:
        print(f"Owner not found: {args.owner_email}", file=sys.stderr)
        sys.exit(1)

    if by_slug(args.slug):
        update_status(args.slug, args.status, n_views=args.n_views, n_scanpoints=args.n_scanpoints)
        print(f"Updated existing space '{args.slug}'.")
        return

    create_space(slug=args.slug, title=args.title, owner_id=owner["id"], status=args.status, y_up=args.y_up)
    update_status(args.slug, args.status, n_views=args.n_views, n_scanpoints=args.n_scanpoints)
    print(f"Registered space '{args.slug}' for {owner['email']}.")


if __name__ == "__main__":
    main()
