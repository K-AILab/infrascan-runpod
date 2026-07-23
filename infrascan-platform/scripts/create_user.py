"""Create a user from the CLI.

Usage:
    python -m scripts.create_user --email you@example.com --name "You" --role admin
"""
from __future__ import annotations

import argparse
import getpass
import sys

from app.auth import create_user, get_user_by_email
from app.db import init


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--role", choices=("capturer", "manager", "admin"), default="capturer")
    ap.add_argument("--password", help="If omitted, prompt for it.")
    args = ap.parse_args()

    init()

    if get_user_by_email(args.email):
        print(f"User {args.email} already exists.", file=sys.stderr)
        sys.exit(1)

    pw = args.password or getpass.getpass("Password: ")
    if len(pw) < 8:
        print("Password must be at least 8 characters.", file=sys.stderr)
        sys.exit(1)

    uid = create_user(email=args.email, name=args.name, password=pw, role=args.role)
    print(f"Created user {args.email} ({args.role}) — id={uid}")


if __name__ == "__main__":
    main()
