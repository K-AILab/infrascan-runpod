"""Single CLI entry-point: `infrascan` (declared in pyproject.toml)."""
from __future__ import annotations

import sys


COMMANDS = {
    "init-db":      ("app.db",                  "init"),
    "create-user":  ("scripts.create_user",     "main"),
    "register-space": ("scripts.register_space", "main"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: infrascan <command> [args]\n")
        print("Commands:")
        for k in COMMANDS:
            print(f"  {k}")
        return
    cmd = sys.argv[1]
    sys.argv = sys.argv[:1] + sys.argv[2:]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(2)
    mod_name, func = COMMANDS[cmd]
    import importlib
    mod = importlib.import_module(mod_name)
    getattr(mod, func)()


if __name__ == "__main__":
    main()
