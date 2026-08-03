"""Single source of truth for per-space paths.

Reads `spaces.json` at the repo root and exposes a clean accessor used by every
pipeline script. Replaces the old hard-coded SPACES dict.

Schema (`spaces.json`):
{
  "spaces": {
    "<name>": {
      "title":        "Human-readable name",
      "data_root":    "data/<name>",
      "out_dir":      "out/fastsam_<name>",
      "y_up":         true | false,
      "n_views":      <int>,
      "n_scanpoints": <int>
    }
  }
}

Each space's `data_root` is expected to contain:
    views/                        # 504x504 perspective JPGs
    cameras.json                  # one entry per view (id, pos, R, pano, xy)
    intrinsics.json               # shared K matrix (optional but recommended)
    depth/frame_<id>.npz          # DA3 outputs (created by 00b_gen_da3.py)
    pointcloud.ply                # the captured PCD (only used by gen_topdown.py)
"""
from __future__ import annotations

import json
from pathlib import Path

# Repo root = the directory two levels above this file's parent (pipeline/ → repo)
REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "spaces.json"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"spaces": {}}
    return json.loads(CONFIG_PATH.read_text())


def space_choices() -> list[str]:
    """All registered space names (used by argparse `choices=`)."""
    return sorted(_load_config().get("spaces", {}).keys())


def space(name: str) -> dict:
    """Return resolved Paths for one space.

    Raises KeyError if the space isn't registered. Run
    `scripts/add_space.sh <name> <source>` first to register a new one.
    """
    cfg = _load_config()
    spaces = cfg.get("spaces", {})
    if name not in spaces:
        raise KeyError(
            f"Space '{name}' not in spaces.json. "
            f"Registered: {sorted(spaces.keys())}. "
            f"To register: bash scripts/add_space.sh {name} <source_dir>."
        )
    s = spaces[name]
    data_root = REPO / s["data_root"]
    return {
        "name":         name,
        "title":        s.get("title", name),
        "data_root":    data_root,
        "views":        data_root / "views",
        "cameras":      data_root / "cameras.json",
        "intrinsics":   data_root / "intrinsics.json",
        "da3":          data_root / "depth",
        "pointcloud":   data_root / "pointcloud.ply",
        "out_dir":      REPO / s["out_dir"],
        "y_up":         bool(s.get("y_up", False)),
        "n_views":      int(s.get("n_views", 0)),
        "n_scanpoints": int(s.get("n_scanpoints", 0)),
    }


# Back-compat alias for scripts using the old name
SPACE_CHOICES = space_choices()


def out_dir(name: str) -> Path:
    """Convenience: just the pipeline output directory for one space."""
    return space(name)["out_dir"]
