#!/usr/bin/env python
"""Create the ui/_spaces/<space>/ directory the web viewer serves a space from.

run_space_pipeline.py writes a scene graph into ui/_spaces/<space>/scene_graph.json,
but the viewer needs three more things in that directory before the space is
usable, and it fails quietly without them:

  index.html          the per-space landing page. ui/index.html sends Dev mode to
                      /<space>/, which is this file. Without it that link 404s.
  Data_/              the captured assets the viewer loads directly - the point
                      cloud (downsampled_web.ply), cameras.json, and the panorama
                      and frame images.
  topdown/            topdown.png + bounds.json, the floor plan behind the
                      top-down mode's canvas overlay. Missing, the mode opens to
                      a broken image with no error.

Data_ and topdown/ are captured/derived assets, not source, so they are linked
from wherever they already exist rather than copied. topdown.png and bounds.json
are functions of the POINT CLOUD alone, so any space sharing this space's cloud
has a valid pair. Only those two are linked - the scene_graph_*.png files beside
them are renders of that other space's graph and would draw the wrong boxes.

Usage:
  python pipeline9/provision_ui_space.py --space shinhan_owlv2_pointcloud_scenegraph \\
      --from-space shinhan_space_p4

--from-space is the space to borrow Data_ and topdown/ from; it must be a space
whose capture is the same one this space's graph was reprojected onto. Pass
--data-root / --topdown-dir instead to point at directories directly.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPACES_DIR = REPO / "ui" / "_spaces"
TEMPLATE = REPO / "ui" / "space_template.html"

DEFAULT_DESCRIPTION = (
    "OWLv2 detections lifted from the Gaussian Splat and re-projected onto this "
    "space's captured point cloud, then labelled and reconciled. Nodes and "
    "relationship edges come from the same scene graph."
)


def _link(target: Path, link: Path):
    """Replace link with a symlink to target, resolving target through its own links."""
    real = Path(os.path.realpath(target))
    if not real.exists():
        sys.exit(f"error: {target} does not exist (resolved to {real})")
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(real)
    return real


def main():
    ap = argparse.ArgumentParser(
        description="Provision a ui/_spaces/<space> directory for the web viewer.")
    ap.add_argument("--space", required=True,
                    help="space name, matching its key in spaces.json")
    ap.add_argument("--from-space", default=None,
                    help="space to borrow Data_ and topdown/ from — must share this "
                         "space's capture, since the floor plan and point cloud are "
                         "the same assets")
    ap.add_argument("--data-root", default=None,
                    help="Data_ directory to link, instead of --from-space's")
    ap.add_argument("--topdown-dir", default=None,
                    help="topdown directory to link topdown.png and bounds.json "
                         "from, instead of --from-space's")
    ap.add_argument("--title", default=None,
                    help="page heading (default: the title in spaces.json, else the "
                         "space name)")
    ap.add_argument("--description", default=DEFAULT_DESCRIPTION)
    args = ap.parse_args()

    d = SPACES_DIR / args.space
    d.mkdir(parents=True, exist_ok=True)

    src = SPACES_DIR / args.from_space if args.from_space else None
    data_root = Path(args.data_root) if args.data_root else (src / "Data_" if src else None)
    topdown = Path(args.topdown_dir) if args.topdown_dir else (src / "topdown" if src else None)
    if data_root is None or topdown is None:
        sys.exit("error: pass --from-space, or both --data-root and --topdown-dir")

    _link(data_root, d / "Data_")
    print(f"[provision] Data_    -> {os.path.realpath(data_root)}")

    td = d / "topdown"
    td.mkdir(exist_ok=True)
    for f in ("topdown.png", "bounds.json"):
        _link(topdown / f, td / f)
    print(f"[provision] topdown/ -> {os.path.realpath(topdown)} "
          f"(topdown.png, bounds.json only)")

    title = args.title
    if title is None:
        cfg = REPO / "spaces.json"
        if cfg.exists():
            title = json.loads(cfg.read_text()).get("spaces", {}) \
                        .get(args.space, {}).get("title")
        title = title or args.space
    html = (TEMPLATE.read_text()
            .replace("__SPACE_TITLE__", title)
            .replace("__SPACE_NAME__", args.space)
            .replace("__SPACE_DESCRIPTION__", args.description))
    (d / "index.html").write_text(html)
    print(f"[provision] index.html written (title: {title!r})")

    sg = d / "scene_graph.json"
    if not sg.exists():
        print(f"[provision] NOTE: no scene_graph.json here yet — "
              f"run run_space_pipeline.py to write one")
    print(f"\n[provision] {args.space} ready:  /{args.space}/viewer/?mode=3d&normal=1")


if __name__ == "__main__":
    main()
