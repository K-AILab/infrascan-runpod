"""Generate preflight_frames + preflight_json for a legacy-imported space.

The upload pipeline normally writes these during validation. Legacy spaces
brought in by bootstrap_dev.sh --with-legacy-icc skip validation entirely,
so the space-detail page has no preview strip. This script backfills:

  data/<slug>/preflight_frames/frame_NN.jpg  (symlinks to 10 spread-out
                                              y000_pz000 views)
  spaces.preflight_json                     (minimal stub with
                                              sampled_frames + a 'legacy'
                                              grade so the UI renders the
                                              strip)

Usage:
    python -m scripts.gen_preflight_for_legacy --slug icc1 [--slug icc2 ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from app.db import get_conn, init, tx
from app.spaces import data_dir

N_SAMPLES = 10


def gen(slug: str) -> None:
    ddir = data_dir(slug)
    cams_path = ddir / "cameras.json"
    if not cams_path.exists():
        print(f"[{slug}] no cameras.json — skipping", file=sys.stderr)
        return
    cams = json.loads(cams_path.read_text())

    # Pick one y000_pz000 view per scanpoint.
    by_sp: dict[int, str] = {}
    for c in cams:
        pano = (c.get("pano") or "").lstrip("/")
        if pano.startswith("panos/"):
            pano = pano[len("panos/"):]
        m = re.match(r"(\d+)_pz(\d+)_y(\d+)", pano)
        if not m:
            continue
        sp, pz, yaw = (int(m.group(i)) for i in (1, 2, 3))
        if pz == 0 and yaw == 0 and sp not in by_sp:
            by_sp[sp] = pano
    if not by_sp:
        print(f"[{slug}] no pz000_y000 views in cameras.json — skipping", file=sys.stderr)
        return

    sp_ids = sorted(by_sp.keys())
    # Pick N_SAMPLES evenly-spaced indices.
    step = max(len(sp_ids) // N_SAMPLES, 1)
    chosen = [sp_ids[i] for i in range(0, len(sp_ids), step)][:N_SAMPLES]
    if len(chosen) < N_SAMPLES:
        chosen += sp_ids[len(chosen):N_SAMPLES]

    pf_dir = ddir / "preflight_frames"
    pf_dir.mkdir(exist_ok=True)
    sampled = []
    for i, sp in enumerate(chosen):
        src = ddir / "views" / by_sp[sp]
        if not src.exists():
            print(f"[{slug}] view missing: {src}", file=sys.stderr)
            continue
        dst = pf_dir / f"frame_{i:02d}.jpg"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        sampled.append(f"preflight_frames/frame_{i:02d}.jpg")

    pj = {
        "grade": "legacy",
        "checks": [],
        "summary": f"Legacy import — {len(sp_ids)} scan-points, {len(cams)} views",
        "est_scanpoints": len(sp_ids),
        "est_views": len(cams),
        "sampled_frames": sampled,
    }
    with tx() as conn:
        conn.execute(
            "UPDATE spaces SET preflight_json = ? WHERE slug = ?",
            (json.dumps(pj), slug),
        )
    print(f"[{slug}] {len(sampled)} preflight frames + preflight_json written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", action="append", required=True,
                    help="repeat for each slug")
    args = ap.parse_args()
    init()
    for slug in args.slug:
        gen(slug)


if __name__ == "__main__":
    main()
