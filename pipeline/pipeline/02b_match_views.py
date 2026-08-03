#!/usr/bin/env python3
"""
Offline step 2b: Within-scanpoint AND cross-scanpoint-neighbor identity
matching with SuperPoint + LightGlue.

Two passes, sharing the same SuperPoint feature cache:

1. Within-scanpoint:
   For each scanpoint, for each adjacent yaw pair within each pitch
   level, run SuperPoint+LightGlue. A proposal in view A is declared
   the same physical object as a proposal in view B when
   ≥ MIN_KEYPOINT_MATCHES of A's bbox's keypoints land inside B's bbox.

2. Cross-scanpoint (NEIGHBORS only):
   For each scanpoint pair within NEIGHBOR_RADIUS_M of each other (KD-
   tree), and for each view pair whose forward direction cosines exceed
   YAW_COS_MIN, run the same keypoint test. This catches:
     - Sequential prev/next scanpoint pairs (same as 03b but stronger)
     - Doubling-back / loop revisits (NOT catchable by within-sp logic)
     - Parallel-rail scans through a corridor

Union-find merges all identity links. Output is a dense int32
`object_ids.npy` (shape (N,)) where row i ↔ embeddings.npy[i].

Usage:
    python sandbox/offline/02b_match_views.py --proposer fastsam --space v2

Input:  <out-dir>/metadata.json    (from 02_embed.py)
        cameras.json               (R + pos per view, for forward vectors + KD-tree)
Output: <out-dir>/object_ids.npy

Cost on a 3090 for v2 fastsam (267k rows, 7,812 views, 217 scanpoints):
    ~13-16 min total (within-sp ~6 min + cross-sp ~8 min).
"""
import argparse
import json
import re
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
import torch

_sys_path = str(Path(__file__).parent)
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)
from _paths import space, space_choices  # noqa: E402


# ── Config ──
MIN_KEYPOINT_MATCHES = 2          # ≥ this many partner kpts inside the candidate bbox → declare a match
MAX_KEYPOINTS        = 2048       # SuperPoint cap per view
YAWS                 = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
PITCH_BUCKETS        = ["000"]   # horizontal pitch only — matches reference behavior

# Cross-scanpoint neighbor pairing (calibrated to v2 data — sp spacing
# is ~22 cm median, with 41 doubling-back loops at 16-30 cm spatial dist)
NEIGHBOR_RADIUS_M    = 0.60       # max spatial distance between paired scanpoints
MAX_NEIGHBORS        = 6          # cap per scanpoint to bound runtime
YAW_COS_MIN          = 0.70       # forward-vector cosine for view-pair pre-filter (~45°)

# SuperPoint feature cache: LRU eviction. Each cached entry is roughly
# 2048 kpts × (2 + 256) floats ≈ 2 MB. 200 views → ~0.4 GB GPU memory.
SP_CACHE_MAX         = 200


# ── Union-Find ──
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def parse_pano(name: str):
    m = re.search(r"(\d{6})_pz(\d{3})_y(\d{3})", name)
    if not m:
        return None
    return int(m.group(1)), m.group(2), m.group(3)


def kpts_in_bbox(kpts: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return ((kpts[:, 0] >= x1) & (kpts[:, 0] <= x2) &
            (kpts[:, 1] >= y1) & (kpts[:, 1] <= y2))


def match_view_pair(meta, uf, rows_a, rows_b, feats_a, feats_b,
                    matcher, min_matches) -> int:
    """Run LightGlue on cached features and union-find merge proposals
    based on keypoint-in-bbox correspondence. Returns # of new unions."""
    with torch.no_grad():
        out = matcher({"image0": feats_a, "image1": feats_b})
    k_a = feats_a["keypoints"][0].cpu().numpy()
    k_b = feats_b["keypoints"][0].cpu().numpy()
    idx = out["matches"][0].cpu().numpy()
    if idx.shape[0] == 0:
        return 0
    pts_a = k_a[idx[:, 0]]
    pts_b = k_b[idx[:, 1]]

    matched_b = set()
    n_new = 0
    for ra in rows_a:
        inside = kpts_in_bbox(pts_a, meta[ra]["bbox"])
        if inside.sum() < min_matches:
            continue
        partners = pts_b[inside]
        best_rb, best_count = None, 0
        for rb in rows_b:
            if rb in matched_b:
                continue
            cnt = int(kpts_in_bbox(partners, meta[rb]["bbox"]).sum())
            if cnt > best_count:
                best_count, best_rb = cnt, rb
        if best_rb is not None and best_count >= min_matches:
            uf.union(ra, best_rb)
            matched_b.add(best_rb)
            n_new += 1
    return n_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, choices=space_choices(),
                        help="Space (must be registered in spaces.json).")
    parser.add_argument("--out-dir", default=None,
                        help="Pipeline output dir override (default: spaces.json out_dir).")
    parser.add_argument("--min-matches", type=int, default=MIN_KEYPOINT_MATCHES,
                        help=f"Min shared keypoints to merge (default: {MIN_KEYPOINT_MATCHES})")
    parser.add_argument("--neighbor-radius-m", type=float, default=NEIGHBOR_RADIUS_M,
                        help=f"Cross-scanpoint pairing radius (default: {NEIGHBOR_RADIUS_M})")
    parser.add_argument("--max-neighbors", type=int, default=MAX_NEIGHBORS,
                        help=f"Cap neighbors per scanpoint (default: {MAX_NEIGHBORS})")
    # Cross-sp neighbor matching is DEFAULT OFF — it over-merged in
    # practice (route walks through similar-looking corridors). Use
    # --cross-sp to opt in if you've tuned NEIGHBOR_RADIUS_M for your data.
    parser.add_argument("--cross-sp", dest="no_cross_sp", action="store_false",
                        help="Enable the cross-scanpoint neighbor pass (default: OFF)")
    parser.add_argument("--no-cross-sp", dest="no_cross_sp", action="store_true",
                        help="(default) Skip cross-scanpoint neighbor pass — within-sp only")
    parser.set_defaults(no_cross_sp=True)
    args = parser.parse_args()
    args.proposer = "fastsam"   # fastsam-only build

    ROOT = Path(__file__).parent
    out_dir = Path(args.out_dir) if args.out_dir else space(args.space)["out_dir"]
    META_FILE = out_dir / "metadata.json"
    OUT_FILE  = out_dir / "object_ids.npy"
    views_dir = space(args.space)["views"]
    cameras_path = space(args.space)["cameras"]

    if not META_FILE.exists():
        print(f"[match] ERROR: {META_FILE} not found — run 02_embed.py first")
        sys.exit(1)

    print(f"[match] space={args.space} out_dir={out_dir}")
    print(f"[match] views_dir={views_dir}")
    print(f"[match] min_matches={args.min_matches}")
    print(f"[match] neighbor_radius={args.neighbor_radius_m} m  max_nbrs={args.max_neighbors}  "
          f"cross_sp={'OFF' if args.no_cross_sp else 'ON'}")

    meta = json.loads(META_FILE.read_text())
    print(f"[match] {len(meta)} proposal rows loaded")

    # Bucket rows by (scanpoint, pitch, yaw)
    by_spy: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for ri, m in enumerate(meta):
        parsed = parse_pano(Path(m["pano"]).name)
        if parsed is None:
            continue
        sp, pz, yw = parsed
        by_spy[sp][pz][yw].append(ri)

    n_scanpoints = len(by_spy)
    print(f"[match] {n_scanpoints} scanpoints to process")

    # Load camera poses for forward vectors + scanpoint positions
    cams = json.loads(Path(cameras_path).read_text())
    fwd_by_view: dict = {}     # (sp, pz, yw) → unit world-frame forward
    sp_pos_map: dict = {}      # sp → world position
    for c in cams:
        parsed = parse_pano(Path(c["pano"]).name)
        if parsed is None:
            continue
        sp, pz, yw = parsed
        R = np.array(c["R"], dtype=np.float32)
        # OpenCV: +Z is forward in camera frame; world-frame forward = R @ [0,0,1]
        fwd = R @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        fwd_by_view[(sp, pz, yw)] = fwd / (np.linalg.norm(fwd) + 1e-12)
        if sp not in sp_pos_map:
            sp_pos_map[sp] = np.asarray(c["pos"], dtype=np.float32)

    # Load LightGlue + SuperPoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[match] loading SuperPoint + LightGlue on {device}")
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image
    extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(device)
    matcher   = LightGlue(features="superpoint").eval().to(device)

    # LRU SuperPoint feature cache shared across both passes
    sp_cache: "OrderedDict[str, dict]" = OrderedDict()

    def get_feats(pano_path: Path):
        key = pano_path.name
        if key in sp_cache:
            sp_cache.move_to_end(key)
            return sp_cache[key]
        img = load_image(str(pano_path)).to(device)
        with torch.no_grad():
            feats = extractor.extract(img)
        sp_cache[key] = feats
        while len(sp_cache) > SP_CACHE_MAX:
            sp_cache.popitem(last=False)
        return feats

    uf = UnionFind(len(meta))
    n_unions_within = 0
    n_unions_cross  = 0
    n_pairs_within  = 0
    n_pairs_cross   = 0

    # ── PASS 1: Within-scanpoint, adjacent-yaw within each pitch ──────────
    print("\n[match] === Pass 1: within-scanpoint adjacent-yaw matching ===")
    for spi, (sp, by_pz) in enumerate(sorted(by_spy.items())):
        for pz in PITCH_BUCKETS:
            by_yaw = by_pz.get(pz, {})
            if not by_yaw:
                continue
            for i, yaw_a_int in enumerate(YAWS):
                yaw_b_int = YAWS[(i + 1) % len(YAWS)]
                yaw_a = f"{yaw_a_int:03d}"
                yaw_b = f"{yaw_b_int:03d}"
                rows_a = by_yaw.get(yaw_a, [])
                rows_b = by_yaw.get(yaw_b, [])
                if not rows_a or not rows_b:
                    continue
                pano_a = views_dir / f"{sp:06d}_pz{pz}_y{yaw_a}_normal.jpg"
                pano_b = views_dir / f"{sp:06d}_pz{pz}_y{yaw_b}_normal.jpg"
                if not pano_a.exists() or not pano_b.exists():
                    continue
                feats_a = get_feats(pano_a)
                feats_b = get_feats(pano_b)
                n_new = match_view_pair(meta, uf, rows_a, rows_b,
                                        feats_a, feats_b, matcher, args.min_matches)
                n_unions_within += n_new
                n_pairs_within  += 1

        if (spi + 1) % 20 == 0 or spi + 1 == n_scanpoints:
            print(f"[match] within-sp {spi+1}/{n_scanpoints} · pairs {n_pairs_within} · unions {n_unions_within}")

    if args.no_cross_sp:
        # ── Resolve and write
        raw = np.array([uf.find(i) for i in range(len(meta))], dtype=np.int64)
        _, object_ids = np.unique(raw, return_inverse=True)
        object_ids = object_ids.astype(np.int32)
        n_unique = int(object_ids.max()) + 1 if len(object_ids) else 0
        np.save(str(OUT_FILE), object_ids)
        print(f"\n[match] {len(meta)} rows → {n_unique} unique objects (within-sp only)")
        print(f"[match] Wrote {OUT_FILE}")
        return

    # ── PASS 2: Cross-scanpoint NEIGHBOR matching ─────────────────────────
    print("\n[match] === Pass 2: cross-scanpoint neighbor matching ===")
    from scipy.spatial import cKDTree
    sp_ids = sorted(by_spy.keys())
    sp_pts = np.array([sp_pos_map[sp] for sp in sp_ids])
    tree = cKDTree(sp_pts)

    # Build neighbor map (each pair only listed once: sp_b > sp_a)
    neighbor_pairs: list = []
    for i, sp_a in enumerate(sp_ids):
        dists, jdxs = tree.query(sp_pts[i],
                                 k=args.max_neighbors + 1,
                                 distance_upper_bound=args.neighbor_radius_m)
        for jd, j in zip(dists, jdxs):
            if j == i or j >= len(sp_ids):
                continue
            if jd > args.neighbor_radius_m:
                continue
            sp_b = sp_ids[j]
            if sp_b > sp_a:
                neighbor_pairs.append((sp_a, sp_b, float(jd)))

    print(f"[match] {len(neighbor_pairs)} cross-scanpoint pairs within "
          f"{args.neighbor_radius_m} m (cap {args.max_neighbors}/sp)")

    last_progress = 0
    for pi, (sp_a, sp_b, dist) in enumerate(neighbor_pairs):
        by_pz_a = by_spy[sp_a]
        by_pz_b = by_spy[sp_b]
        for pz in PITCH_BUCKETS:
            yaws_a = by_pz_a.get(pz, {})
            yaws_b = by_pz_b.get(pz, {})
            if not yaws_a or not yaws_b:
                continue

            # For each yaw in A, pick the best forward-aligned yaw in B
            for yaw_a in yaws_a:
                yaw_a_int = int(yaw_a)
                rows_a = yaws_a[yaw_a]
                if not rows_a:
                    continue
                fwd_a = fwd_by_view.get((sp_a, pz, yaw_a))
                if fwd_a is None:
                    continue
                best_yaw_b, best_cos = None, YAW_COS_MIN
                for yaw_b in yaws_b:
                    fwd_b = fwd_by_view.get((sp_b, pz, yaw_b))
                    if fwd_b is None:
                        continue
                    cos = float(fwd_a @ fwd_b)
                    if cos > best_cos:
                        best_cos, best_yaw_b = cos, yaw_b
                if best_yaw_b is None:
                    continue
                rows_b = yaws_b[best_yaw_b]
                if not rows_b:
                    continue

                pano_a = views_dir / f"{sp_a:06d}_pz{pz}_y{yaw_a}_normal.jpg"
                pano_b = views_dir / f"{sp_b:06d}_pz{pz}_y{best_yaw_b}_normal.jpg"
                if not pano_a.exists() or not pano_b.exists():
                    continue
                feats_a = get_feats(pano_a)
                feats_b = get_feats(pano_b)
                n_new = match_view_pair(meta, uf, rows_a, rows_b,
                                        feats_a, feats_b, matcher, args.min_matches)
                n_unions_cross += n_new
                n_pairs_cross  += 1

        if pi - last_progress >= 50 or pi + 1 == len(neighbor_pairs):
            last_progress = pi
            print(f"[match] cross-sp pair {pi+1}/{len(neighbor_pairs)} · "
                  f"view-pairs {n_pairs_cross} · unions {n_unions_cross}")

    # ── Resolve and write
    raw = np.array([uf.find(i) for i in range(len(meta))], dtype=np.int64)
    _, object_ids = np.unique(raw, return_inverse=True)
    object_ids = object_ids.astype(np.int32)
    n_unique = int(object_ids.max()) + 1 if len(object_ids) else 0

    np.save(str(OUT_FILE), object_ids)
    print(f"\n[match] Summary:")
    print(f"  within-sp pairs  {n_pairs_within:>6}  unions {n_unions_within:>6}")
    print(f"  cross-sp  pairs  {n_pairs_cross:>6}  unions {n_unions_cross:>6}")
    print(f"  {len(meta)} rows → {n_unique} unique objects (within + neighbor cross-sp)")
    print(f"[match] Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
