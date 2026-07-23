#!/usr/bin/env python3
"""
Offline step 3b: Cross-scanpoint merge of the within-scanpoint groups
produced by 02b_match_views.py.

For each within-scanpoint group (object_id from 02b), compute:
    rep_world_pos = median of member world_pos
    rep_embed     = L2-normalize(mean of member embeddings)

Then merge two groups (assign the same new object_id) when:
    ‖rep_world_pos_a − rep_world_pos_b‖ < WORLD_DIST_M
    cosine(rep_embed_a, rep_embed_b)    > COS_THRES

A grid bucket (1 m cells) limits comparison to nearby groups, so the
overall pass is O(N × k) where k is the average bucket density (typ. < 50).

Input:
    <out-dir>/embeddings.npy       (N × 1024, from 02_embed)
    <out-dir>/metadata.json        (N rows with world_pos, from 03_backproject)
    <out-dir>/object_ids.npy       (N int32, from 02b_match_views)

Output:
    <out-dir>/object_ids.npy       (overwritten — now dense across all sps)

Run AFTER 03_backproject.py (needs world_pos) and 02b_match_views.py.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_sys_path = str(Path(__file__).parent)
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)
from _paths import space, space_choices


# ── Config ──
# Both thresholds must be tight to avoid TRANSITIVE merge runaway:
# union-find chains pairwise matches (A↔B, B↔C, C↔D ...) so a loose
# cosine threshold lets repetitive architecture (ceiling tiles, floor
# tiles, wall panels) collapse into one giant cross-building blob.
# Cosine 0.92+ is what kills the chain on this kind of data.
WORLD_DIST_M = 0.40      # max 3D distance between two groups' rep positions
COS_THRES    = 0.92      # min cosine between two groups' rep embeddings
MAX_GROUP_EXTENT_M = 3.0 # post-merge guard: split any group whose member
                         # rep_pos span exceeds this (back-stop against
                         # transitive runaway that slips past the gate)
BUCKET_M     = 1.0       # (unused — KD-tree replaced bucketing)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, choices=space_choices(),
                        help="Space (must be registered in spaces.json).")
    parser.add_argument("--out-dir", default=None,
                        help="Pipeline output dir override (default: spaces.json out_dir).")
    parser.add_argument("--world-dist-m", type=float, default=WORLD_DIST_M,
                        help=f"Max 3D distance to merge (default: {WORLD_DIST_M})")
    parser.add_argument("--cos-thres", type=float, default=COS_THRES,
                        help=f"Min cosine to merge (default: {COS_THRES})")
    args = parser.parse_args()
    args.proposer = "fastsam"   # fastsam-only build

    ROOT = Path(__file__).parent
    out_dir = Path(args.out_dir) if args.out_dir else space(args.space)["out_dir"]
    EMB_FILE  = out_dir / "embeddings.npy"
    META_FILE = out_dir / "metadata.json"
    OID_FILE  = out_dir / "object_ids.npy"

    for f in (EMB_FILE, META_FILE, OID_FILE):
        if not f.exists():
            print(f"[merge] ERROR: {f} not found — earlier step did not run")
            sys.exit(1)

    print(f"[merge] out_dir={out_dir}")
    print(f"[merge] world_dist_m={args.world_dist_m} cos_thres={args.cos_thres}")

    embs = np.load(str(EMB_FILE))        # (N, 1024) L2-normalized
    meta = json.loads(META_FILE.read_text())
    obj  = np.load(str(OID_FILE))        # (N,) int32 — within-scanpoint groups
    N = len(meta)
    print(f"[merge] {N} rows, {int(obj.max())+1} within-sp groups")

    # ── Compute per-group representative (world_pos, embed)
    members_by_oid: dict = defaultdict(list)
    for ri, oid in enumerate(obj):
        members_by_oid[int(oid)].append(ri)

    group_ids   = sorted(members_by_oid.keys())
    rep_pos     = np.full((len(group_ids), 3), np.nan, dtype=np.float64)
    rep_emb     = np.zeros((len(group_ids), embs.shape[1]), dtype=np.float32)
    has_world   = np.zeros(len(group_ids), dtype=bool)

    g_index = {oid: gi for gi, oid in enumerate(group_ids)}

    for gi, oid in enumerate(group_ids):
        members = members_by_oid[oid]
        # Mean embedding (members already L2-normalized)
        rep_emb[gi] = embs[members].mean(axis=0)
        # World-pos median over members that have one
        pts = [meta[r]["world_pos"] for r in members if meta[r].get("world_pos")]
        if pts:
            rep_pos[gi] = np.median(np.asarray(pts, dtype=np.float64), axis=0)
            has_world[gi] = True

    # L2-normalize rep embeddings
    n = np.linalg.norm(rep_emb, axis=1, keepdims=True)
    rep_emb = rep_emb / np.maximum(n, 1e-8)

    # ── Find all group-pairs within WORLD_DIST_M via KD-tree (fast)
    from scipy.spatial import cKDTree
    valid_gis = np.where(has_world)[0]
    pts = rep_pos[valid_gis]
    print(f"[merge] building KD-tree over {len(valid_gis)} grouped points")
    tree = cKDTree(pts)
    pairs_local = tree.query_pairs(r=args.world_dist_m, output_type="ndarray")
    print(f"[merge] {len(pairs_local)} candidate pairs within {args.world_dist_m} m")

    # Map back to global gi indices
    pairs = valid_gis[pairs_local]   # (P, 2)

    # Vectorized cosine for all candidate pairs — chunked to keep peak
    # memory under control (one chunk = 100k pairs × 1024 floats ≈ 400 MB).
    CHUNK = 100_000
    keep_idx = []
    for s in range(0, len(pairs), CHUNK):
        chunk = pairs[s:s + CHUNK]
        c = np.einsum("ij,ij->i",
                      rep_emb[chunk[:, 0]],
                      rep_emb[chunk[:, 1]])
        mask = c > args.cos_thres
        if mask.any():
            keep_idx.append(chunk[mask])
        if (s // CHUNK) % 50 == 0:
            print(f"[merge] cosine chunk {s:>12,}/{len(pairs):,}")
    merge_pairs = np.concatenate(keep_idx) if keep_idx else np.empty((0, 2), dtype=np.int64)

    uf = UnionFind(len(group_ids))
    for a, b in merge_pairs:
        uf.union(int(a), int(b))

    n_unions = len(merge_pairs)
    n_pairs  = len(pairs)
    print(f"[merge] tested {n_pairs} candidate pairs, merged {n_unions}")

    # ── Post-merge extent guard: split runaway chains ───────────────────
    # Even with a tight cos threshold, a chain of small merges can drift
    # over many meters. Any merged super-group whose member rep_pos span
    # exceeds MAX_GROUP_EXTENT_M is reverted to its within-scanpoint
    # constituents (i.e. we undo all the cross-sp unions for that root).
    members_by_root: dict = defaultdict(list)
    for gi in range(len(group_ids)):
        members_by_root[uf.find(gi)].append(gi)

    n_split = 0
    for root, gis in list(members_by_root.items()):
        if len(gis) < 2:
            continue
        gis_world = [g for g in gis if has_world[g]]
        if len(gis_world) < 2:
            continue
        pts = rep_pos[gis_world]
        extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        if extent <= MAX_GROUP_EXTENT_M:
            continue
        # Revert: each within-sp group goes back to being its own root
        for g in gis:
            uf.p[g] = g
            uf.r[g] = 0
        n_split += 1
    print(f"[merge] split {n_split} runaway groups (extent > {MAX_GROUP_EXTENT_M} m)")

    # ── Resolve, propagate to row-level, dense-renumber
    new_group_for_local = np.array([uf.find(gi) for gi in range(len(group_ids))],
                                   dtype=np.int64)
    # Map each row's old (within-sp) group → its new (cross-sp) group → local idx
    new_obj_raw = np.empty(N, dtype=np.int64)
    for ri in range(N):
        old_oid   = int(obj[ri])
        local_idx = g_index[old_oid]
        new_obj_raw[ri] = new_group_for_local[local_idx]

    _, dense = np.unique(new_obj_raw, return_inverse=True)
    object_ids = dense.astype(np.int32)
    n_unique = int(object_ids.max()) + 1 if N else 0

    np.save(str(OID_FILE), object_ids)
    print(f"[merge] {N} rows → {n_unique} unique objects (across all scanpoints)")
    print(f"[merge] Overwrote {OID_FILE}")


if __name__ == "__main__":
    main()
