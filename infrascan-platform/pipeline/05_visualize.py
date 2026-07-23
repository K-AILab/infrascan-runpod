#!/usr/bin/env python3
"""
Offline step 5 (optional): Visualize the embedding distribution in 2D.

Projects DINOv2 embeddings to 2D with UMAP (default) or t-SNE, colored by
proposal score. Helps you eyeball whether "lights" form tight clusters or
smear out → indicator of retrieval quality.

Usage:
    pip install umap-learn scikit-learn matplotlib    # once
    python sandbox/offline/05_visualize.py --out-dir sandbox/offline/out_sam3
    python sandbox/offline/05_visualize.py --method tsne --color-by scanpoint
    python sandbox/offline/05_visualize.py --query-idx 1234   # highlight top-20 for one proposal

Output: <out-dir>/embedding_map.png
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paths import space, space_choices

parser = argparse.ArgumentParser()
parser.add_argument("--space", required=True, choices=space_choices(),
                    help="Space (must be registered in spaces.json).")
parser.add_argument("--out-dir", default=None,
                    help="Directory containing embeddings.npy + metadata.json "
                         "(default: spaces.json out_dir).")
parser.add_argument("--method", choices=["umap", "tsne", "pca"], default="umap")
parser.add_argument("--color-by", choices=["score", "scanpoint", "query"], default="score")
parser.add_argument("--query-idx", type=int, default=None,
                    help="Proposal index to highlight; top-20 FAISS neighbors get drawn larger")
parser.add_argument("--sample", type=int, default=None,
                    help="If set, randomly sample N points (speeds up t-SNE)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

ROOT = Path(__file__).parent
OUT_DIR = Path(args.out_dir) if args.out_dir else space(args.space)["out_dir"]
EMB_FILE  = OUT_DIR / "embeddings.npy"
META_FILE = OUT_DIR / "metadata.json"
OUT_PNG   = OUT_DIR / "embedding_map.png"

print(f"[viz] Loading {EMB_FILE}")
embs = np.load(str(EMB_FILE)).astype("float32")
meta = json.loads(META_FILE.read_text())
N = embs.shape[0]
print(f"[viz] {N} embeddings, dim={embs.shape[1]}")

if args.sample and args.sample < N:
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(N, args.sample, replace=False)
    if args.query_idx is not None and args.query_idx not in idx:
        idx[0] = args.query_idx
    embs_s = embs[idx]
    meta_s = [meta[i] for i in idx]
    idx_map = {orig: i for i, orig in enumerate(idx)}
    print(f"[viz] Sampled to {args.sample} points")
else:
    embs_s = embs
    meta_s = meta
    idx_map = {i: i for i in range(N)}

# ── Projection ──
print(f"[viz] Projecting with {args.method.upper()} ...")
if args.method == "umap":
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                        random_state=args.seed)
    pts = reducer.fit_transform(embs_s)
elif args.method == "tsne":
    from sklearn.manifold import TSNE
    pts = TSNE(n_components=2, metric="cosine", init="pca",
               perplexity=30, random_state=args.seed).fit_transform(embs_s)
else:  # pca
    from sklearn.decomposition import PCA
    pts = PCA(n_components=2, random_state=args.seed).fit_transform(embs_s)

# ── Color ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
ax.set_facecolor("#0e0e0e")

if args.color_by == "score":
    scores = np.array([m.get("score", 0.0) for m in meta_s])
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=scores, cmap="viridis",
                    s=4, alpha=0.6, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("Proposal score", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
elif args.color_by == "scanpoint":
    def sp_id(m):
        match = re.search(r"(\d+)_pz", m.get("pano", ""))
        return int(match.group(1)) if match else -1
    sps = np.array([sp_id(m) for m in meta_s])
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=sps, cmap="tab20", s=4, alpha=0.6, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("Scanpoint ID", color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
elif args.color_by == "query":
    if args.query_idx is None:
        raise SystemExit("--color-by query requires --query-idx")
    import faiss
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    D, I = index.search(embs[args.query_idx:args.query_idx+1], 20)
    neighbors = set(I[0].tolist())
    # Plot all in grey, then overlay neighbors green, query red
    ax.scatter(pts[:, 0], pts[:, 1], c="#666666", s=3, alpha=0.4, linewidths=0)
    nb_local = [idx_map[n] for n in neighbors if n in idx_map]
    if nb_local:
        ax.scatter(pts[nb_local, 0], pts[nb_local, 1], c="#69db7c",
                   s=30, alpha=0.95, label="top-20 neighbors", edgecolors="white", linewidths=0.5)
    if args.query_idx in idx_map:
        q = idx_map[args.query_idx]
        ax.scatter(pts[q, 0], pts[q, 1], c="#ff6b6b", s=80,
                   label="query", edgecolors="white", linewidths=1.2)
    ax.legend(loc="upper right", facecolor="#222222", labelcolor="white")

ax.set_title(f"{args.method.upper()} of {N} DINOv2 embeddings — color: {args.color_by}",
             color="white", fontsize=12)
ax.tick_params(colors="white")
for s in ax.spines.values():
    s.set_color("#444444")

plt.tight_layout()
plt.savefig(OUT_PNG, facecolor="#0e0e0e")
print(f"[viz] Saved → {OUT_PNG}")
