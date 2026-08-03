#!/usr/bin/env python3
"""
Offline step 4: Build FAISS index from embeddings.

Usage:
    python sandbox/offline/04_index.py --proposer fastsam           # uses out_fastsam/
    python sandbox/offline/04_index.py --out-dir out_sam3_lights    # custom dir

Input:  <out-dir>/embeddings.npy
Output: <out-dir>/index.faiss
"""
import argparse
import numpy as np
import faiss
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paths import space, space_choices

parser = argparse.ArgumentParser()
parser.add_argument("--space", required=True, choices=space_choices(),
                    help="Space (must be registered in spaces.json).")
parser.add_argument("--out-dir", default=None,
                    help="Directory containing embeddings.npy "
                         "(default: spaces.json out_dir).")
args = parser.parse_args()
ROOT = Path(__file__).parent
OUT_DIR = Path(args.out_dir) if args.out_dir else space(args.space)["out_dir"]
EMBEDDINGS_FILE = OUT_DIR / "embeddings.npy"
INDEX_FILE      = OUT_DIR / "index.faiss"

embs = np.load(str(EMBEDDINGS_FILE)).astype("float32")
print(f"[index] Loaded {embs.shape[0]} embeddings, dim={embs.shape[1]} from {OUT_DIR}")

norms = np.linalg.norm(embs[:100], axis=1)
print(f"[index] Norm check (should be ~1.0): min={norms.min():.4f} max={norms.max():.4f}")

index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)
faiss.write_index(index, str(INDEX_FILE))

print(f"[index] {index.ntotal} vectors indexed.")
print(f"[index] Output: {INDEX_FILE}")

D, I = index.search(embs[:1], k=2)
print(f"[index] Sanity: top-2 for emb[0] → ids={I[0].tolist()} scores={D[0].tolist()}")
