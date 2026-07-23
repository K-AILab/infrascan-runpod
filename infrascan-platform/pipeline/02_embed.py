#!/usr/bin/env python3
"""
Offline step 2: Embed each proposal crop with DINOv2 ViT-L/14.

Usage:
    conda activate sam3
    pip install huggingface_hub timm   # once
    python sandbox/offline/02_embed.py

Input:  sandbox/offline/out/proposals.jsonl
Output:
    sandbox/offline/out/embeddings.npy   shape (N, 1024) float32, L2-normalized
    sandbox/offline/out/metadata.json    list of N dicts with bbox / view info

Resume-safe: appends to existing embeddings.npy + metadata.json.
"""
import io
import os
import json
import base64
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── Paths ──────────────────────────────────────────────────────────────────
import argparse

ROOT          = Path(__file__).parent
PROJ          = Path(os.environ.get("INFRASCAN_TAGGING_MODELS", str(ROOT.parent / "external")))

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paths import space, space_choices  # noqa: E402

# ── Config ─────────────────────────────────────────────────────────────────
BATCH_SIZE     = 64
PAD_FRAC       = 0.10      # 10% padding around each bbox crop
RESIZE         = (224, 224)
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_EVERY     = 5000      # checkpoint every N proposals
MASK_BG_COLOR  = (128, 128, 128)   # grey background when masking
USE_MASK       = True              # use mask_b64 if present in proposals

IMAGENET_TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_dinov2():
    print("[embed] Loading DINOv2 ViT-L/14 ...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14",
                           verbose=False)
    model = model.to(DEVICE).half().eval()
    print("[embed] DINOv2 ready.")
    return model


def crop_padded(pil: Image.Image, bbox, pad: float = PAD_FRAC) -> Image.Image:
    W, H = pil.size
    x1, y1, x2, y2 = bbox
    pw = (x2 - x1) * pad
    ph = (y2 - y1) * pad
    x1 = max(0.0, x1 - pw)
    y1 = max(0.0, y1 - ph)
    x2 = min(float(W), x2 + pw)
    y2 = min(float(H), y2 + ph)
    return pil.crop((int(x1), int(y1), int(x2), int(y2))).resize(RESIZE, Image.BILINEAR)


def mask_apply_and_crop(pil: Image.Image, bbox, mask_b64: str,
                        pad: float = PAD_FRAC) -> Image.Image:
    """Apply segmentation mask → grey background → crop with padding → resize."""
    # Decode mask PNG → boolean array
    mask_png = base64.b64decode(mask_b64)
    mask_pil = Image.open(io.BytesIO(mask_png)).convert("L")
    if mask_pil.size != pil.size:
        mask_pil = mask_pil.resize(pil.size, Image.NEAREST)

    # Composite: keep where mask>0, fill grey elsewhere
    bg = Image.new("RGB", pil.size, MASK_BG_COLOR)
    composed = Image.composite(pil, bg, mask_pil)
    return crop_padded(composed, bbox, pad)


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)


def embed_batch(model, crops: list) -> np.ndarray:
    tensors = torch.stack([IMAGENET_TFM(c) for c in crops]).to(DEVICE).half()
    with torch.no_grad():
        feats = model(tensors)
    return feats.cpu().float().numpy()


def load_proposals(PROPOSALS_FILE):
    items = []
    with open(PROPOSALS_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            pano_name = Path(rec["pano"]).name
            for prop in rec["proposals"]:
                items.append({
                    "view_id":   rec["view_id"],
                    "pano":      rec["pano"],
                    "pano_name": pano_name,
                    "frame_idx": rec["frame_idx"],
                    "pos":       rec["pos"],
                    "bbox":      prop["bbox"],
                    "score":     prop["score"],
                    "mask_b64":  prop.get("mask_b64"),   # may be None
                })
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, choices=space_choices(),
                        help="Space (must be registered in spaces.json).")
    parser.add_argument("--out-dir", default=None,
                        help="Directory containing proposals.jsonl "
                             "(default: spaces.json out_dir).")
    args = parser.parse_args()
    args.proposer = "fastsam"   # fastsam-only build
    views_dir = space(args.space)["views"]
    out_dir = Path(args.out_dir) if args.out_dir else space(args.space)["out_dir"]
    PROPOSALS_FILE  = out_dir / "proposals.jsonl"
    EMBEDDINGS_FILE = out_dir / "embeddings.npy"
    METADATA_FILE   = out_dir / "metadata.json"
    print(f"[embed] Using {out_dir}")

    if not PROPOSALS_FILE.exists():
        print(f"[embed] ERROR: {PROPOSALS_FILE} not found — run 01_propose.py first")
        return

    items = load_proposals(PROPOSALS_FILE)
    print(f"[embed] {len(items)} proposals loaded from proposals.jsonl")

    # Resume
    start_idx = 0
    emb_chunks = []
    metadata = []
    if EMBEDDINGS_FILE.exists() and METADATA_FILE.exists():
        existing = np.load(str(EMBEDDINGS_FILE))
        metadata = json.loads(METADATA_FILE.read_text())
        start_idx = len(metadata)
        emb_chunks.append(existing)
        print(f"[embed] Resuming from index {start_idx}")

    items = items[start_idx:]
    if not items:
        print("[embed] Nothing to do — all proposals already embedded.")
        return

    model = load_dinov2()

    # Group by pano to open each image once
    items_by_pano: dict = {}
    for it in items:
        items_by_pano.setdefault(it["pano_name"], []).append(it)

    processed = 0
    for pano_name, pano_items in items_by_pano.items():
        view_path = views_dir / pano_name
        if not view_path.exists():
            print(f"[warn] missing {view_path}, skipping {len(pano_items)} proposals")
            continue

        pil = Image.open(view_path).convert("RGB")

        for batch_start in range(0, len(pano_items), BATCH_SIZE):
            batch = pano_items[batch_start : batch_start + BATCH_SIZE]
            crops = []
            for it in batch:
                if USE_MASK and it.get("mask_b64"):
                    crops.append(mask_apply_and_crop(pil, it["bbox"], it["mask_b64"]))
                else:
                    crops.append(crop_padded(pil, it["bbox"]))
            embs  = embed_batch(model, crops)       # (B, 1024)
            embs  = l2_normalize(embs)

            emb_chunks.append(embs)
            for it in batch:
                metadata.append({
                    "view_id":   it["view_id"],
                    "pano":      it["pano"],
                    "frame_idx": it["frame_idx"],
                    "pos":       it["pos"],
                    "bbox":      it["bbox"],
                    "score":     it["score"],
                })

            processed += len(batch)

        # Checkpoint
        if (start_idx + processed) % SAVE_EVERY < BATCH_SIZE or processed == len(items):
            np.save(str(EMBEDDINGS_FILE), np.vstack(emb_chunks).astype("float32"))
            METADATA_FILE.write_text(json.dumps(metadata))
            print(f"[embed] {start_idx + processed}/{start_idx + len(items)} embedded "
                  f"(checkpoint saved)")

    final = np.vstack(emb_chunks).astype("float32")
    np.save(str(EMBEDDINGS_FILE), final)
    METADATA_FILE.write_text(json.dumps(metadata))

    print(f"\n[embed] Done.")
    print(f"  Embeddings shape : {final.shape}")
    print(f"  Output embs      : {EMBEDDINGS_FILE}")
    print(f"  Output metadata  : {METADATA_FILE}")


if __name__ == "__main__":
    main()
