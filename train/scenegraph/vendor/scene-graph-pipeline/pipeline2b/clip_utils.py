#!/usr/bin/env python3
"""
clip_utils.py — shared open_clip loading/encoding helpers.

Extracted from scene_graph.py's `_build_clip_classifier()` so the same CLIP
model-loading and prompt-ensembling logic can be reused by:
  - scene_graph.py           (per-object zero-shot classification vs LABEL_VOCAB)
  - 03d_clip_instances.py    (per-instance CLIP embedding for open-vocab search)
  - server/server.py         (live free-text query encoding)

Model: open_clip ViT-H-14 / dfn5b (same checkpoint scene_graph.py already used).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

DEFAULT_MODEL_NAME = "ViT-H-14"
DEFAULT_PRETRAINED = "dfn5b"


def crop_bbox_padded(pil: "Image.Image", bbox, pad_frac: float = 0.10,
                      min_size: int = 10) -> "Image.Image | None":
    """Crop `bbox` ([x1,y1,x2,y2] absolute pixel coords) out of `pil` with a
    relative padding margin, matching scene_graph.py's original label_objects()
    crop logic. Returns None if the resulting crop is degenerate."""
    x1, y1, x2, y2 = bbox
    W, H = pil.size
    pw, ph = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    crop = pil.crop((max(0, x1 - pw), max(0, y1 - ph),
                      min(W, x2 + pw), min(H, y2 + ph)))
    if crop.width <= min_size or crop.height <= min_size:
        return None
    return crop


def load_clip_model(model_name: str = DEFAULT_MODEL_NAME,
                     pretrained: str = DEFAULT_PRETRAINED,
                     device: str | None = None):
    """Load an open_clip model+preprocess+tokenizer. Raises on failure (missing
    open_clip, no weights, etc) — callers should catch and fall back."""
    import torch
    import open_clip

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device


def encode_images(model, preprocess, device, crops: list) -> np.ndarray:
    """crops: list of PIL images. Returns (N_crops, D) float32, L2-normalised."""
    import torch

    tensors = torch.stack([preprocess(c) for c in crops]).to(device)
    with torch.no_grad():
        feats = model.encode_image(tensors).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def encode_texts(model, tokenizer, device, texts: list[str]) -> np.ndarray:
    """texts: list of strings. Returns (N_texts, D) float32, L2-normalised
    (per-text — caller decides whether/how to pool)."""
    import torch

    with torch.no_grad():
        toks = tokenizer(texts).to(device)
        feats = model.encode_text(toks).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def average_normalize(feats: np.ndarray) -> np.ndarray:
    """Mean-pool a (N, D) feature set into one L2-normalised (D,) vector."""
    avg = feats.mean(axis=0)
    norm = float(np.linalg.norm(avg))
    return avg / max(norm, 1e-8)


def encode_text_query(model, tokenizer, device, prompt: str) -> np.ndarray:
    """Encode a single free-text query into one L2-normalised (D,) vector."""
    feats = encode_texts(model, tokenizer, device, [prompt])
    return average_normalize(feats)


def build_label_text_embeddings(model, tokenizer, device,
                                 vocab: list[str],
                                 prompts_dict: dict[str, list[str]]) -> np.ndarray:
    """
    Per-class multi-prompt ensemble (CuPL / CLIP paper §3.1): for each label,
    encode all its descriptive prompts (plus 2 generic fallbacks) and average
    the L2-normalised embeddings into one representative vector per class.

    Returns (N_labels, D) float32.
    """
    label_embeds = []
    for lbl in vocab:
        specific = prompts_dict.get(lbl, [])
        generic = [f"a photo of a {lbl.replace('_', ' ')}",
                   f"an image of {lbl.replace('_', ' ')} in a building"]
        all_prompts = list(dict.fromkeys(specific + generic))  # dedupe
        feats = encode_texts(model, tokenizer, device, all_prompts)
        label_embeds.append(average_normalize(feats))
    return np.stack(label_embeds).astype(np.float32)


def build_clip_classifier(vocab: list[str],
                           prompts_dict: dict[str, list[str]] | None = None,
                           model_name: str = DEFAULT_MODEL_NAME,
                           pretrained: str = DEFAULT_PRETRAINED,
                           device: str | None = None):
    """
    Back-compat wrapper matching scene_graph.py's original
    `_build_clip_classifier()` contract:

    Returns (encode_images_fn, text_features) where:
      encode_images_fn(crops) → (N_crops, D) float32 numpy, L2-normalised
      text_features           → (N_labels, D) float32 numpy, L2-normalised
    Returns None on any failure (missing open_clip, OOM, etc).
    """
    prompts_dict = prompts_dict or {}
    try:
        model, preprocess, tokenizer, device = load_clip_model(
            model_name, pretrained, device
        )
        t_np = build_label_text_embeddings(model, tokenizer, device, vocab, prompts_dict)

        def _encode(crops: list) -> np.ndarray:
            return encode_images(model, preprocess, device, crops)

        n_with_prompts = sum(1 for l in vocab if l in prompts_dict)
        print(f"[clip_utils] CLIP ({model_name}/{pretrained}) ready on {device} "
              f"— {len(vocab)} labels encoded "
              f"({n_with_prompts} with custom prompt sets)")
        return _encode, t_np
    except Exception as e:
        print(f"[clip_utils] open_clip unavailable ({e}) — falling back to heuristics")
        return None
