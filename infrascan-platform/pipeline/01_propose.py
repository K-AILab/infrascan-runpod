#!/usr/bin/env python3
"""
Offline step 1: Run class-agnostic object proposals on all 5,940 views.

Supports four proposers (--proposer flag):
  grounding_dino  (default) — GroundingDINO with prompt "object."
                               Sparser, faster, less over-segmentation.
  sam3                       — SAM3 with text prompt "object."
                               Denser, more proposals per view.
  fastsam                    — Class-agnostic mask proposer (ultralytics
                               FastSAM-x). Ignores PROMPT. Produces masks
                               like SAM3 but no text grounding.
  mobilesam                  — MobileSAM via ultralytics. Lighter weight
                               (~40 MB) class-agnostic SAM. Ignores PROMPT.

Usage:
    conda activate sam3
    python sandbox/offline/01_propose.py                       # Grounding DINO
    python sandbox/offline/01_propose.py --proposer sam3       # SAM3
    python sandbox/offline/01_propose.py --proposer fastsam    # FastSAM
    python sandbox/offline/01_propose.py --proposer mobilesam  # MobileSAM

Output: sandbox/offline/out/proposals.jsonl
  One JSON line per view:
  {view_id, pano, frame_idx, pos, proposer, proposals: [{bbox, score}, ...]}

Resume-safe: skips view_ids already written to the output file.
"""
import sys
import os
import io
import json
import base64
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms as box_nms

# proposal model (used as default when --proposer is not passed)
PROP_MODEL = "sam3"  # "dino", "sam3", "fastsam", or "mobilesam"
_PROP_MODEL_ALIASES = {"dino": "grounding_dino", "grounding_dino": "grounding_dino",
                       "sam3": "sam3", "fastsam": "fastsam",
                       "mobilesam": "mobilesam"}


# ── Paths ──────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent
PROJ   = Path(os.environ.get("INFRASCAN_TAGGING_MODELS", str(ROOT.parent / "external")))
OBJ_PROP     = PROJ / "object_proposals"

sys.path.insert(0, str(ROOT))
from _paths import space, space_choices  # noqa: E402
SAM3_DIR     = OBJ_PROP / "sam3" / "model_codes"
DINO_DIR     = OBJ_PROP / "groundingdino" / "model_codes"
FASTSAM_WEIGHTS = OBJ_PROP / "fastsam" / "weights" / "FastSAM-x.pt"
MOBILESAM_WEIGHTS = OBJ_PROP / "mobilesam" / "weights" / "mobile_sam.pt"
DEFAULT_OUT_DIR = ROOT / "out"

# ── Config ─────────────────────────────────────────────────────────────────
PROMPT         = "lights"
NMS_IOU        = 0.50
MIN_AREA_FRAC  = 0.000   # drop boxes < 0.3% of image area
MAX_AREA_FRAC  = 0.35    # drop boxes > 35% (walls / background)
LOG_EVERY      = 50
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.bfloat16

# Grounding DINO defaults (from object_proposals/groundingdino/propose.py)
DINO_BOX_THRESH  = 0.30
DINO_TEXT_THRESH = 0.25

# SAM3 defaults
SAM3_CONFIDENCE  = 0.20

# FastSAM defaults — matches object_proposals/fastsam/inference.py
FASTSAM_CONFIDENCE = 0.2

# MobileSAM defaults — same conf scale as FastSAM
MOBILESAM_CONFIDENCE = 0.5


# ── Package loaders ────────────────────────────────────────────────────────
def _load_pkg(pkg_name: str, pkg_dir: Path):
    init = pkg_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        pkg_name, init, submodule_search_locations=[str(pkg_dir)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)


# ── Model loaders ──────────────────────────────────────────────────────────
def load_grounding_dino():
    _load_pkg("groundingdino", DINO_DIR)
    from groundingdino import build_groundingdino_image_model
    from groundingdino.processor import GroundingDinoProcessor

    print(f"[propose] Loading GroundingDINO on {DEVICE} ...")
    model = build_groundingdino_image_model(device=DEVICE)
    processor = GroundingDinoProcessor(
        model, device=DEVICE,
        box_threshold=DINO_BOX_THRESH,
        text_threshold=DINO_TEXT_THRESH,
    )
    print("[propose] GroundingDINO ready.")
    return processor


def load_sam3():
    _load_pkg("sam3", SAM3_DIR)
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    bpe_path = str(SAM3_DIR / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    print(f"[propose] Loading SAM3 on {DEVICE} ...")
    with torch.autocast(DEVICE, dtype=DTYPE):
        model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=SAM3_CONFIDENCE)
    print("[propose] SAM3 ready.")
    return processor


def load_fastsam():
    from ultralytics import FastSAM
    if not FASTSAM_WEIGHTS.exists():
        FASTSAM_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        print(f"[propose] FastSAM weights not at {FASTSAM_WEIGHTS} — downloading from ultralytics ...")
        import urllib.request
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt"
        urllib.request.urlretrieve(url, str(FASTSAM_WEIGHTS))
    print(f"[propose] Loading FastSAM on {DEVICE} ...")
    model = FastSAM(str(FASTSAM_WEIGHTS))
    print("[propose] FastSAM ready.")
    return model


def load_mobilesam():
    from ultralytics import SAM
    if not MOBILESAM_WEIGHTS.exists():
        raise FileNotFoundError(f"MobileSAM weights not found at {MOBILESAM_WEIGHTS}")
    print(f"[propose] Loading MobileSAM on {DEVICE} ...")
    model = SAM(str(MOBILESAM_WEIGHTS))
    print("[propose] MobileSAM ready.")
    return model


# ── Mask encoding ──────────────────────────────────────────────────────────
def encode_mask_b64(mask_arr: np.ndarray) -> str:
    """Encode a HxW bool mask as base64 PNG (1-bit). ~1-3 KB per mask."""
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(mask_arr, mode="L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Inference ──────────────────────────────────────────────────────────────
def run_view(processor, pil: Image.Image, proposer: str):
    """Return (boxes_list, scores_list, masks_b64_list_or_None) for one PIL image.

    Grounding DINO does not produce masks → masks_b64_list is None.
    SAM3 produces masks → list of base64-encoded PNGs (one per kept proposal).
    """
    if proposer == "grounding_dino":
        with torch.no_grad():
            state = processor.set_image(pil)
            state = processor.set_text_prompt(state=state, prompt=PROMPT)
    elif proposer == "sam3":
        with torch.autocast(DEVICE, dtype=DTYPE), torch.no_grad():
            processor.set_confidence_threshold(SAM3_CONFIDENCE)
            state = processor.set_image(pil)
            state = processor.set_text_prompt(prompt=PROMPT, state=state)
    elif proposer in ("fastsam", "mobilesam"):
        # ultralytics FastSAM / MobileSAM: class-agnostic, ignores PROMPT.
        # Both return ultralytics Results with .boxes and .masks.
        conf = FASTSAM_CONFIDENCE if proposer == "fastsam" else MOBILESAM_CONFIDENCE
        results = processor.predict(pil, conf=conf, device=DEVICE, verbose=False)
        r = results[0]
        if r.boxes is None or r.boxes.xyxy.numel() == 0:
            return [], [], None
        m = r.masks.data if r.masks is not None else None  # (N, H, W) float/bool
        # Masks come back at the network's input resolution. Resize to
        # the source view so the encoded mask aligns with the bbox coords.
        if m is not None:
            W, H = pil.size
            if m.shape[-2:] != (H, W):
                m = torch.nn.functional.interpolate(
                    m.unsqueeze(1).float(), size=(H, W),
                    mode="nearest").squeeze(1)
            m = m > 0.5
        state = {
            "boxes":  r.boxes.xyxy,    # (N, 4) in source-image pixels
            "scores": r.boxes.conf,    # (N,)
            "masks":  m,               # (N, H, W) bool or None
        }
    else:
        raise ValueError(f"Unknown proposer: {proposer}")

    if state["scores"].numel() == 0:
        return [], [], None

    boxes_t  = state["boxes"].cpu().float()
    scores_t = state["scores"].cpu().float()

    # Mask-producing proposers (SAM3, FastSAM) ship per-detection masks.
    masks_t = None
    if state.get("masks") is not None and state["masks"].numel() > 0:
        m = state["masks"]
        if m.dim() == 4:
            m = m.squeeze(1)            # (N, 1, H, W) → (N, H, W)
        masks_t = m.cpu()               # bool tensor

    keep = box_nms(boxes_t, scores_t, NMS_IOU)
    boxes  = boxes_t[keep].numpy().tolist()
    scores = scores_t[keep].numpy().tolist()
    masks  = masks_t[keep].numpy() if masks_t is not None else None

    W, H = pil.size
    img_area = W * H
    filtered_b, filtered_s, filtered_m = [], [], []
    for i, (b, s) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = b
        area = (x2 - x1) * (y2 - y1)
        if MIN_AREA_FRAC * img_area <= area <= MAX_AREA_FRAC * img_area:
            filtered_b.append(b)
            filtered_s.append(s)
            if masks is not None:
                filtered_m.append(encode_mask_b64(masks[i]))

    return filtered_b, filtered_s, (filtered_m if masks is not None else None)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, choices=space_choices(),
                        help="Space to run on (must be registered in spaces.json).")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory override (default: spaces.json out_dir).")
    args = parser.parse_args()

    # Public-repo build: fastsam is the only supported proposer.
    args.proposer = "fastsam"

    sp = space(args.space)
    views_dir = sp["views"]
    cameras_json_path = sp["cameras"]
    out_dir = Path(args.out_dir) if args.out_dir else sp["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT_FILE = out_dir / "proposals.jsonl"
    print(f"[propose] space={args.space} views={views_dir}")
    print(f"[propose] Writing to {out_dir}")

    cameras = json.loads(cameras_json_path.read_text())
    print(f"[propose] {len(cameras)} views | proposer={args.proposer}")

    # Resume: collect already-processed view_ids
    done_ids: set = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["view_id"])
                except Exception:
                    pass
        if done_ids:
            print(f"[propose] Resuming — {len(done_ids)} views already done")

    if args.proposer == "grounding_dino":
        processor = load_grounding_dino()
    elif args.proposer == "sam3":
        processor = load_sam3()
    elif args.proposer == "fastsam":
        processor = load_fastsam()
    elif args.proposer == "mobilesam":
        processor = load_mobilesam()
    else:
        raise ValueError(f"Unknown proposer: {args.proposer}")

    total_views = 0
    total_proposals = 0

    with open(OUT_FILE, "a") as out_f:
        for cam in cameras:
            view_id = cam["id"]
            if view_id in done_ids:
                continue

            pano_name = Path(cam["pano"]).name
            view_path = views_dir / pano_name
            if not view_path.exists():
                print(f"[warn] missing view: {view_path}")
                continue

            pil = Image.open(view_path).convert("RGB")
            boxes, scores, masks_b64 = run_view(processor, pil, args.proposer)

            proposals = []
            for i, (b, s) in enumerate(zip(boxes, scores)):
                entry = {"bbox": b, "score": s}
                if masks_b64 is not None:
                    entry["mask_b64"] = masks_b64[i]
                proposals.append(entry)

            record = {
                "view_id":   view_id,
                "pano":      cam["pano"],
                "frame_idx": view_id,
                "pos":       cam["pos"],
                "proposer":  args.proposer,
                "proposals": proposals,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            total_views     += 1
            total_proposals += len(boxes)

            if total_views % LOG_EVERY == 0:
                print(f"[propose] {view_id+1}/{len(cameras)} "
                      f"| {total_proposals} total proposals "
                      f"| last: {len(boxes)} boxes")

    print(f"\n[propose] Done.")
    print(f"  Proposer        : {args.proposer}")
    print(f"  Views processed : {total_views}")
    print(f"  Total proposals : {total_proposals}")
    print(f"  Output          : {OUT_FILE}")


if __name__ == "__main__":
    main()
