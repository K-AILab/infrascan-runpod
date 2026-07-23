"""Bake model weights into the image at build time (no network volume, so we
avoid re-downloading multi-GB weights on every serverless cold start).
Best-effort: a miss here just falls back to a one-time runtime download."""
import os
os.environ.setdefault("HF_HOME", "/opt/models/hf")
os.environ.setdefault("TORCH_HOME", "/opt/models/torch")
try:
    from depth_anything_3.api import DepthAnything3
    DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    print("[prefetch] DA3 baked", flush=True)
except Exception as e:
    print("[prefetch] DA3 skipped:", e, flush=True)
try:
    import torch
    torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
    print("[prefetch] DINOv2 baked", flush=True)
except Exception as e:
    print("[prefetch] DINOv2 skipped:", e, flush=True)
