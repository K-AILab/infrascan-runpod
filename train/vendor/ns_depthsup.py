"""Run ns-train splatfacto with scanner-cloud depth supervision (SHINHAN_SPLATFACTO_PLAN.md
§3, Tier 3): an EdgeAwareLogL1 loss between the rendered depth and the pre-reprojected
scanner-cloud depth (see reproject_scanner_depth.py) — metric and multi-view-consistent,
unlike the monocular DA3 maps that failed earlier from per-view scale drift.

Adaptation note: DN-Splatter's original EdgeAwareLogL1 compares depth between ADJACENT
pixel pairs (dense sensor depth). Our GT is a SPARSE point-cloud reprojection (~30-40%
pixel hit-rate) — adjacent-pair comparisons would almost always have an empty neighbour,
making that formulation nearly inert. This version applies the same core idea (log-L1,
downweighted at RGB edges) POINTWISE at each individually-valid GT pixel instead.

Mechanism for matching each training image to its cached depth file: nerfstudio's model
has no direct handle on the dataset's filenames (only a bare index), so this patches
FullImageDatamanager.next_train — the exact place nerfstudio itself resolves
index -> dataset entry — to also stash the file stem onto camera.metadata. Reading that
back in get_loss_dict avoids re-deriving (and risking mis-deriving) that correspondence.

Env: DEPTH_DIR (required), DEPTH_W (default 0.1), DEPTH_START_ITER (default 500)
Usage:  python ns_depthsup.py <ns-train args...>
        python ns_depthsup.py --selftest
"""
import os, sys
from pathlib import Path
import torch
import torch.nn.functional as F

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

DEPTH_DIR    = os.environ.get("DEPTH_DIR", "")
W_DEPTH      = float(os.environ.get("DEPTH_W", "0.1"))
START_ITER   = int(os.environ.get("DEPTH_START_ITER", "500"))
_CACHE_MAX   = 64

# GT depth (from reproject_scanner_depth.py) is in RAW, unscaled scanner-cloud metres.
# The model's rendered "depth" output is in nerfstudio's internal auto-scale-poses frame
# (typically shrunk by ~0.1-0.2x for numerical stability). Comparing them directly, with
# no conversion, means even a PERFECT reconstruction produces a large constant loss
# dominated by the scale mismatch rather than genuine depth error -- confirmed on the
# shinhan depthsup run (dataparser scale 0.1465: perfect-recon loss ~1.0 vs ~0.015 for a
# real 10cm error, properly scaled -- ~1:65 signal-to-noise, with a systematic gradient
# direction, not neutral noise). DEPTH_SCALE must be set to that run's own
# dataparser_transforms.json "scale" value (get it from a short dry-run with the same
# data/orientation-method/center-method/auto-scale-poses settings before launching).
DEPTH_SCALE  = float(os.environ.get("DEPTH_SCALE", "1.0"))


class _DepthCache:
    """Tiny bounded LRU over the reprojected-depth npz files — loaded lazily from disk
    per training step rather than held all in RAM (2604 dense 1024x1024 arrays would be
    ~13GB, and this box has a documented history of hard crashes under memory pressure)."""
    def __init__(self, root):
        self.root = Path(root)
        self._d = {}
        self._order = []

    def get(self, stem):
        if stem in self._d:
            self._order.remove(stem); self._order.append(stem)
            return self._d[stem]
        f = self.root / f"{stem}.npz"
        if not f.exists():
            return None
        import numpy as np
        z = np.load(f)
        depth = torch.from_numpy(z["depth"]).float()
        mask = torch.from_numpy(z["mask"]).bool()
        self._d[stem] = (depth, mask); self._order.append(stem)
        if len(self._order) > _CACHE_MAX:
            old = self._order.pop(0); del self._d[old]
        return depth, mask


_depth_cache = None


def edge_aware_log_l1(pred_depth, gt_depth, mask, gt_rgb):
    """pred_depth,gt_depth: (H,W). mask: (H,W) bool. gt_rgb: (H,W,3) in [0,1]."""
    if not mask.any():
        return torch.tensor(0.0, device=pred_depth.device)
    rgb = gt_rgb.permute(2, 0, 1).unsqueeze(0)                        # (1,3,H,W)
    rgb_pad = F.pad(rgb, (1, 1, 1, 1), mode="replicate")
    gx = (rgb_pad[:, :, 1:-1, 2:] - rgb_pad[:, :, 1:-1, :-2]).abs().mean(1).squeeze(0)   # (H,W)
    gy = (rgb_pad[:, :, 2:, 1:-1] - rgb_pad[:, :, :-2, 1:-1]).abs().mean(1).squeeze(0)   # (H,W)
    grad = gx + gy
    lam = torch.exp(-grad)                                             # (H,W), ->1 smooth, ->0 edge
    err = torch.log1p((pred_depth - gt_depth).abs())
    return (lam[mask] * err[mask]).mean()


def _selftest():
    torch.manual_seed(0)
    H, W = 32, 32
    gt_rgb = torch.rand(H, W, 3)
    gt_depth = torch.rand(H, W) * 3 + 1
    mask = torch.rand(H, W) > 0.6   # ~sparse, like real reprojection

    # perfect prediction -> zero loss
    l0 = edge_aware_log_l1(gt_depth.clone(), gt_depth, mask, gt_rgb)
    print(f"perfect pred loss: {l0.item():.6f} (expect ~0)")
    assert l0.item() < 1e-6

    # wrong prediction -> positive loss
    pred = gt_depth + 0.5
    l1 = edge_aware_log_l1(pred, gt_depth, mask, gt_rgb)
    print(f"off-by-0.5 pred loss: {l1.item():.4f} (expect > 0)")
    assert l1.item() > 0

    # empty mask -> zero, no crash
    l2 = edge_aware_log_l1(pred, gt_depth, torch.zeros(H, W, dtype=torch.bool), gt_rgb)
    print(f"empty-mask loss: {l2.item():.6f} (expect 0, no crash)")
    assert l2.item() == 0.0

    # sharper RGB edge at a valid pixel -> that pixel's contribution should shrink
    # (build two scenes differing only in local RGB gradient at one already-valid pixel)
    rgb_smooth = torch.zeros(H, W, 3) + 0.5
    rgb_edge = rgb_smooth.clone(); rgb_edge[H // 2, W // 2 + 1] = 1.0  # hard local edge
    d = torch.ones(H, W) * 2.0; p = d + 1.0
    m = torch.zeros(H, W, dtype=torch.bool); m[H // 2, W // 2] = True
    l_smooth = edge_aware_log_l1(p, d, m, rgb_smooth)
    l_edge = edge_aware_log_l1(p, d, m, rgb_edge)
    print(f"smooth-region loss: {l_smooth.item():.4f}  vs  near-edge loss: {l_edge.item():.4f} (expect edge < smooth)")
    assert l_edge.item() < l_smooth.item()

    print("SELFTEST OK")


def _install():
    from nerfstudio.models.splatfacto import SplatfactoModel
    from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager

    global _depth_cache
    assert DEPTH_DIR, "DEPTH_DIR env var must point at the reprojected depth_scanner/ dir"
    _depth_cache = _DepthCache(DEPTH_DIR)

    _orig_populate = SplatfactoModel.populate_modules
    _orig_loss = SplatfactoModel.get_loss_dict
    _orig_next_train = FullImageDatamanager.next_train

    def populate_modules(self):
        _orig_populate(self)
        self.config.output_depth_during_training = True   # force depth rendering on

    def next_train(self, step):
        camera, data = _orig_next_train(self, step)
        idx = camera.metadata["cam_idx"]
        stem = Path(self.train_dataset.image_filenames[idx]).stem
        camera.metadata["depth_stem"] = stem
        return camera, data

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = _orig_loss(self, outputs, batch, metrics_dict)
        if not self.training or self.step < START_ITER:
            return loss_dict
        # camera.metadata isn't visible here directly (get_loss_dict only receives
        # outputs/batch) -- get_outputs (patched below) stashes it into outputs instead.
        stem = outputs.get("_depth_stem")
        if stem is None:
            return loss_dict
        cached = _depth_cache.get(stem)
        if cached is None:
            return loss_dict
        gt_depth, mask = cached
        pred_depth = outputs.get("depth")
        if pred_depth is None:
            return loss_dict
        pred_depth = pred_depth.squeeze(-1) if pred_depth.dim() == 3 else pred_depth
        gt_depth = gt_depth.to(pred_depth.device) * DEPTH_SCALE; mask = mask.to(pred_depth.device)
        gt_rgb = self.get_gt_img(batch["image"]).to(pred_depth.device) if isinstance(batch, dict) else None
        if gt_rgb is None:
            return loss_dict
        loss_dict["depth_scanner"] = W_DEPTH * edge_aware_log_l1(pred_depth, gt_depth, mask, gt_rgb)
        return loss_dict

    _orig_get_outputs = SplatfactoModel.get_outputs
    def get_outputs(self, camera):
        out = _orig_get_outputs(self, camera)
        if camera.metadata is not None and "depth_stem" in camera.metadata:
            out["_depth_stem"] = camera.metadata["depth_stem"]
        return out

    SplatfactoModel.populate_modules = populate_modules
    SplatfactoModel.get_outputs = get_outputs
    SplatfactoModel.get_loss_dict = get_loss_dict
    FullImageDatamanager.next_train = next_train
    print(f"[depthsup] patched splatfacto | depth_dir={DEPTH_DIR} w={W_DEPTH} start={START_ITER} "
          f"depth_scale={DEPTH_SCALE}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest(); sys.exit(0)
    _install()
    from nerfstudio.scripts.train import entrypoint
    sys.argv = ["ns-train"] + sys.argv[1:]
    entrypoint()
