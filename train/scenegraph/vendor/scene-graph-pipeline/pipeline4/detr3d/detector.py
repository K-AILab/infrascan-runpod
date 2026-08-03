# Inference wrapper around vendored 3DETR: builds the model from a checkpoint's
# saved args, and runs detection on a single Z-up point-cloud window.
from types import SimpleNamespace

import numpy as np
import torch

from .model_3detr import build_3detr
from .scannet_config import ScannetDatasetConfig

SCANNET_CLASSES = [
    "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator", "showercurtrain",
    "toilet", "sink", "bathtub", "garbagebin",
]

# Defaults from 3detr main.py (scannet_masked_ep1080 was trained with enc_type=masked)
_DEFAULT_ARGS = dict(
    enc_type="masked",
    enc_nlayers=3,
    enc_dim=256,
    enc_ffn_dim=128,
    enc_dropout=0.1,
    enc_nhead=4,
    enc_activation="relu",
    dec_nlayers=8,
    dec_dim=256,
    dec_ffn_dim=256,
    dec_dropout=0.1,
    dec_nhead=4,
    mlp_dropout=0.3,
    preenc_npoints=2048,
    nqueries=256,
    use_color=False,
)


class Detr3DDetector:
    def __init__(self, ckpt_path, device="cuda", num_points=40000):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_points = num_points
        self.dataset_config = ScannetDatasetConfig()

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args = dict(_DEFAULT_ARGS)
        saved = ckpt.get("args", None)
        if saved is not None:
            for k in args:
                if hasattr(saved, k):
                    args[k] = getattr(saved, k)
        self.args = SimpleNamespace(**args)

        model, _ = build_3detr(self.args, self.dataset_config)
        sd = ckpt["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def detect(self, xyz, min_prob=0.05, seed=0):
        """xyz: (N,3) float array in the model frame (Z-up, meters, room-scale).

        Returns dict of numpy arrays: boxes_min (M,3), boxes_max (M,3),
        prob (M,), sem_cls (M,), sem_prob (M,) — in the same frame as the input.
        """
        n = xyz.shape[0]
        rng = np.random.default_rng(seed)
        if n >= self.num_points:
            choice = rng.choice(n, self.num_points, replace=False)
        else:
            choice = np.concatenate(
                [np.arange(n), rng.choice(n, self.num_points - n, replace=True)]
            )
        pc = xyz[choice].astype(np.float32)

        pc_t = torch.from_numpy(pc).unsqueeze(0).to(self.device)
        inputs = {
            "point_clouds": pc_t,
            "point_cloud_dims_min": pc_t.min(dim=1).values,
            "point_cloud_dims_max": pc_t.max(dim=1).values,
        }
        out = self.model(inputs)["outputs"]

        prob = out["objectness_prob"][0]                       # (nq,)
        keep = prob > min_prob
        center = out["center_unnormalized"][0][keep]           # (M,3) model frame
        size = out["size_unnormalized"][0][keep]               # (M,3) = (dx,dy,dz)
        sem_prob, sem_cls = out["sem_cls_prob"][0][keep].max(dim=-1)

        # ScanNet has no box rotation (angle==0): the box is an AABB in the
        # model (Z-up) frame with extents (dx, dy, dz) = size_unnormalized.
        bmin = (center - size / 2).cpu().numpy()
        bmax = (center + size / 2).cpu().numpy()
        return {
            "boxes_min": bmin,
            "boxes_max": bmax,
            "prob": prob[keep].cpu().numpy(),
            "sem_cls": sem_cls.cpu().numpy(),
            "sem_prob": sem_prob.cpu().numpy(),
        }
