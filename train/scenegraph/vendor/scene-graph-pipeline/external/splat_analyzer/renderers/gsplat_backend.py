"""
gsplat (nerfstudio, CUDA) backend — the deployed server + CUDA desktops.

This is the original rasterization path extracted verbatim from render_cameras.py:
a single batched `rasterization` call for RGB, and the per-view depth-as-color trick
(camera-space Z passed as a degree-0 SH "color", then decoded with SH_C0).
"""

from __future__ import annotations

import numpy as np
import torch

from gsplat import rasterization

from .base import Renderer, NEAR_PLANE, FAR_PLANE, SH_C0, ALPHA_EPS


class GsplatRenderer(Renderer):
    name = "gsplat"

    def __init__(self):
        if not torch.cuda.is_available():
            # gsplat's rasterization is a compiled CUDA kernel; it cannot run on CPU.
            raise RuntimeError(
                "gsplat backend requires CUDA. On Apple Silicon use renderer='gsplat-metal'."
            )
        self.device = torch.device("cuda")

    def prepare(self, arrays):
        # gsplat 1.5.3 rasterization() wants activated scales + opacities.
        g = self._to_device(arrays)
        g["scales"] = torch.exp(g["log_scales"])
        g["opacities"] = torch.sigmoid(g["logit_opacities"])
        return g

    def render_rgb(self, g, w2c, K, width, height):
        B = w2c.shape[0]
        K_batch = K.unsqueeze(0).expand(B, -1, -1).contiguous()
        rgb_out, _, _ = rasterization(
            means=g["means"], quats=g["quats"], scales=g["scales"],
            opacities=g["opacities"], colors=g["sh"],
            viewmats=w2c, Ks=K_batch,
            width=width, height=height,
            sh_degree=g["sh_degree"],
            near_plane=NEAR_PLANE, far_plane=FAR_PLANE,
        )
        return (rgb_out.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # (B,H,W,3)

    def render_depth(self, g, w2c, K, width, height):
        """Depth rendered as a degree-0 SH colour, then decoded and normalised by alpha.

        With sh_degree=0 each Gaussian's "colour" is SH_C0*z + 0.5, so the
        rasterized pixel is

            raw = Sum_i w_i (SH_C0*z_i + 0.5) = SH_C0*Sum_i w_i z_i + 0.5*A

        where A = Sum_i w_i is the accumulated alpha. The expected depth is
        Sum_i w_i z_i / A, which requires subtracting the 0.5*A background term
        BEFORE dividing. Decoding as (raw - 0.5)/SH_C0 instead assumes A == 1
        and leaves a constant -1.7725*(1 - A) offset on every partially covered
        pixel — large enough on a scene of modest depth range to drive the
        result negative, where it is clamped to zero and read as "no hit".

        Alpha is returned alongside because the pipeline needs it: a detection
        whose box contains no accumulated alpha is a detection on empty space.
        """
        B = w2c.shape[0]
        N = g["means"].shape[0]
        K_batch = K.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Camera-space Z for every Gaussian across all B views in one bmm.
        means_h = torch.cat(
            [g["means"], torch.ones(N, 1, device=self.device)], dim=1
        ).contiguous()  # (N, 4)
        z_batch = torch.bmm(
            w2c, means_h.T.unsqueeze(0).expand(B, -1, -1).contiguous(),
        )[:, 2, :].clamp(min=NEAR_PLANE).contiguous()  # (B, N)

        depth, alpha = [], []
        for bi in range(B):
            dc = z_batch[bi].view(-1, 1, 1).expand(-1, 1, 3).contiguous()  # (N,1,3)
            dr, da, _ = rasterization(
                means=g["means"], quats=g["quats"], scales=g["scales"],
                opacities=g["opacities"], colors=dc,
                viewmats=w2c[bi:bi + 1], Ks=K_batch[bi:bi + 1],
                width=width, height=height, sh_degree=0,
                near_plane=NEAR_PLANE, far_plane=FAR_PLANE,
            )
            raw = dr[0, :, :, 0].cpu().numpy().astype(np.float64)        # (H,W)
            A   = da[0, :, :, 0].cpu().numpy().astype(np.float64)        # (H,W)
            # Σᵢwᵢzᵢ, with the 0.5·A composited-background term removed.
            wsum = (raw - 0.5 * A) / SH_C0
            dm = np.where(A > ALPHA_EPS, wsum / np.maximum(A, ALPHA_EPS), 0.0)
            depth.append(np.maximum(dm, 0.0).astype(np.float32))
            alpha.append(A.astype(np.float32))
        return depth, alpha
