"""
Renderer interface + shared Gaussian preparation.

A backend takes the raw arrays loaded from a .ply (means, quats wxyz, log-scales,
logit-opacities, SH coeffs) and renders a batch of camera views to RGB + depth.

Conventions (shared by every backend — must match `render_cameras.py`):
  • quaternions wxyz
  • poses are c2w (OpenCV: X right, Y down, Z forward); the backend receives w2c
  • intrinsics K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
  • depth is camera-space Z in world units, 0 where nothing was hit
  • depth is ALPHA-NORMALIZED expected depth, E[z] = (Σ wᵢ zᵢ) / A, not the raw
    alpha-weighted sum — see `render_depth`'s docstring for why this matters
"""

from __future__ import annotations

import numpy as np
import torch

# Rasterization clip planes (shared so every backend matches).
NEAR_PLANE = 0.01
FAR_PLANE = 1000.0
# Degree-0 spherical-harmonic constant, used by the depth-as-color trick.
SH_C0 = 0.28209479177387814
# Accumulated alpha below which a pixel is treated as "nothing hit" — the
# divisor in the E[z] = Σwᵢzᵢ / A normalization is meaningless below this, and
# the depth it would produce is pure noise amplified by 1/A.
ALPHA_EPS = 1e-3


class Renderer:
    """Base class for a Gaussian-Splat rasterizer backend.

    Subclasses set ``name`` and ``device`` and implement ``render_rgb`` /
    ``render_depth``. ``prepare`` is shared: it applies the standard activations
    (exp on scales, sigmoid on opacities) and moves tensors onto the backend device.
    """

    name: str = "base"
    device: torch.device

    def _to_device(self, arrays: dict) -> dict:
        """Move raw .ply arrays onto the backend device, WITHOUT activations.

        Activations (exp on scales, sigmoid on opacity) differ per backend —
        gsplat (1.5.3) wants exp'd scales, gsplat-mps (0.1.x) wants raw log-scales
        with glob_scale=1 — so they are applied in each backend's ``prepare``.

        ``arrays`` keys: means (N,3), quats (N,4 wxyz), scales (N,3 log),
        opacities (N, logit), sh_coeffs (N,K,3), sh_degree (int).
        """
        d = self.device
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=d)
        return {
            "means":           t(arrays["means"]),
            "quats":           t(arrays["quats"]),
            "log_scales":      t(arrays["scales"]),      # raw log-scales
            "logit_opacities": t(arrays["opacities"]),   # raw logit opacities
            "sh":              t(arrays["sh_coeffs"]),
            "sh_degree":       int(arrays["sh_degree"]),
            "device":          d,
        }

    def prepare(self, arrays: dict) -> dict:
        """Return render-ready tensors on the backend device. Backends override
        to apply their activation convention; base just moves to device."""
        return self._to_device(arrays)

    def render_rgb(self, g: dict, w2c: torch.Tensor, K: torch.Tensor,
                   width: int, height: int) -> np.ndarray:
        """Render a batch of B views to RGB. Returns (B, H, W, 3) uint8."""
        raise NotImplementedError

    def render_depth(self, g: dict, w2c: torch.Tensor, K: torch.Tensor,
                     width: int, height: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Render depth for a batch of B views.

        Returns (depths, alphas): two lists of B (H, W) float32 arrays.
        `depths` is camera-space Z in world units, 0 where nothing was hit.
        `alphas` is the accumulated alpha (surface coverage) at each pixel,
        in [0, 1].

        Depth MUST be alpha-normalized — the expected depth E[z] = Σᵢwᵢzᵢ / A,
        where A = Σᵢwᵢ is the accumulated alpha. A rasterizer composites the
        alpha-WEIGHTED sum Σᵢwᵢzᵢ, which underestimates depth by exactly the
        factor A at every partially-covered pixel. This is not a rounding
        detail: measured on this project's own splat, ~18% of rendered frames
        had over half their pixels destroyed by the un-normalized version, and
        even 99%-opaque pixels came out 17.7% too shallow. Since
        `pipeline.py` derives object SIZE as pixel_size × depth, a shallow
        depth yields both a mislocated and an undersized 3D box.

        `alphas` is returned (not just used internally) because the pipeline
        needs it downstream: a 2D detection whose interior has near-zero
        accumulated alpha is, by construction, a detection on empty space.
        """
        raise NotImplementedError
