# Pure-PyTorch replacements for the pointnet2 CUDA extension used by 3DETR.
# Semantics match third_party/pointnet2/pointnet2_utils.py (erikwijmans/Pointnet2_PyTorch):
#   - furthest_point_sample starts from index 0 (deterministic, like the CUDA kernel)
#   - ball_query picks the first `nsample` points (in index order) within `radius`
#     and pads short rows by repeating the first found index
# No custom CUDA compilation needed -> works on aarch64 / Blackwell (GB10).
import torch
import torch.nn as nn


@torch.no_grad()
def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """xyz: (B, N, 3) -> (B, npoint) int32 indices."""
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros(B, npoint, dtype=torch.int64, device=device)
    dist = torch.full((B, N), 1e10, device=device, dtype=xyz.dtype)
    farthest = torch.zeros(B, dtype=torch.int64, device=device)
    batch = torch.arange(B, dtype=torch.int64, device=device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch, farthest, :].unsqueeze(1)  # (B,1,3)
        d = torch.sum((xyz - centroid) ** 2, dim=-1)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(dim=-1)
    return idx.to(torch.int32)


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features: (B, C, N), idx: (B, npoint) -> (B, C, npoint)."""
    idx = idx.long().unsqueeze(1).expand(-1, features.shape[1], -1)
    return torch.gather(features, 2, idx)


def grouping_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features: (B, C, N), idx: (B, npoint, nsample) -> (B, C, npoint, nsample)."""
    B, C, N = features.shape
    _, S, K = idx.shape
    idx = idx.long().view(B, 1, S * K).expand(-1, C, -1)
    out = torch.gather(features, 2, idx)
    return out.view(B, C, S, K)


@torch.no_grad()
def ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor,
               chunk: int = 512) -> torch.Tensor:
    """xyz: (B, N, 3) all points, new_xyz: (B, S, 3) query centers
    -> (B, S, nsample) int indices into xyz.
    Chunked over S to bound the (S x N) distance matrix memory."""
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    device = xyz.device
    r2 = radius * radius
    out = torch.empty(B, S, nsample, dtype=torch.int64, device=device)
    arange_n = torch.arange(N, dtype=torch.int64, device=device).view(1, 1, N)
    for s0 in range(0, S, chunk):
        s1 = min(s0 + chunk, S)
        q = new_xyz[:, s0:s1, :]  # (B, s, 3)
        d2 = torch.cdist(q, xyz, p=2.0) ** 2  # (B, s, N)
        gidx = arange_n.expand(B, s1 - s0, N).clone()
        gidx[d2 > r2] = N
        gidx = gidx.sort(dim=-1).values[:, :, :nsample]  # first-in-order within radius
        first = gidx[:, :, 0:1].expand(-1, -1, nsample)
        pad = gidx == N
        gidx[pad] = first[pad]
        out[:, s0:s1, :] = gidx
    return out


class QueryAndGroup(nn.Module):
    """Groups points within a ball around each query center (matches CUDA version API)."""

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False,
                 normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        super().__init__()
        assert not sample_uniformly and not ret_unique_cnt, \
            "sample_uniformly/ret_unique_cnt not needed by 3DETR"
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz

    def forward(self, xyz, new_xyz, features=None):
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()  # (B,3,N)
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B,3,S,K)
        grouped_xyz = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz = grouped_xyz / self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        return new_features


class GroupAll(nn.Module):
    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        super().__init__()
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz

    def forward(self, xyz, new_xyz, features=None):
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)  # (B,3,1,N)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        return new_features
