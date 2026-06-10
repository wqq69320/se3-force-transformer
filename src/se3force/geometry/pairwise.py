import torch


def pairwise_relative(x: torch.Tensor) -> torch.Tensor:
    """Return r_ij = x_j - x_i with shape [B, N, N, 3]."""
    return x[:, None, :, :] - x[:, :, None, :]


def pairwise_dist2(x: torch.Tensor) -> torch.Tensor:
    r = pairwise_relative(x)
    return (r * r).sum(dim=-1)


def pairwise_dist(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return pairwise_dist2(x).clamp_min(eps).sqrt()


def unit_vectors(r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return r / r.norm(dim=-1, keepdim=True).clamp_min(eps)
