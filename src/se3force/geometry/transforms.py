import torch

from .rotations import apply_rotation


def apply_transform(x: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply x -> R x + t for x shaped [B, N, 3]."""
    return apply_rotation(x, R) + t
