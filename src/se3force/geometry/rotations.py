import torch


def random_rotation_matrix(batch: int, device=None, dtype=None) -> torch.Tensor:
    """Sample proper 3D rotation matrices with shape [B, 3, 3]."""
    dtype = dtype or torch.get_default_dtype()
    q = torch.randn(batch, 4, device=device, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=-1)
    two = torch.tensor(2.0, device=device, dtype=dtype)

    R = torch.empty(batch, 3, 3, device=device, dtype=dtype)
    R[:, 0, 0] = 1 - two * (y * y + z * z)
    R[:, 0, 1] = two * (x * y - z * w)
    R[:, 0, 2] = two * (x * z + y * w)
    R[:, 1, 0] = two * (x * y + z * w)
    R[:, 1, 1] = 1 - two * (x * x + z * z)
    R[:, 1, 2] = two * (y * z - x * w)
    R[:, 2, 0] = two * (x * z - y * w)
    R[:, 2, 1] = two * (y * z + x * w)
    R[:, 2, 2] = 1 - two * (x * x + y * y)
    return R


def random_translation(batch: int, scale: float = 1.0, device=None, dtype=None) -> torch.Tensor:
    """Sample translations with shape [B, 1, 3]."""
    dtype = dtype or torch.get_default_dtype()
    return scale * torch.randn(batch, 1, 3, device=device, dtype=dtype)


def apply_rotation(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply batched rotations to point or vector arrays shaped [B, N, 3]."""
    return torch.einsum("bij,bnj->bni", R, x)
