from __future__ import annotations

import torch
from torch.utils.data import Dataset

from se3force.geometry.rotations import apply_rotation, random_rotation_matrix, random_translation
from se3force.geometry.transforms import apply_transform


def central_force(x: torch.Tensor, mass: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Compute inverse-square central forces for one sample or a batch."""
    single = x.ndim == 2
    if single:
        x = x.unsqueeze(0)
        mass = mass.unsqueeze(0)

    r = x[:, None, :, :] - x[:, :, None, :]
    d2 = (r * r).sum(dim=-1)
    B, N, _ = x.shape
    eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
    pair_mass = mass[:, :, None] * mass[:, None, :]
    scale = pair_mass / (d2 + eps).pow(1.5)
    scale = scale.masked_fill(eye, 0.0)
    force = (scale.unsqueeze(-1) * r).sum(dim=2)
    return force.squeeze(0) if single else force


class CentralForceDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 256,
        num_nodes: int = 8,
        position_scale: float = 1.0,
        anisotropic: bool = True,
        mass_range: tuple[float, float] | list[float] = (0.5, 1.5),
        seed: int = 0,
        random_rotate_translate: bool = False,
    ) -> None:
        self.samples = []
        gen = torch.Generator().manual_seed(seed)
        lo, hi = float(mass_range[0]), float(mass_range[1])
        for _ in range(num_samples):
            x = position_scale * torch.randn(num_nodes, 3, generator=gen)
            if anisotropic:
                x = x * torch.tensor([1.0, 0.7, 1.35])
            mass = lo + (hi - lo) * torch.rand(num_nodes, generator=gen)
            force = central_force(x, mass)
            if random_rotate_translate:
                R = random_rotation_matrix(1, dtype=x.dtype)
                t = random_translation(1, dtype=x.dtype)
                x = apply_transform(x.unsqueeze(0), R, t).squeeze(0)
                force = apply_rotation(force.unsqueeze(0), R).squeeze(0)
            self.samples.append({"x": x, "z": mass[:, None], "force": force})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]
