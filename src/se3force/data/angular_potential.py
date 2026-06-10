from __future__ import annotations

import torch
from torch.utils.data import Dataset

from se3force.geometry.rotations import apply_rotation, random_rotation_matrix, random_translation
from se3force.geometry.transforms import apply_transform


def angular_energy(
    x: torch.Tensor,
    mass: torch.Tensor,
    lambda_angle: float = 0.35,
    sigma2: float = 1.2,
    sigma3: float = 1.7,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Invariant two- and three-body energy for a single sample."""
    N = x.shape[0]
    energy = x.new_zeros(())
    for i in range(N):
        for j in range(i + 1, N):
            rij = x[j] - x[i]
            energy = energy + mass[i] * mass[j] * torch.exp(-(rij @ rij) / (sigma2 * sigma2))

    for i in range(N):
        for j in range(N):
            if j == i:
                continue
            for k in range(j + 1, N):
                if k == i:
                    continue
                rij = x[j] - x[i]
                rik = x[k] - x[i]
                dij = rij.norm()
                dik = rik.norm()
                cos_theta = (rij @ rik) / (dij * dik + eps)
                p2 = 0.5 * (3.0 * cos_theta * cos_theta - 1.0)
                radial = torch.exp(-((rij @ rij) + (rik @ rik)) / (sigma3 * sigma3))
                energy = energy + lambda_angle * mass[i] * mass[j] * mass[k] * radial * p2
    return energy


def angular_force(
    x: torch.Tensor,
    mass: torch.Tensor,
    lambda_angle: float = 0.35,
    sigma2: float = 1.2,
    sigma3: float = 1.7,
) -> torch.Tensor:
    x_req = x.detach().clone().requires_grad_(True)
    energy = angular_energy(x_req, mass, lambda_angle=lambda_angle, sigma2=sigma2, sigma3=sigma3)
    grad = torch.autograd.grad(energy, x_req, create_graph=False)[0]
    return -grad.detach()


class AngularPotentialDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 256,
        num_nodes: int = 8,
        position_scale: float = 1.0,
        mass_range: tuple[float, float] | list[float] = (0.5, 1.5),
        lambda_angle: float = 0.35,
        sigma2: float = 1.2,
        sigma3: float = 1.7,
        seed: int = 0,
        random_rotate_translate: bool = False,
    ) -> None:
        self.samples = []
        gen = torch.Generator().manual_seed(seed)
        lo, hi = float(mass_range[0]), float(mass_range[1])
        for _ in range(num_samples):
            x = position_scale * torch.randn(num_nodes, 3, generator=gen)
            mass = lo + (hi - lo) * torch.rand(num_nodes, generator=gen)
            force = angular_force(x, mass, lambda_angle=lambda_angle, sigma2=sigma2, sigma3=sigma3)
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
