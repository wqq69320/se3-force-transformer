from __future__ import annotations

import torch
from torch import nn

from .common import make_mlp


class GaussianRadialBasis(nn.Module):
    def __init__(self, num_basis: int = 16, cutoff: float = 5.0) -> None:
        super().__init__()
        self.num_basis = int(num_basis)
        self.cutoff = float(cutoff)
        centers = torch.linspace(0.0, cutoff, num_basis)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / (centers[1] - centers[0]).clamp_min(1e-6).item() ** 2 if num_basis > 1 else 1.0

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.unsqueeze(-1).clamp_min(0.0)
        return torch.exp(-self.gamma * (d - self.centers) ** 2)


class RadialMLP(nn.Module):
    def __init__(
        self,
        num_basis: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        self.basis = GaussianRadialBasis(num_basis=num_basis, cutoff=cutoff)
        self.net = make_mlp(num_basis, hidden_dim, output_dim, num_layers=num_layers)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return self.net(self.basis(distances))
