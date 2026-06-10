from __future__ import annotations

import torch
from torch import nn

from se3force.models.common import make_mlp


class CoordMLP(nn.Module):
    """Absolute-coordinate MLP baseline. It is intentionally not equivariant."""

    def __init__(self, scalar_input_dim: int = 1, num_nodes: int = 6, hidden_dim: int = 128, num_layers: int = 3) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        input_dim = self.num_nodes * (3 + scalar_input_dim)
        self.net = make_mlp(input_dim, hidden_dim, self.num_nodes * 3, num_layers=num_layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.num_nodes:
            raise ValueError(f"CoordMLP was configured for {self.num_nodes} nodes, got {x.shape[1]}")
        h = torch.cat([x, z], dim=-1).reshape(x.shape[0], -1)
        return self.net(h).view(x.shape[0], self.num_nodes, 3)
