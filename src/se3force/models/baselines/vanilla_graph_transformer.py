from __future__ import annotations

import torch
from torch import nn


class VanillaGraphTransformer(nn.Module):
    """Standard node transformer over absolute coordinates and scalar attributes."""

    def __init__(
        self,
        scalar_input_dim: int = 1,
        hidden_dim: int = 96,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(3 + scalar_input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            activation="gelu",
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.embed(torch.cat([x, z], dim=-1))
        return self.head(self.encoder(h))
