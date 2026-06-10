from __future__ import annotations

import torch
from e3nn import o3
from torch import nn

from se3force.geometry.irreps import build_hidden_irreps

from .se3_transformer_block import SE3TransformerBlock


class SE3ForceTransformer(nn.Module):
    """Irrep-based SE(3)-Transformer for force-vector prediction."""

    def __init__(
        self,
        scalar_input_dim: int = 1,
        lmax: int = 2,
        channels_by_l: dict[int | str, int] | None = None,
        num_layers: int = 3,
        num_heads: int = 2,
        num_query_channels: int = 8,
        radial_num_basis: int = 16,
        radial_hidden_dim: int = 64,
        dropout: float = 0.0,
        use_attention: bool = True,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        self.irreps_hidden = build_hidden_irreps(lmax=lmax, channels_by_l=channels_by_l)
        self.scalar_input_dim = int(scalar_input_dim)
        self.scalar_slices: list[tuple[int, int]] = []
        start = 0
        scalar_dim = 0
        for mul, ir in self.irreps_hidden:
            dim = mul * ir.dim
            if ir.l == 0:
                self.scalar_slices.append((start, start + dim))
                scalar_dim += dim
            start += dim
        if scalar_dim <= 0:
            raise ValueError("SE3ForceTransformer requires at least one scalar hidden field")
        self.scalar_dim = scalar_dim
        self.scalar_embed = nn.Sequential(
            nn.Linear(self.scalar_input_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.blocks = nn.ModuleList(
            [
                SE3TransformerBlock(
                    irreps=self.irreps_hidden,
                    lmax=lmax,
                    num_heads=num_heads,
                    num_query_channels=num_query_channels,
                    radial_num_basis=radial_num_basis,
                    radial_hidden_dim=radial_hidden_dim,
                    dropout=dropout,
                    use_attention=use_attention,
                    use_gate=use_gate,
                )
                for _ in range(num_layers)
            ]
        )
        self.force_head = o3.Linear(self.irreps_hidden, o3.Irreps("1x1o"))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        features = x.new_zeros(B, N, self.irreps_hidden.dim)
        scalar_features = self.scalar_embed(z)
        offset = 0
        for start, end in self.scalar_slices:
            width = end - start
            features[..., start:end] = scalar_features[..., offset : offset + width]
            offset += width
        for block in self.blocks:
            features = block(x, features)
        return self.force_head(features)
