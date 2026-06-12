from __future__ import annotations

import torch
from e3nn import o3
from torch import nn

from se3force.geometry.irreps import build_hidden_irreps
from se3force.models.edge_graph import aggregate_to_nodes, dense_edges
from se3force.models.radial import GaussianRadialBasis

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
        if self.force_head.weight_numel == 0:
            self.force_radial_basis = GaussianRadialBasis(radial_num_basis)
            self.force_edge_mlp = nn.Sequential(
                nn.Linear(2 * self.irreps_hidden.dim + radial_num_basis, radial_hidden_dim),
                nn.SiLU(),
                nn.Linear(radial_hidden_dim, 1),
            )
        else:
            self.force_radial_basis = None
            self.force_edge_mlp = None

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
        if self.force_edge_mlp is None:
            return self.force_head(features)

        edges = dense_edges(x)
        hi = features[edges.batch, edges.dst]
        hj = features[edges.batch, edges.src]
        radial = self.force_radial_basis(edges.distances)
        weights = self.force_edge_mlp(torch.cat([hi, hj, radial], dim=-1))
        force_messages = weights * edges.edge_vec
        return aggregate_to_nodes(force_messages, edges) / max(1, x.shape[1] - 1)
