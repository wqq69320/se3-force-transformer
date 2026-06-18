from __future__ import annotations

import math

import torch
from e3nn import o3
from torch import nn

from se3force.geometry.irreps import spherical_harmonics_irreps
from se3force.models.edge_graph import aggregate_to_nodes, dense_edges, edge_softmax
from se3force.models.radial import RadialMLP


class SE3AttentionHead(nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps | str,
        irreps_value: o3.Irreps | str,
        lmax: int = 2,
        num_query_channels: int = 8,
        radial_hidden_dim: int = 64,
        radial_num_basis: int = 16,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_value = o3.Irreps(irreps_value)
        self.irreps_key = o3.Irreps(f"{num_query_channels}x0e")
        self.irreps_sh = spherical_harmonics_irreps(lmax)
        self.query = o3.Linear(self.irreps_in, self.irreps_key)
        self.key_tp = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_key, shared_weights=False)
        self.value_tp = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_value, shared_weights=False)
        self.key_radial = RadialMLP(radial_num_basis, radial_hidden_dim, self.key_tp.weight_numel)
        self.value_radial = RadialMLP(radial_num_basis, radial_hidden_dim, self.value_tp.weight_numel)
        self.bias_radial = RadialMLP(radial_num_basis, radial_hidden_dim, 1)
        self.dropout = nn.Dropout(attention_dropout)
        self.num_query_channels = int(num_query_channels)

    def forward(self, x: torch.Tensor, features: torch.Tensor, edges=None) -> torch.Tensor:
        edges = dense_edges(x) if edges is None else edges
        q = self.query(features)
        sh = o3.spherical_harmonics(self.irreps_sh, edges.edge_vec, normalize=True, normalization="component")
        src_features = features[edges.batch, edges.src]
        key = self.key_tp(src_features, sh, self.key_radial(edges.distances))
        value = self.value_tp(src_features, sh, self.value_radial(edges.distances))
        q_dst = q[edges.batch, edges.dst]
        logits = (q_dst * key).sum(dim=-1, keepdim=True) / math.sqrt(self.num_query_channels)
        logits = logits + self.bias_radial(edges.distances)
        alpha = edge_softmax(logits, edges)
        if self.training:
            alpha = self.dropout(alpha)
        out = aggregate_to_nodes(alpha * value, edges)
        return out


class SE3MultiHeadAttention(nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps | str,
        irreps_value: o3.Irreps | str,
        irreps_out: o3.Irreps | str,
        num_heads: int = 2,
        lmax: int = 2,
        num_query_channels: int = 8,
        radial_hidden_dim: int = 64,
        radial_num_basis: int = 16,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.irreps_value = o3.Irreps(irreps_value)
        self.irreps_out = o3.Irreps(irreps_out)
        self.heads = nn.ModuleList(
            [
                SE3AttentionHead(
                    irreps_in=irreps_in,
                    irreps_value=self.irreps_value,
                    lmax=lmax,
                    num_query_channels=num_query_channels,
                    radial_hidden_dim=radial_hidden_dim,
                    radial_num_basis=radial_num_basis,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_heads)
            ]
        )
        concat_irreps = o3.Irreps("+".join(str(self.irreps_value) for _ in range(num_heads)))
        self.mix = o3.Linear(concat_irreps, self.irreps_out)

    def forward(self, x: torch.Tensor, features: torch.Tensor, edges=None) -> torch.Tensor:
        return self.mix(torch.cat([head(x, features, edges=edges) for head in self.heads], dim=-1))
