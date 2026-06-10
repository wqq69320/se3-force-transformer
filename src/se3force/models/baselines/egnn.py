from __future__ import annotations

import torch
from torch import nn

from se3force.models.edge_graph import aggregate_to_nodes, dense_edges


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        edges = dense_edges(x)
        hi = h[edges.batch, edges.dst]
        hj = h[edges.batch, edges.src]
        d2 = (edges.edge_vec * edges.edge_vec).sum(dim=-1, keepdim=True)
        msg = self.edge_mlp(torch.cat([hi, hj, d2], dim=-1))
        agg = aggregate_to_nodes(msg, edges)
        return h + self.node_mlp(torch.cat([h, agg], dim=-1))


class EGNN(nn.Module):
    """Equivariant scalar-message baseline with relative-vector force readout."""

    def __init__(self, scalar_input_dim: int = 1, hidden_dim: int = 96, num_layers: int = 3) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(scalar_input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])
        self.force_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.embed(z)
        for layer in self.layers:
            h = layer(x, h)
        edges = dense_edges(x)
        hi = h[edges.batch, edges.dst]
        hj = h[edges.batch, edges.src]
        d2 = (edges.edge_vec * edges.edge_vec).sum(dim=-1, keepdim=True)
        weights = self.force_mlp(torch.cat([hi, hj, d2], dim=-1))
        messages = weights * edges.edge_vec
        return aggregate_to_nodes(messages, edges)
