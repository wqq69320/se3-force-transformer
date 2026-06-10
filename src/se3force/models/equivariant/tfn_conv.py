from __future__ import annotations

import math

import torch
from e3nn import o3
from torch import nn

from se3force.geometry.irreps import spherical_harmonics_irreps
from se3force.models.edge_graph import aggregate_to_nodes, dense_edges
from se3force.models.radial import RadialMLP


class TFNConv(nn.Module):
    """Tensor Field Network convolution with radial tensor-product kernels."""

    def __init__(
        self,
        irreps_in: o3.Irreps | str,
        irreps_out: o3.Irreps | str,
        lmax: int = 2,
        radial_num_basis: int = 16,
        radial_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = spherical_harmonics_irreps(lmax)
        self.tp = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False)
        self.radial = RadialMLP(radial_num_basis, radial_hidden_dim, self.tp.weight_numel)

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        edges = dense_edges(x)
        sh = o3.spherical_harmonics(self.irreps_sh, edges.edge_vec, normalize=True, normalization="component")
        weights = self.radial(edges.distances)
        src_features = features[edges.batch, edges.src]
        messages = self.tp(src_features, sh, weights)
        out = aggregate_to_nodes(messages, edges)
        return out / math.sqrt(max(1, x.shape[1] - 1))
