from __future__ import annotations

import torch
from e3nn import o3
from torch import nn
from torch.nn import functional as F


class EquivariantGate(nn.Module):
    """Scalar-driven gates for non-scalar irreps plus SiLU on scalar irreps."""

    def __init__(self, irreps: o3.Irreps | str) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.scalar_dim = sum(mul * ir.dim for mul, ir in self.irreps if ir.l == 0)
        self.num_gated_fields = sum(mul for mul, ir in self.irreps if ir.l > 0)
        if self.num_gated_fields > 0 and self.scalar_dim > 0:
            self.gate_net = nn.Linear(self.scalar_dim, self.num_gated_fields)
        elif self.num_gated_fields > 0:
            self.constant_gates = nn.Parameter(torch.zeros(self.num_gated_fields))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scalar_blocks = []
        start = 0
        for mul, ir in self.irreps:
            dim = mul * ir.dim
            if ir.l == 0:
                scalar_blocks.append(features[..., start : start + dim])
            start += dim

        if scalar_blocks:
            scalar_features = torch.cat(scalar_blocks, dim=-1)
        else:
            scalar_features = features.new_zeros(*features.shape[:-1], 0)

        if self.num_gated_fields > 0:
            if self.scalar_dim > 0:
                gates = torch.sigmoid(self.gate_net(scalar_features))
            else:
                gates = torch.sigmoid(self.constant_gates).expand(*features.shape[:-1], self.num_gated_fields)
        else:
            gates = None

        chunks = []
        start = 0
        gate_start = 0
        for mul, ir in self.irreps:
            dim = mul * ir.dim
            block = features[..., start : start + dim]
            if ir.l == 0:
                chunks.append(F.silu(block))
            else:
                gate = gates[..., gate_start : gate_start + mul].unsqueeze(-1)
                block = block.reshape(*features.shape[:-1], mul, ir.dim)
                chunks.append((block * gate).reshape(*features.shape[:-1], dim))
                gate_start += mul
            start += dim
        return torch.cat(chunks, dim=-1)
