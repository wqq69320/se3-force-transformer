from __future__ import annotations

import torch
from e3nn import o3
from torch import nn


class IrrepNorm(nn.Module):
    """Equivariant affine rescaling by one learned scalar per irrep field."""

    def __init__(self, irreps: o3.Irreps | str) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.weights = nn.ParameterList()
        for mul, _ir in self.irreps:
            self.weights.append(nn.Parameter(torch.ones(mul)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        chunks = []
        start = 0
        for weight, (mul, ir) in zip(self.weights, self.irreps):
            dim = mul * ir.dim
            block = features[..., start : start + dim].reshape(*features.shape[:-1], mul, ir.dim)
            chunks.append((block * weight.view(*([1] * (block.ndim - 2)), mul, 1)).reshape(*features.shape[:-1], dim))
            start += dim
        return torch.cat(chunks, dim=-1)
