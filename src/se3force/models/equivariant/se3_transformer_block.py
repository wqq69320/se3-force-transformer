from __future__ import annotations

from e3nn import o3
from torch import nn

from .gate import EquivariantGate
from .irrep_norm import IrrepNorm
from .se3_attention import SE3MultiHeadAttention
from .tfn_conv import TFNConv


class SE3TransformerBlock(nn.Module):
    def __init__(
        self,
        irreps: o3.Irreps | str,
        lmax: int = 2,
        num_heads: int = 2,
        num_query_channels: int = 8,
        radial_num_basis: int = 16,
        radial_hidden_dim: int = 64,
        dropout: float = 0.0,
        use_attention: bool = True,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.norm1 = IrrepNorm(self.irreps)
        if use_attention:
            self.message = SE3MultiHeadAttention(
                irreps_in=self.irreps,
                irreps_value=self.irreps,
                irreps_out=self.irreps,
                num_heads=num_heads,
                lmax=lmax,
                num_query_channels=num_query_channels,
                radial_hidden_dim=radial_hidden_dim,
                radial_num_basis=radial_num_basis,
                attention_dropout=dropout,
            )
        else:
            self.message = TFNConv(
                irreps_in=self.irreps,
                irreps_out=self.irreps,
                lmax=lmax,
                radial_num_basis=radial_num_basis,
                radial_hidden_dim=radial_hidden_dim,
            )
        self.norm2 = IrrepNorm(self.irreps)
        self.ff1 = o3.Linear(self.irreps, self.irreps)
        self.activation = EquivariantGate(self.irreps) if use_gate else nn.Identity()
        self.ff2 = o3.Linear(self.irreps, self.irreps)

    def forward(self, x, features):
        features = features + self.message(x, self.norm1(features))
        features = features + self.ff2(self.activation(self.ff1(self.norm2(features))))
        return features
