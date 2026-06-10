from __future__ import annotations

import torch
from torch import nn


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    activation: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(activation())
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


def build_model(config: dict) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    name = model_cfg.pop("name", "se3_transformer")
    if name == "coord_mlp":
        from .baselines.coord_mlp import CoordMLP

        return CoordMLP(**model_cfg)
    if name == "vanilla_gt":
        from .baselines.vanilla_graph_transformer import VanillaGraphTransformer

        return VanillaGraphTransformer(**model_cfg)
    if name == "egnn":
        from .baselines.egnn import EGNN

        return EGNN(**model_cfg)
    if name == "se3_transformer":
        from .equivariant.se3_force_transformer import SE3ForceTransformer

        return SE3ForceTransformer(**model_cfg)
    raise ValueError(f"unknown model: {name}")


def to_device(batch: dict[str, torch.Tensor], device: torch.device | str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}
