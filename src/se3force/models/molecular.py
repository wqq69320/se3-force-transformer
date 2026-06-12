from __future__ import annotations

import torch
from torch import nn

from se3force.models.molecular_graph import aggregate_to_padded_nodes, build_molecular_graph, graph_stats
from se3force.models.radial import GaussianRadialBasis


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].mean() if mask.any() else values.new_zeros(())


class ScalarCutoffLayer(nn.Module):
    def __init__(self, hidden_dim: int, radial_num_basis: int, radial_hidden_dim: int) -> None:
        super().__init__()
        self.radial = GaussianRadialBasis(radial_num_basis)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + radial_num_basis, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, graph) -> torch.Tensor:
        if graph.edge_count == 0:
            return h
        hi = h[graph.batch, graph.dst]
        hj = h[graph.batch, graph.src]
        edge_attr = self.radial(graph.distances)
        messages = self.edge_mlp(torch.cat([hi, hj, edge_attr], dim=-1))
        agg = aggregate_to_padded_nodes(messages, graph)
        return h + self.node_mlp(torch.cat([h, agg], dim=-1))


class MolecularScalarForceField(nn.Module):
    """Cutoff scalar-message molecular force field with equivariant vector readout."""

    def __init__(
        self,
        scalar_input_dim: int = 1,
        hidden_dim: int = 96,
        num_layers: int = 3,
        cutoff_radius: float = 5.0,
        max_neighbors: int | None = None,
        max_atomic_number: int = 100,
        radial_num_basis: int = 16,
        radial_hidden_dim: int = 64,
        training_mode: str = "direct_force",
        model_label: str = "molecular_scalar",
        **_unused,
    ) -> None:
        super().__init__()
        self.cutoff_radius = float(cutoff_radius)
        self.max_neighbors = max_neighbors
        self.training_mode = training_mode
        self.model_label = model_label
        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.layers = nn.ModuleList(
            [ScalarCutoffLayer(hidden_dim, radial_num_basis, radial_hidden_dim) for _ in range(num_layers)]
        )
        self.radial = GaussianRadialBasis(radial_num_basis)
        self.force_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + radial_num_basis, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, 1),
        )
        self.node_energy = nn.Sequential(nn.Linear(hidden_dim, radial_hidden_dim), nn.SiLU(), nn.Linear(radial_hidden_dim, 1))
        self.pair_energy = nn.Sequential(
            nn.Linear(2 * hidden_dim + radial_num_basis, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, 1),
        )

    def encode(self, z: torch.Tensor, mask: torch.Tensor, graph):
        h = self.atom_embedding(z.clamp_min(0))
        h = h * mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, graph) * mask.unsqueeze(-1)
        return h

    def invariant_energy(self, h: torch.Tensor, graph, mask: torch.Tensor) -> torch.Tensor:
        node_e = self.node_energy(h).squeeze(-1).masked_fill(~mask, 0.0).sum(dim=1, keepdim=True)
        if graph.edge_count == 0:
            return node_e
        hi = h[graph.batch, graph.dst]
        hj = h[graph.batch, graph.src]
        radial = self.radial(graph.distances)
        pair_e = self.pair_energy(torch.cat([hi, hj, radial], dim=-1)).squeeze(-1)
        pair_sum = h.new_zeros(graph.num_batches, 1)
        pair_sum.index_add_(0, graph.batch, pair_e.unsqueeze(-1))
        return node_e + 0.5 * pair_sum

    def direct_forces(self, pos: torch.Tensor, h: torch.Tensor, graph, mask: torch.Tensor) -> torch.Tensor:
        if graph.edge_count == 0:
            return pos.new_zeros(pos.shape)
        hi = h[graph.batch, graph.dst]
        hj = h[graph.batch, graph.src]
        radial = self.radial(graph.distances)
        weights = self.force_mlp(torch.cat([hi, hj, radial], dim=-1))
        messages = weights * graph.edge_vec
        forces = aggregate_to_padded_nodes(messages, graph)
        return forces * mask.unsqueeze(-1)

    def forward(self, pos: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        if mask is None:
            mask = torch.ones(pos.shape[:2], dtype=torch.bool, device=pos.device)
        if self.training_mode == "energy_force" and not pos.requires_grad:
            pos = pos.requires_grad_(True)
        graph = build_molecular_graph(pos, mask, cutoff_radius=self.cutoff_radius, max_neighbors=self.max_neighbors)
        h = self.encode(z, mask, graph)
        energy = self.invariant_energy(h, graph, mask)
        if self.training_mode == "energy_force":
            grad = torch.autograd.grad(
                energy.sum(),
                pos,
                create_graph=self.training,
                retain_graph=True,
                allow_unused=False,
            )[0]
            forces = -grad * mask.unsqueeze(-1)
        else:
            forces = self.direct_forces(pos, h, graph, mask)
        return {
            "forces": forces,
            "energy": energy,
            "graph_stats": graph_stats(graph, self.cutoff_radius),
        }


class MolecularEGNN(MolecularScalarForceField):
    def __init__(self, **kwargs) -> None:
        super().__init__(model_label="egnn", **kwargs)


class MolecularTFNConv(MolecularScalarForceField):
    def __init__(self, **kwargs) -> None:
        super().__init__(model_label="tfn", **kwargs)


class MolecularSE3ForceTransformer(MolecularScalarForceField):
    def __init__(self, channels_by_l: dict | None = None, lmax: int = 2, num_query_channels: int = 8, **kwargs) -> None:
        channels_by_l = channels_by_l or {0: kwargs.pop("hidden_dim", 96)}
        hidden_dim = int(channels_by_l.get(0, channels_by_l.get("0", 96)))
        super().__init__(hidden_dim=hidden_dim, model_label=f"se3_l{lmax}", **kwargs)
        self.lmax = lmax
        self.num_query_channels = num_query_channels


def build_molecular_model(config: dict) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    dataset_cfg = config.get("dataset", {})
    name = model_cfg.pop("name", "se3_transformer")
    model_cfg.setdefault("cutoff_radius", dataset_cfg.get("cutoff_radius", 5.0))
    model_cfg.setdefault("max_neighbors", dataset_cfg.get("max_neighbors"))
    training_mode = str(config.get("training", {}).get("mode", model_cfg.pop("training_mode", "direct_force")))
    model_cfg["training_mode"] = training_mode
    if name in {"egnn", "molecular_egnn"}:
        return MolecularEGNN(**model_cfg)
    if name in {"tfn", "baseline_tfn", "molecular_tfn"} or model_cfg.get("use_attention") is False:
        return MolecularTFNConv(**model_cfg)
    if name in {"se3_transformer", "molecular_se3"}:
        return MolecularSE3ForceTransformer(**model_cfg)
    raise ValueError(f"unknown molecular model: {name}")
