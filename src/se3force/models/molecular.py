from __future__ import annotations

import math

import torch
from e3nn import o3
from torch import nn

from se3force.geometry.irreps import build_hidden_irreps, spherical_harmonics_irreps
from se3force.models.equivariant.se3_force_transformer import SE3ForceTransformer
from se3force.models.molecular_graph import aggregate_to_padded_nodes, build_molecular_graph, graph_stats
from se3force.models.radial import GaussianRadialBasis


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].mean() if mask.any() else values.new_zeros(())


def edge_softmax(logits: torch.Tensor, graph) -> torch.Tensor:
    """Softmax over incoming edges for each padded destination node."""
    out = torch.empty_like(logits)
    node_index = graph.batch * graph.num_nodes + graph.dst
    for index in node_index.unique():
        mask = node_index == index
        out[mask] = torch.softmax(logits[mask], dim=0)
    return out


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, activation=nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = int(input_dim)
    for _ in range(max(0, int(num_layers))):
        layers.extend([nn.Linear(dim, int(hidden_dim)), activation()])
        dim = int(hidden_dim)
    layers.append(nn.Linear(dim, int(output_dim)))
    return nn.Sequential(*layers)


def has_non_scalar_irrep(irreps: o3.Irreps | str) -> bool:
    return any(ir.l > 0 for _, ir in o3.Irreps(irreps))


def directed_pair_indices(num_atoms: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = torch.arange(num_atoms, device=device)
    dst_grid, src_grid = torch.meshgrid(nodes, nodes, indexing="ij")
    keep = dst_grid != src_grid
    return dst_grid[keep], src_grid[keep]


def pair_index_from_z(z_i: torch.Tensor, z_j: torch.Tensor, max_atomic_number: int) -> torch.Tensor:
    z_i = z_i.clamp(0, max_atomic_number)
    z_j = z_j.clamp(0, max_atomic_number)
    return z_i * (max_atomic_number + 1) + z_j


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
        graph_mode: str = "cutoff",
        max_atomic_number: int = 100,
        radial_num_basis: int = 16,
        radial_hidden_dim: int = 64,
        use_atom_pair_embedding: bool = False,
        atom_embedding_dim: int | None = None,
        pair_embedding_dim: int = 16,
        edge_mlp_hidden_dim: int | None = None,
        edge_mlp_layers: int = 1,
        training_mode: str = "direct_force",
        model_label: str = "molecular_scalar",
        backbone_class: str = "scalar_cutoff",
        lmax: int | None = None,
        hidden_irreps: str = "scalar_only",
        use_attention: bool = False,
        use_gate: bool = False,
        pair_skip: bool = False,
        energy_output_scale: float = 1.0,
        energy_output_shift: float = 0.0,
        output_head_init_scale: float = 1.0,
        force_output_scale: float = 1.0,
        learnable_force_output_scale: bool = False,
        initial_force_output_scale: float | None = None,
        force_output_scale_regularization: float = 0.0,
        **_unused,
    ) -> None:
        super().__init__()
        self.cutoff_radius = float(cutoff_radius)
        self.max_neighbors = max_neighbors
        self.graph_mode = str(graph_mode)
        self.training_mode = training_mode
        self.model_label = model_label
        self.backbone_class = backbone_class
        self.lmax = lmax
        self.hidden_irreps = hidden_irreps
        self.use_attention = bool(use_attention)
        self.use_gate = bool(use_gate)
        self.pair_skip = bool(pair_skip)
        self.energy_output_scale = float(energy_output_scale)
        self.energy_output_shift = float(energy_output_shift)
        self.output_head_init_scale = float(output_head_init_scale)
        self.force_output_scale = float(force_output_scale)
        self.learnable_force_output_scale = bool(learnable_force_output_scale)
        self.initial_force_output_scale = float(initial_force_output_scale if initial_force_output_scale is not None else force_output_scale)
        self.force_output_scale_regularization = float(force_output_scale_regularization)
        self.hidden_dim = int(atom_embedding_dim or hidden_dim)
        self.num_layers = int(num_layers)
        self.radial_num_basis = int(radial_num_basis)
        self.radial_hidden_dim = int(radial_hidden_dim)
        self.edge_mlp_hidden_dim = int(edge_mlp_hidden_dim or radial_hidden_dim)
        self.edge_mlp_layers = int(edge_mlp_layers)
        self.max_atomic_number = int(max_atomic_number)
        self.use_atom_pair_embedding = bool(use_atom_pair_embedding)
        self.pair_embedding_dim = int(pair_embedding_dim) if self.use_atom_pair_embedding else 0
        self.atom_embedding = nn.Embedding(max_atomic_number + 1, self.hidden_dim)
        self.pair_embedding = (
            nn.Embedding((self.max_atomic_number + 1) * (self.max_atomic_number + 1), self.pair_embedding_dim)
            if self.use_atom_pair_embedding
            else None
        )
        self.layers = nn.ModuleList(
            [ScalarCutoffLayer(self.hidden_dim, radial_num_basis, radial_hidden_dim) for _ in range(num_layers)]
        )
        self.radial = GaussianRadialBasis(radial_num_basis)
        self.edge_feature_dim = 2 * self.hidden_dim + radial_num_basis + self.pair_embedding_dim
        self.force_mlp = make_mlp(self.edge_feature_dim, self.edge_mlp_hidden_dim, 1, self.edge_mlp_layers)
        self.node_energy = nn.Sequential(nn.Linear(self.hidden_dim, radial_hidden_dim), nn.SiLU(), nn.Linear(radial_hidden_dim, 1))
        self.pair_energy = make_mlp(self.edge_feature_dim, self.edge_mlp_hidden_dim, 1, self.edge_mlp_layers)
        self.attention_mlp = (
            nn.Sequential(
                nn.Linear(self.edge_feature_dim, radial_hidden_dim),
                nn.SiLU(),
                nn.Linear(radial_hidden_dim, 1),
            )
            if self.use_attention
            else None
        )
        self.force_gate = (
            nn.Sequential(nn.Linear(self.hidden_dim, radial_hidden_dim), nn.SiLU(), nn.Linear(radial_hidden_dim, 1))
            if self.use_gate
            else None
        )
        self.energy_gate = (
            nn.Sequential(nn.Linear(self.hidden_dim, radial_hidden_dim), nn.SiLU(), nn.Linear(radial_hidden_dim, 1))
            if self.use_gate
            else None
        )
        self.pair_skip_mlp = nn.Linear(radial_num_basis, 1) if self.pair_skip else None
        if self.learnable_force_output_scale:
            initial = max(abs(self.initial_force_output_scale), 1e-12)
            self.force_output_log_scale = nn.Parameter(torch.tensor(math.log(initial), dtype=torch.float32))
        else:
            self.force_output_log_scale = None
        self.architecture_signature = self._architecture_signature()
        self._apply_output_head_init_scale()

    def _architecture_signature(self) -> str:
        return "|".join(
            [
                self.model_label,
                f"model_class={self.__class__.__name__}",
                f"backbone={self.backbone_class}",
                f"hidden={self.hidden_dim}",
                f"layers={self.num_layers}",
                f"graph_mode={self.graph_mode}",
                f"radial_basis={self.radial_num_basis}",
                f"radial_hidden={self.radial_hidden_dim}",
                f"edge_hidden={self.edge_mlp_hidden_dim}",
                f"edge_layers={self.edge_mlp_layers}",
                f"lmax={self.lmax if self.lmax is not None else 'n/a'}",
                f"hidden_irreps={self.hidden_irreps}",
                f"attention={self.use_attention}",
                f"gate={self.use_gate}",
                f"pair_skip={self.pair_skip}",
                f"atom_pair_embedding={self.use_atom_pair_embedding}",
                f"pair_embedding_dim={self.pair_embedding_dim}",
                f"energy_scale={self.energy_output_scale:.6g}",
                f"force_head_scale={self.output_head_init_scale:.6g}",
                f"force_output_scale={self.force_output_scale:.6g}",
                f"learnable_force_output_scale={self.learnable_force_output_scale}",
                f"initial_force_output_scale={self.initial_force_output_scale:.6g}",
            ]
        )

    def _apply_output_head_init_scale(self) -> None:
        if self.output_head_init_scale == 1.0:
            return
        final = self.force_mlp[-1]
        if isinstance(final, nn.Linear):
            with torch.no_grad():
                final.weight.mul_(self.output_head_init_scale)
                if final.bias is not None:
                    final.bias.mul_(self.output_head_init_scale)

    def force_output_scale_value(self) -> torch.Tensor:
        if self.force_output_log_scale is not None:
            return self.force_output_log_scale.exp()
        return torch.tensor(self.force_output_scale, dtype=torch.float32, device=self.atom_embedding.weight.device)

    def pair_features(self, z: torch.Tensor, graph) -> torch.Tensor:
        if self.pair_embedding is None:
            return z.new_zeros(graph.edge_count, 0, dtype=self.atom_embedding.weight.dtype).to(device=z.device)
        zi = z[graph.batch, graph.dst].clamp(0, self.max_atomic_number)
        zj = z[graph.batch, graph.src].clamp(0, self.max_atomic_number)
        pair_index = zi * (self.max_atomic_number + 1) + zj
        return self.pair_embedding(pair_index)

    def edge_features(self, h: torch.Tensor, z: torch.Tensor, graph) -> torch.Tensor:
        hi = h[graph.batch, graph.dst]
        hj = h[graph.batch, graph.src]
        radial = self.radial(graph.distances)
        return torch.cat([hi, hj, radial, self.pair_features(z, graph).to(dtype=h.dtype)], dim=-1)

    def architecture_metadata(self) -> dict[str, object]:
        scalar_irreps = f"{self.hidden_dim}x0e"
        return {
            "actual_hidden_irreps": scalar_irreps,
            "irreps_in": f"{self.hidden_dim}x0e",
            "irreps_hidden": scalar_irreps,
            "irreps_out": "1x1o",
            "irreps_sh": "",
            "uses_non_scalar_hidden": False,
            "uses_spherical_harmonics_in_value": False,
            "force_head_type": "central_pair_scalar_mlp",
            "force_head_irreps": "1x1o",
            "uses_relative_vector_fallback": True,
            "uses_pairwise_force_skip": bool(self.pair_skip),
            "uses_atom_pair_embedding": bool(self.use_atom_pair_embedding),
            "atom_embedding_dim": int(self.hidden_dim),
            "pair_embedding_dim": int(self.pair_embedding_dim),
            "edge_mlp_hidden_dim": int(self.edge_mlp_hidden_dim),
            "edge_mlp_layers": int(self.edge_mlp_layers),
            "graph_mode": self.graph_mode,
        }

    def encode(self, z: torch.Tensor, mask: torch.Tensor, graph):
        h = self.atom_embedding(z.clamp_min(0))
        h = h * mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, graph) * mask.unsqueeze(-1)
        return h

    def invariant_energy(self, h: torch.Tensor, z: torch.Tensor, graph, mask: torch.Tensor) -> torch.Tensor:
        node_e = self.node_energy(h).squeeze(-1).masked_fill(~mask, 0.0).sum(dim=1, keepdim=True)
        if self.energy_gate is not None:
            gated = self.energy_gate(h).squeeze(-1).masked_fill(~mask, 0.0).sum(dim=1, keepdim=True)
            node_e = node_e + 0.1 * gated
        if graph.edge_count == 0:
            return node_e
        radial = self.radial(graph.distances)
        pair_e = self.pair_energy(self.edge_features(h, z, graph)).squeeze(-1)
        if self.pair_skip_mlp is not None:
            pair_e = pair_e + self.pair_skip_mlp(radial).squeeze(-1)
        pair_sum = h.new_zeros(graph.num_batches, 1)
        pair_sum.index_add_(0, graph.batch, pair_e.unsqueeze(-1))
        return node_e + 0.5 * pair_sum

    def direct_forces(self, pos: torch.Tensor, h: torch.Tensor, z: torch.Tensor, graph, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if graph.edge_count == 0:
            return pos.new_zeros(pos.shape), {"force_final_activation_norm": pos.new_zeros(())}
        radial = self.radial(graph.distances)
        edge_features = self.edge_features(h, z, graph)
        if isinstance(self.force_mlp, nn.Sequential) and isinstance(self.force_mlp[-1], nn.Linear):
            hidden = self.force_mlp[:-1](edge_features)
            weights = self.force_mlp[-1](hidden)
            activation_norm = hidden.norm(dim=-1).mean()
        else:
            weights = self.force_mlp(edge_features)
            activation_norm = weights.norm(dim=-1).mean()
        if self.pair_skip_mlp is not None:
            weights = weights + self.pair_skip_mlp(radial)
        if self.force_gate is not None:
            hi = h[graph.batch, graph.dst]
            weights = weights * torch.sigmoid(self.force_gate(hi))
        if self.attention_mlp is not None:
            weights = weights * edge_softmax(self.attention_mlp(edge_features), graph)
        messages = weights * graph.edge_vec
        forces = aggregate_to_padded_nodes(messages, graph)
        return forces * mask.unsqueeze(-1), {
            "force_final_activation_norm": activation_norm,
            "last_hidden_norm": h.norm(dim=-1)[mask].mean() if mask.any() else h.new_zeros(()),
            "message_norm_mean": messages.norm(dim=-1).mean(),
            "edge_message_norm_mean": messages.norm(dim=-1).mean(),
            "force_head_output_norm": weights.norm(dim=-1).mean(),
        }

    def forward(self, pos: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        if mask is None:
            mask = torch.ones(pos.shape[:2], dtype=torch.bool, device=pos.device)
        if self.training_mode == "energy_force" and not pos.requires_grad:
            pos = pos.requires_grad_(True)
        graph = build_molecular_graph(
            pos,
            mask,
            cutoff_radius=self.cutoff_radius,
            max_neighbors=self.max_neighbors,
            graph_mode=self.graph_mode,
        )
        h = self.encode(z, mask, graph)
        energy = self.invariant_energy(h, z, graph, mask)
        raw_energy = energy * self.energy_output_scale + self.energy_output_shift
        if self.training_mode == "energy_force":
            grad = torch.autograd.grad(
                raw_energy.sum(),
                pos,
                create_graph=self.training,
                retain_graph=True,
                allow_unused=False,
            )[0]
            forces = -grad * mask.unsqueeze(-1)
            force_diagnostics = {"force_output_scale": pos.new_ones(())}
        else:
            raw_forces, force_diagnostics = self.direct_forces(pos, h, z, graph, mask)
            scale = self.force_output_scale_value().to(device=raw_forces.device, dtype=raw_forces.dtype)
            forces = raw_forces * scale
            force_diagnostics["force_output_scale"] = scale
        return {
            "forces": forces,
            "energy": energy,
            "energy_raw": raw_energy,
            "graph_stats": graph_stats(graph, self.cutoff_radius),
            "diagnostics": force_diagnostics,
        }


class MolecularEGNN(MolecularScalarForceField):
    def __init__(self, **kwargs) -> None:
        kwargs.pop("use_attention", None)
        kwargs.pop("use_gate", None)
        kwargs.pop("pair_skip", None)
        super().__init__(
            model_label="egnn",
            backbone_class="egnn_scalar_message",
            use_attention=False,
            use_gate=False,
            pair_skip=False,
            **kwargs,
        )


class MolecularTFNConv(MolecularScalarForceField):
    def __init__(self, **kwargs) -> None:
        kwargs.pop("use_attention", None)
        kwargs.pop("use_gate", None)
        kwargs.pop("pair_skip", None)
        super().__init__(
            model_label="tfn_no_attention",
            backbone_class="tfn_scalar_kernel",
            use_attention=False,
            use_gate=False,
            pair_skip=True,
            **kwargs,
        )


class MolecularSE3ForceTransformer(MolecularScalarForceField):
    def __init__(self, channels_by_l: dict | None = None, lmax: int = 2, num_query_channels: int = 8, **kwargs) -> None:
        channels_by_l = channels_by_l or {0: kwargs.pop("hidden_dim", 96)}
        hidden_dim = int(channels_by_l.get(0, channels_by_l.get("0", 96)))
        hidden_irreps = "+".join(f"{mul}x{degree}" for degree, mul in sorted((int(k), int(v)) for k, v in channels_by_l.items()))
        use_attention = bool(kwargs.pop("use_attention", True))
        use_gate = bool(kwargs.pop("use_gate", True))
        super().__init__(
            hidden_dim=hidden_dim,
            model_label=f"se3_scalar_l{lmax}",
            backbone_class="se3_scalar_attention_kernel",
            lmax=int(lmax),
            hidden_irreps=hidden_irreps,
            use_attention=use_attention,
            use_gate=use_gate,
            pair_skip=False,
            **kwargs,
        )
        self.lmax = lmax
        self.num_query_channels = num_query_channels


class MolecularRadialForceBaseline(MolecularScalarForceField):
    def __init__(self, **kwargs) -> None:
        kwargs.pop("use_attention", None)
        kwargs.pop("use_gate", None)
        kwargs.pop("pair_skip", None)
        kwargs.pop("num_layers", None)
        super().__init__(
            num_layers=0,
            model_label="radial_pair",
            backbone_class="pairwise_radial_mlp",
            hidden_irreps="scalar_radial",
            use_attention=False,
            use_gate=False,
            pair_skip=False,
            **kwargs,
        )


class HighCapacityRadialForceModel(nn.Module):
    """Equivariant central pair-force diagnostic with atom-pair conditioning."""

    def __init__(
        self,
        hidden_dim: int = 192,
        radial_num_basis: int = 32,
        radial_hidden_dim: int = 256,
        edge_mlp_hidden_dim: int | None = None,
        edge_mlp_layers: int = 4,
        cutoff_radius: float = 5.0,
        max_neighbors: int | None = None,
        graph_mode: str = "cutoff",
        max_atomic_number: int = 100,
        atom_embedding_dim: int | None = None,
        use_atom_pair_embedding: bool = True,
        pair_embedding_dim: int = 32,
        pair_residual_terms: bool = False,
        training_mode: str = "direct_force",
        output_head_init_scale: float = 1.0,
        force_output_scale: float = 1.0,
        learnable_force_output_scale: bool = False,
        initial_force_output_scale: float | None = None,
        force_output_scale_regularization: float = 0.0,
        **_unused,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(atom_embedding_dim or hidden_dim)
        self.radial_num_basis = int(radial_num_basis)
        self.radial_hidden_dim = int(radial_hidden_dim)
        self.edge_mlp_hidden_dim = int(edge_mlp_hidden_dim or radial_hidden_dim)
        self.edge_mlp_layers = int(edge_mlp_layers)
        self.cutoff_radius = float(cutoff_radius)
        self.max_neighbors = max_neighbors
        self.graph_mode = str(graph_mode)
        self.max_atomic_number = int(max_atomic_number)
        self.use_atom_pair_embedding = bool(use_atom_pair_embedding)
        self.pair_embedding_dim = int(pair_embedding_dim) if self.use_atom_pair_embedding else 0
        self.pair_residual_terms = bool(pair_residual_terms)
        self.training_mode = training_mode
        self.model_label = "radial_pair"
        self.backbone_class = "high_capacity_radial_pair"
        self.lmax = 1
        self.hidden_irreps = "scalar_pair_conditioned"
        self.use_attention = False
        self.use_gate = False
        self.output_head_init_scale = float(output_head_init_scale)
        self.force_output_scale = float(force_output_scale)
        self.learnable_force_output_scale = bool(learnable_force_output_scale)
        self.initial_force_output_scale = float(initial_force_output_scale if initial_force_output_scale is not None else force_output_scale)
        self.force_output_scale_regularization = float(force_output_scale_regularization)

        self.atom_embedding = nn.Embedding(self.max_atomic_number + 1, self.hidden_dim)
        self.pair_embedding = (
            nn.Embedding((self.max_atomic_number + 1) * (self.max_atomic_number + 1), self.pair_embedding_dim)
            if self.use_atom_pair_embedding
            else None
        )
        self.radial = GaussianRadialBasis(self.radial_num_basis)
        input_dim = self.radial_num_basis + 2 * self.hidden_dim + self.pair_embedding_dim
        self.force_mlp = make_mlp(input_dim, self.edge_mlp_hidden_dim, 1, self.edge_mlp_layers)
        self.pair_residual_mlp = make_mlp(self.radial_num_basis, self.edge_mlp_hidden_dim, 1, 1) if self.pair_residual_terms else None
        if self.learnable_force_output_scale:
            initial = max(abs(self.initial_force_output_scale), 1e-12)
            self.force_output_log_scale = nn.Parameter(torch.tensor(math.log(initial), dtype=torch.float32))
        else:
            self.force_output_log_scale = None
        self._apply_output_head_init_scale()
        self.architecture_signature = "|".join(
            [
                "radial_pair",
                f"model_class={self.__class__.__name__}",
                f"backbone={self.backbone_class}",
                f"hidden={self.hidden_dim}",
                f"edge_hidden={self.edge_mlp_hidden_dim}",
                f"edge_layers={self.edge_mlp_layers}",
                f"radial_basis={self.radial_num_basis}",
                f"graph_mode={self.graph_mode}",
                f"atom_pair_embedding={self.use_atom_pair_embedding}",
                f"pair_embedding_dim={self.pair_embedding_dim}",
                f"pair_residual_terms={self.pair_residual_terms}",
                f"force_output_scale={self.force_output_scale:.6g}",
                f"learnable_force_output_scale={self.learnable_force_output_scale}",
                f"initial_force_output_scale={self.initial_force_output_scale:.6g}",
            ]
        )

    def _apply_output_head_init_scale(self) -> None:
        final = self.force_mlp[-1]
        if isinstance(final, nn.Linear) and self.output_head_init_scale != 1.0:
            with torch.no_grad():
                final.weight.mul_(self.output_head_init_scale)
                if final.bias is not None:
                    final.bias.mul_(self.output_head_init_scale)

    def force_output_scale_value(self) -> torch.Tensor:
        if self.force_output_log_scale is not None:
            return self.force_output_log_scale.exp()
        return torch.tensor(self.force_output_scale, dtype=torch.float32, device=self.atom_embedding.weight.device)

    def pair_features(self, z: torch.Tensor, graph) -> torch.Tensor:
        if self.pair_embedding is None:
            return self.atom_embedding.weight.new_zeros(graph.edge_count, 0).to(device=z.device)
        zi = z[graph.batch, graph.dst].clamp(0, self.max_atomic_number)
        zj = z[graph.batch, graph.src].clamp(0, self.max_atomic_number)
        pair_index = zi * (self.max_atomic_number + 1) + zj
        return self.pair_embedding(pair_index)

    def architecture_metadata(self) -> dict[str, object]:
        return {
            "actual_hidden_irreps": "scalar_pair_conditioned",
            "irreps_in": f"{self.hidden_dim}x0e",
            "irreps_hidden": "scalar_pair_conditioned",
            "irreps_out": "1x1o",
            "irreps_sh": "",
            "uses_non_scalar_hidden": False,
            "uses_spherical_harmonics_in_value": False,
            "force_head_type": "central_pair_scalar_mlp",
            "force_head_irreps": "1x1o",
            "uses_relative_vector_fallback": True,
            "uses_pairwise_force_skip": bool(self.pair_residual_terms),
            "uses_atom_pair_embedding": bool(self.use_atom_pair_embedding),
            "atom_embedding_dim": int(self.hidden_dim),
            "pair_embedding_dim": int(self.pair_embedding_dim),
            "edge_mlp_hidden_dim": int(self.edge_mlp_hidden_dim),
            "edge_mlp_layers": int(self.edge_mlp_layers),
            "graph_mode": self.graph_mode,
        }

    def forward(self, pos: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        if mask is None:
            mask = torch.ones(pos.shape[:2], dtype=torch.bool, device=pos.device)
        graph = build_molecular_graph(
            pos,
            mask,
            cutoff_radius=self.cutoff_radius,
            max_neighbors=self.max_neighbors,
            graph_mode=self.graph_mode,
        )
        if graph.edge_count == 0:
            scale = self.force_output_scale_value().to(device=pos.device, dtype=pos.dtype)
            return {
                "forces": pos.new_zeros(pos.shape),
                "energy": pos.new_zeros(pos.shape[0], 1),
                "energy_raw": pos.new_zeros(pos.shape[0], 1),
                "graph_stats": graph_stats(graph, self.cutoff_radius),
                "diagnostics": {"force_output_scale": scale},
            }
        atom_h = self.atom_embedding(z.clamp(0, self.max_atomic_number))
        hi = atom_h[graph.batch, graph.dst]
        hj = atom_h[graph.batch, graph.src]
        radial = self.radial(graph.distances)
        pair = self.pair_features(z, graph).to(dtype=pos.dtype)
        edge_features = torch.cat([radial, hi, hj, pair], dim=-1)
        hidden = self.force_mlp[:-1](edge_features)
        weights = self.force_mlp[-1](hidden)
        if self.pair_residual_mlp is not None:
            weights = weights + self.pair_residual_mlp(radial)
        messages = weights * graph.edge_vec
        raw_forces = aggregate_to_padded_nodes(messages, graph) * mask.unsqueeze(-1)
        scale = self.force_output_scale_value().to(device=raw_forces.device, dtype=raw_forces.dtype)
        forces = raw_forces * scale
        return {
            "forces": forces,
            "energy": pos.new_zeros(pos.shape[0], 1),
            "energy_raw": pos.new_zeros(pos.shape[0], 1),
            "graph_stats": graph_stats(graph, self.cutoff_radius),
            "diagnostics": {
                "force_final_activation_norm": hidden.norm(dim=-1).mean(),
                "last_hidden_norm": hidden.norm(dim=-1).mean(),
                "message_norm_mean": messages.norm(dim=-1).mean(),
                "edge_message_norm_mean": messages.norm(dim=-1).mean(),
                "force_head_output_norm": weights.norm(dim=-1).mean(),
                "force_output_scale": scale,
            },
        }


class GlobalContextRadialForceModel(nn.Module):
    """Local radial force model conditioned on invariant global molecular context."""

    def __init__(
        self,
        max_atoms: int = 32,
        hidden_dim: int = 192,
        radial_num_basis: int = 32,
        radial_hidden_dim: int = 256,
        edge_mlp_hidden_dim: int | None = None,
        edge_mlp_layers: int = 4,
        global_context_dim: int = 128,
        global_hidden_dim: int | None = None,
        global_layers: int = 2,
        use_global_context: bool = True,
        distance_scale: float = 5.0,
        cutoff_radius: float = 5.0,
        max_neighbors: int | None = None,
        graph_mode: str = "full",
        max_atomic_number: int = 100,
        atom_embedding_dim: int | None = None,
        use_atom_pair_embedding: bool = True,
        pair_embedding_dim: int = 32,
        training_mode: str = "direct_force",
        output_head_init_scale: float = 1.0,
        force_output_scale: float = 1.0,
        learnable_force_output_scale: bool = False,
        initial_force_output_scale: float | None = None,
        force_output_scale_regularization: float = 0.0,
        **_unused,
    ) -> None:
        super().__init__()
        self.max_atoms = int(max_atoms)
        self.hidden_dim = int(atom_embedding_dim or hidden_dim)
        self.radial_num_basis = int(radial_num_basis)
        self.radial_hidden_dim = int(radial_hidden_dim)
        self.edge_mlp_hidden_dim = int(edge_mlp_hidden_dim or radial_hidden_dim)
        self.edge_mlp_layers = int(edge_mlp_layers)
        self.global_context_dim = int(global_context_dim)
        self.use_global_context = bool(use_global_context)
        self.global_hidden_dim = int(global_hidden_dim or max(self.global_context_dim, self.hidden_dim))
        self.global_layers = int(global_layers)
        self.distance_scale = float(distance_scale)
        self.cutoff_radius = float(cutoff_radius)
        self.max_neighbors = max_neighbors
        self.graph_mode = str(graph_mode)
        self.max_atomic_number = int(max_atomic_number)
        self.use_atom_pair_embedding = bool(use_atom_pair_embedding)
        self.pair_embedding_dim = int(pair_embedding_dim) if self.use_atom_pair_embedding else 0
        self.training_mode = training_mode
        self.model_label = "global_context_radial"
        self.backbone_class = "global_context_radial_pair"
        self.lmax = 1
        self.hidden_irreps = "global_context_scalar_pair_conditioned"
        self.use_attention = False
        self.use_gate = False
        self.output_head_init_scale = float(output_head_init_scale)
        self.force_output_scale = float(force_output_scale)
        self.learnable_force_output_scale = bool(learnable_force_output_scale)
        self.initial_force_output_scale = float(initial_force_output_scale if initial_force_output_scale is not None else force_output_scale)
        self.force_output_scale_regularization = float(force_output_scale_regularization)

        self.atom_embedding = nn.Embedding(self.max_atomic_number + 1, self.hidden_dim)
        self.pair_embedding = (
            nn.Embedding((self.max_atomic_number + 1) * (self.max_atomic_number + 1), self.pair_embedding_dim)
            if self.use_atom_pair_embedding
            else None
        )
        self.radial = GaussianRadialBasis(self.radial_num_basis, cutoff=self.distance_scale)
        descriptor_dim = self.max_atoms * self.max_atoms + self.max_atoms * self.hidden_dim + self.max_atoms
        if self.use_global_context:
            self.global_mlp = make_mlp(descriptor_dim, self.global_hidden_dim, self.global_context_dim, self.global_layers)
            active_global_context_dim = self.global_context_dim
        else:
            self.global_mlp = None
            active_global_context_dim = 0
        edge_input_dim = self.radial_num_basis + 2 * self.hidden_dim + self.pair_embedding_dim + active_global_context_dim
        self.force_mlp = make_mlp(edge_input_dim, self.edge_mlp_hidden_dim, 1, self.edge_mlp_layers)
        if self.learnable_force_output_scale:
            initial = max(abs(self.initial_force_output_scale), 1e-12)
            self.force_output_log_scale = nn.Parameter(torch.tensor(math.log(initial), dtype=torch.float32))
        else:
            self.force_output_log_scale = None
        self._apply_output_head_init_scale()
        self.architecture_signature = "|".join(
            [
                "global_context_radial",
                f"model_class={self.__class__.__name__}",
                f"backbone={self.backbone_class}",
                f"max_atoms={self.max_atoms}",
                f"hidden={self.hidden_dim}",
                f"context={self.global_context_dim}",
                f"use_global_context={self.use_global_context}",
                f"context_type=full_distance_atom_embedding",
                f"edge_hidden={self.edge_mlp_hidden_dim}",
                f"edge_layers={self.edge_mlp_layers}",
                f"radial_basis={self.radial_num_basis}",
                f"graph_mode={self.graph_mode}",
                f"atom_pair_embedding={self.use_atom_pair_embedding}",
                f"force_output_scale={self.force_output_scale:.6g}",
            ]
        )

    def _apply_output_head_init_scale(self) -> None:
        final = self.force_mlp[-1]
        if isinstance(final, nn.Linear) and self.output_head_init_scale != 1.0:
            with torch.no_grad():
                final.weight.mul_(self.output_head_init_scale)
                if final.bias is not None:
                    final.bias.mul_(self.output_head_init_scale)

    def force_output_scale_value(self) -> torch.Tensor:
        if self.force_output_log_scale is not None:
            return self.force_output_log_scale.exp()
        return torch.tensor(self.force_output_scale, dtype=torch.float32, device=self.atom_embedding.weight.device)

    def global_descriptor(self, pos: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_atoms, _ = pos.shape
        if num_atoms > self.max_atoms:
            raise ValueError(f"num_atoms={num_atoms} exceeds global_context_radial max_atoms={self.max_atoms}")
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        valid_pair = (mask[:, :, None] & mask[:, None, :]).to(pos.dtype)
        distance = (diff * diff).sum(dim=-1).clamp_min(1e-24).sqrt() / max(self.distance_scale, 1e-12)
        padded_distance = pos.new_zeros(batch_size, self.max_atoms, self.max_atoms)
        padded_distance[:, :num_atoms, :num_atoms] = distance * valid_pair
        atom_h = self.atom_embedding(z.clamp(0, self.max_atomic_number)).to(dtype=pos.dtype) * mask.unsqueeze(-1)
        padded_atom_h = pos.new_zeros(batch_size, self.max_atoms, self.hidden_dim)
        padded_atom_h[:, :num_atoms] = atom_h
        padded_mask = pos.new_zeros(batch_size, self.max_atoms)
        padded_mask[:, :num_atoms] = mask.to(pos.dtype)
        return torch.cat([padded_distance.reshape(batch_size, -1), padded_atom_h.reshape(batch_size, -1), padded_mask], dim=-1)

    def pair_features(self, z: torch.Tensor, graph) -> torch.Tensor:
        if self.pair_embedding is None:
            return self.atom_embedding.weight.new_zeros(graph.edge_count, 0).to(device=z.device)
        zi = z[graph.batch, graph.dst].clamp(0, self.max_atomic_number)
        zj = z[graph.batch, graph.src].clamp(0, self.max_atomic_number)
        return self.pair_embedding(pair_index_from_z(zi, zj, self.max_atomic_number))

    def architecture_metadata(self) -> dict[str, object]:
        return {
            "actual_hidden_irreps": "global_context_scalar_pair_conditioned",
            "irreps_in": f"{self.hidden_dim}x0e",
            "irreps_hidden": f"{self.global_context_dim}x0e+scalar_pair_conditioned",
            "irreps_out": "1x1o",
            "irreps_sh": "",
            "uses_non_scalar_hidden": False,
            "uses_spherical_harmonics_in_value": False,
            "force_head_type": "global_context_central_pair_scalar_mlp",
            "force_head_irreps": "1x1o",
            "uses_relative_vector_fallback": True,
            "uses_pairwise_force_skip": False,
            "uses_atom_pair_embedding": bool(self.use_atom_pair_embedding),
            "atom_embedding_dim": int(self.hidden_dim),
            "pair_embedding_dim": int(self.pair_embedding_dim),
            "edge_mlp_hidden_dim": int(self.edge_mlp_hidden_dim),
            "edge_mlp_layers": int(self.edge_mlp_layers),
            "graph_mode": self.graph_mode,
            "uses_global_context": bool(self.use_global_context),
            "global_context_dim": int(self.global_context_dim if self.use_global_context else 0),
            "global_context_type": "full_distance_atom_embedding" if self.use_global_context else "none",
        }

    def forward(self, pos: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        if mask is None:
            mask = torch.ones(pos.shape[:2], dtype=torch.bool, device=pos.device)
        graph = build_molecular_graph(
            pos,
            mask,
            cutoff_radius=self.cutoff_radius,
            max_neighbors=self.max_neighbors,
            graph_mode=self.graph_mode,
        )
        if self.global_mlp is not None:
            context = self.global_mlp(self.global_descriptor(pos, z, mask))
        else:
            context = pos.new_zeros(pos.shape[0], 0)
        if graph.edge_count == 0:
            scale = self.force_output_scale_value().to(device=pos.device, dtype=pos.dtype)
            return {
                "forces": pos.new_zeros(pos.shape),
                "energy": pos.new_zeros(pos.shape[0], 1),
                "energy_raw": pos.new_zeros(pos.shape[0], 1),
                "graph_stats": graph_stats(graph, self.cutoff_radius),
                "diagnostics": {"force_output_scale": scale, "last_hidden_norm": context.norm(dim=-1).mean()},
            }
        atom_h = self.atom_embedding(z.clamp(0, self.max_atomic_number)).to(dtype=pos.dtype)
        hi = atom_h[graph.batch, graph.dst]
        hj = atom_h[graph.batch, graph.src]
        radial = self.radial(graph.distances)
        pair = self.pair_features(z, graph).to(dtype=pos.dtype)
        features = [radial, hi, hj, pair]
        if self.global_mlp is not None:
            features.append(context[graph.batch].to(dtype=pos.dtype))
        edge_features = torch.cat(features, dim=-1)
        hidden = self.force_mlp[:-1](edge_features)
        weights = self.force_mlp[-1](hidden)
        messages = weights * graph.edge_vec
        raw_forces = aggregate_to_padded_nodes(messages, graph) * mask.unsqueeze(-1)
        scale = self.force_output_scale_value().to(device=raw_forces.device, dtype=raw_forces.dtype)
        forces = raw_forces * scale
        return {
            "forces": forces,
            "energy": pos.new_zeros(pos.shape[0], 1),
            "energy_raw": pos.new_zeros(pos.shape[0], 1),
            "graph_stats": graph_stats(graph, self.cutoff_radius),
            "diagnostics": {
                "force_final_activation_norm": hidden.norm(dim=-1).mean(),
                "last_hidden_norm": context.norm(dim=-1).mean(),
                "message_norm_mean": messages.norm(dim=-1).mean(),
                "edge_message_norm_mean": messages.norm(dim=-1).mean(),
                "force_head_output_norm": weights.norm(dim=-1).mean(),
                "force_output_scale": scale,
            },
        }


class MolecularFullIrrepSE3ForceTransformer(nn.Module):
    """Molecular wrapper around the true irrep-based SE3ForceTransformer."""

    def __init__(
        self,
        scalar_input_dim: int | None = None,
        atom_embedding_dim: int = 16,
        max_atomic_number: int = 100,
        lmax: int = 2,
        channels_by_l: dict | None = None,
        num_layers: int = 3,
        num_heads: int = 2,
        num_query_channels: int = 8,
        radial_num_basis: int = 16,
        radial_hidden_dim: int = 64,
        dropout: float = 0.0,
        use_attention: bool = True,
        use_gate: bool = True,
        cutoff_radius: float = 5.0,
        max_neighbors: int | None = None,
        graph_mode: str = "cutoff",
        training_mode: str = "direct_force",
        output_head_init_scale: float = 1.0,
        force_output_scale: float = 1.0,
        learnable_force_output_scale: bool = False,
        initial_force_output_scale: float | None = None,
        force_output_scale_regularization: float = 0.0,
        **_unused,
    ) -> None:
        super().__init__()
        self.atom_embedding_dim = int(scalar_input_dim or atom_embedding_dim)
        self.max_atomic_number = int(max_atomic_number)
        self.cutoff_radius = float(cutoff_radius)
        self.max_neighbors = max_neighbors
        self.graph_mode = str(graph_mode)
        self.training_mode = training_mode
        self.lmax = int(lmax)
        self.use_attention = bool(use_attention)
        self.use_gate = bool(use_gate)
        self.output_head_init_scale = float(output_head_init_scale)
        self.force_output_scale = float(force_output_scale)
        self.learnable_force_output_scale = bool(learnable_force_output_scale)
        self.initial_force_output_scale = float(initial_force_output_scale if initial_force_output_scale is not None else force_output_scale)
        self.force_output_scale_regularization = float(force_output_scale_regularization)
        self.atom_embedding = nn.Embedding(self.max_atomic_number + 1, self.atom_embedding_dim)
        self.backbone = SE3ForceTransformer(
            scalar_input_dim=self.atom_embedding_dim,
            lmax=self.lmax,
            channels_by_l=channels_by_l,
            num_layers=num_layers,
            num_heads=num_heads,
            num_query_channels=num_query_channels,
            radial_num_basis=radial_num_basis,
            radial_hidden_dim=radial_hidden_dim,
            dropout=dropout,
            use_attention=self.use_attention,
            use_gate=self.use_gate,
        )
        self.hidden_irreps = str(self.backbone.irreps_hidden)
        self.backbone_class = "se3_full_irrep"
        self.model_label = f"se3_full_l{self.lmax}"
        if self.learnable_force_output_scale:
            initial = max(abs(self.initial_force_output_scale), 1e-12)
            self.force_output_log_scale = nn.Parameter(torch.tensor(math.log(initial), dtype=torch.float32))
        else:
            self.force_output_log_scale = None
        self._apply_output_head_init_scale()
        self.architecture_signature = "|".join(
            [
                self.model_label,
                f"model_class={self.__class__.__name__}",
                f"backbone={self.backbone_class}",
                f"graph_mode={self.graph_mode}",
                f"irreps_hidden={self.hidden_irreps}",
                f"lmax={self.lmax}",
                f"attention={self.use_attention}",
                f"gate={self.use_gate}",
                f"force_head_type={self.force_head_type()}",
                f"force_output_scale={self.force_output_scale:.6g}",
                f"learnable_force_output_scale={self.learnable_force_output_scale}",
                f"initial_force_output_scale={self.initial_force_output_scale:.6g}",
            ]
        )

    def _apply_output_head_init_scale(self) -> None:
        head = getattr(self.backbone, "force_head", None)
        weight = getattr(head, "weight", None)
        if weight is not None and self.output_head_init_scale != 1.0:
            with torch.no_grad():
                weight.mul_(self.output_head_init_scale)

    def force_head_type(self) -> str:
        return "e3nn_linear" if getattr(self.backbone, "force_edge_mlp", None) is None else "relative_vector_fallback"

    def force_output_scale_value(self) -> torch.Tensor:
        if self.force_output_log_scale is not None:
            return self.force_output_log_scale.exp()
        return torch.tensor(self.force_output_scale, dtype=torch.float32, device=self.atom_embedding.weight.device)

    def architecture_metadata(self) -> dict[str, object]:
        irreps_hidden = o3.Irreps(self.hidden_irreps)
        return {
            "actual_hidden_irreps": str(irreps_hidden),
            "irreps_in": f"{self.atom_embedding_dim}x0e",
            "irreps_hidden": str(irreps_hidden),
            "irreps_out": "1x1o",
            "irreps_sh": str(spherical_harmonics_irreps(self.lmax)),
            "uses_non_scalar_hidden": has_non_scalar_irrep(irreps_hidden),
            "uses_spherical_harmonics_in_value": True,
            "force_head_type": self.force_head_type(),
            "force_head_irreps": "1x1o",
            "uses_relative_vector_fallback": self.force_head_type() == "relative_vector_fallback",
            "uses_pairwise_force_skip": False,
            "uses_atom_pair_embedding": False,
            "atom_embedding_dim": int(self.atom_embedding_dim),
            "pair_embedding_dim": 0,
            "edge_mlp_hidden_dim": int(getattr(self.backbone.blocks[0].message.heads[0].value_radial.net[0], "out_features", 0))
            if self.backbone.blocks and hasattr(getattr(self.backbone.blocks[0], "message", None), "heads")
            else None,
            "edge_mlp_layers": None,
            "graph_mode": self.graph_mode,
        }

    def forward(self, pos: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        if mask is None:
            mask = torch.ones(pos.shape[:2], dtype=torch.bool, device=pos.device)
        graph = build_molecular_graph(
            pos,
            mask,
            cutoff_radius=self.cutoff_radius,
            max_neighbors=self.max_neighbors,
            graph_mode=self.graph_mode,
        )
        atom_features = self.atom_embedding(z.clamp(0, self.max_atomic_number)) * mask.unsqueeze(-1)
        raw_forces, diagnostics = self.backbone(pos, atom_features, mask=mask, edges=graph, return_diagnostics=True)
        scale = self.force_output_scale_value().to(device=raw_forces.device, dtype=raw_forces.dtype)
        forces = raw_forces * scale
        diagnostics["force_output_scale"] = scale
        return {
            "forces": forces,
            "energy": pos.new_zeros(pos.shape[0], 1),
            "energy_raw": pos.new_zeros(pos.shape[0], 1),
            "graph_stats": graph_stats(graph, self.cutoff_radius),
            "diagnostics": diagnostics,
        }


class GlobalInvariantCoefficientForceModel(nn.Module):
    """Fixed-N equivariant diagnostic using global invariant coefficients."""

    def __init__(
        self,
        max_atoms: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 4,
        global_context_dim: int = 256,
        atom_embedding_dim: int = 32,
        use_atom_pair_embedding: bool = True,
        pair_embedding_dim: int = 32,
        radial_num_basis: int = 32,
        edge_mlp_hidden_dim: int | None = None,
        edge_mlp_layers: int = 2,
        distance_scale: float = 5.0,
        cutoff_radius: float = 5.0,
        max_neighbors: int | None = None,
        graph_mode: str = "full",
        max_atomic_number: int = 100,
        training_mode: str = "direct_force",
        output_head_init_scale: float = 1.0,
        force_output_scale: float = 1.0,
        learnable_force_output_scale: bool = False,
        initial_force_output_scale: float | None = None,
        force_output_scale_regularization: float = 0.0,
        use_prototype_memory: bool = False,
        prototype_count: int | None = None,
        prototype_temperature: float = 4096.0,
        prototype_assignment: str = "nearest",
        prototype_coeff_init_scale: float = 0.0,
        **_unused,
    ) -> None:
        super().__init__()
        self.max_atoms = int(max_atoms)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.global_context_dim = int(global_context_dim)
        self.atom_embedding_dim = int(atom_embedding_dim)
        self.use_atom_pair_embedding = bool(use_atom_pair_embedding)
        self.pair_embedding_dim = int(pair_embedding_dim) if self.use_atom_pair_embedding else 0
        self.radial_num_basis = int(radial_num_basis)
        self.edge_mlp_hidden_dim = int(edge_mlp_hidden_dim or hidden_dim)
        self.edge_mlp_layers = int(edge_mlp_layers)
        self.distance_scale = float(distance_scale)
        self.cutoff_radius = float(cutoff_radius)
        self.max_neighbors = max_neighbors
        self.graph_mode = str(graph_mode)
        self.max_atomic_number = int(max_atomic_number)
        self.training_mode = training_mode
        self.backbone_class = "global_invariant_coefficients"
        self.model_label = "global_coeff"
        self.lmax = 1
        self.hidden_irreps = "global_invariant_scalars_to_vectors"
        self.use_attention = False
        self.use_gate = False
        self.output_head_init_scale = float(output_head_init_scale)
        self.force_output_scale = float(force_output_scale)
        self.learnable_force_output_scale = bool(learnable_force_output_scale)
        self.initial_force_output_scale = float(initial_force_output_scale if initial_force_output_scale is not None else force_output_scale)
        self.force_output_scale_regularization = float(force_output_scale_regularization)
        self.use_prototype_memory = bool(use_prototype_memory)
        self.prototype_count = int(prototype_count if prototype_count is not None else 0)
        self.prototype_temperature = float(prototype_temperature)
        self.prototype_assignment = str(prototype_assignment)
        self.prototype_coeff_init_scale = float(prototype_coeff_init_scale)
        if self.use_prototype_memory and self.prototype_count <= 0:
            raise ValueError("prototype_count must be positive when use_prototype_memory is true")
        if self.prototype_assignment not in {"nearest", "softmax"}:
            raise ValueError(f"unknown prototype_assignment: {self.prototype_assignment}")

        self.atom_embedding = nn.Embedding(self.max_atomic_number + 1, self.atom_embedding_dim)
        self.pair_embedding = (
            nn.Embedding((self.max_atomic_number + 1) * (self.max_atomic_number + 1), self.pair_embedding_dim)
            if self.use_atom_pair_embedding
            else None
        )
        self.radial = GaussianRadialBasis(self.radial_num_basis, cutoff=self.distance_scale)
        descriptor_dim = self.max_atoms * self.max_atoms + self.max_atoms * self.atom_embedding_dim + self.max_atoms
        self.prototype_descriptor_dim = self.max_atoms * self.max_atoms + self.max_atoms
        self.global_mlp = make_mlp(descriptor_dim, self.hidden_dim, self.global_context_dim, self.num_layers)
        self.coeff_matrix_head = nn.Linear(self.global_context_dim, self.max_atoms * self.max_atoms)
        edge_input_dim = self.global_context_dim + 2 * self.atom_embedding_dim + self.pair_embedding_dim + self.radial_num_basis
        self.edge_mlp = make_mlp(edge_input_dim, self.edge_mlp_hidden_dim, 1, self.edge_mlp_layers)
        if self.use_prototype_memory:
            self.register_buffer("prototype_descriptors", torch.zeros(self.prototype_count, self.prototype_descriptor_dim))
            self.register_buffer("prototype_descriptors_initialized", torch.tensor(False))
            self.register_buffer("prototype_fill_count", torch.tensor(0, dtype=torch.long))
            prototype_coefficients = torch.empty(self.prototype_count, self.max_atoms, self.max_atoms)
            if self.prototype_coeff_init_scale == 0.0:
                nn.init.zeros_(prototype_coefficients)
            else:
                nn.init.normal_(prototype_coefficients, mean=0.0, std=self.prototype_coeff_init_scale)
            self.prototype_coefficients = nn.Parameter(prototype_coefficients)
        else:
            self.register_buffer("prototype_descriptors", torch.zeros(0, self.prototype_descriptor_dim))
            self.register_buffer("prototype_descriptors_initialized", torch.tensor(False))
            self.register_buffer("prototype_fill_count", torch.tensor(0, dtype=torch.long))
            self.prototype_coefficients = None
        if self.learnable_force_output_scale:
            initial = max(abs(self.initial_force_output_scale), 1e-12)
            self.force_output_log_scale = nn.Parameter(torch.tensor(math.log(initial), dtype=torch.float32))
        else:
            self.force_output_log_scale = None
        self._apply_output_head_init_scale()
        self.architecture_signature = "|".join(
            [
                "global_coeff",
                f"model_class={self.__class__.__name__}",
                f"backbone={self.backbone_class}",
                f"max_atoms={self.max_atoms}",
                f"hidden={self.hidden_dim}",
                f"layers={self.num_layers}",
                f"context={self.global_context_dim}",
                f"edge_hidden={self.edge_mlp_hidden_dim}",
                f"edge_layers={self.edge_mlp_layers}",
                f"graph_mode={self.graph_mode}",
                f"atom_pair_embedding={self.use_atom_pair_embedding}",
                f"force_output_scale={self.force_output_scale:.6g}",
                f"learnable_force_output_scale={self.learnable_force_output_scale}",
                f"initial_force_output_scale={self.initial_force_output_scale:.6g}",
                f"use_prototype_memory={self.use_prototype_memory}",
                f"prototype_count={self.prototype_count}",
                f"prototype_assignment={self.prototype_assignment}",
            ]
        )

    def _apply_output_head_init_scale(self) -> None:
        if self.output_head_init_scale == 1.0:
            return
        for layer in [self.coeff_matrix_head, self.edge_mlp[-1]]:
            if isinstance(layer, nn.Linear):
                with torch.no_grad():
                    layer.weight.mul_(self.output_head_init_scale)
                    if layer.bias is not None:
                        layer.bias.mul_(self.output_head_init_scale)

    def force_output_scale_value(self) -> torch.Tensor:
        if self.force_output_log_scale is not None:
            return self.force_output_log_scale.exp()
        return torch.tensor(self.force_output_scale, dtype=torch.float32, device=self.atom_embedding.weight.device)

    def global_descriptor(self, pos: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_atoms, _ = pos.shape
        if num_atoms > self.max_atoms:
            raise ValueError(f"num_atoms={num_atoms} exceeds global_coeff max_atoms={self.max_atoms}")
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        distance = (diff * diff).sum(dim=-1).clamp_min(1e-24).sqrt() / max(self.distance_scale, 1e-12)
        padded_distance = pos.new_zeros(batch_size, self.max_atoms, self.max_atoms)
        padded_distance[:, :num_atoms, :num_atoms] = distance * (mask[:, :, None] & mask[:, None, :]).to(pos.dtype)
        atom_h = self.atom_embedding(z.clamp(0, self.max_atomic_number)).to(dtype=pos.dtype) * mask.unsqueeze(-1)
        padded_atom_h = pos.new_zeros(batch_size, self.max_atoms, self.atom_embedding_dim)
        padded_atom_h[:, :num_atoms] = atom_h
        padded_mask = pos.new_zeros(batch_size, self.max_atoms)
        padded_mask[:, :num_atoms] = mask.to(pos.dtype)
        return torch.cat(
            [
                padded_distance.reshape(batch_size, -1),
                padded_atom_h.reshape(batch_size, -1),
                padded_mask,
            ],
            dim=-1,
        )

    def prototype_descriptor(self, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_atoms, _ = pos.shape
        if num_atoms > self.max_atoms:
            raise ValueError(f"num_atoms={num_atoms} exceeds global_coeff max_atoms={self.max_atoms}")
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        valid_pair = (mask[:, :, None] & mask[:, None, :]).to(pos.dtype)
        distance = (diff * diff).sum(dim=-1).clamp_min(1e-24).sqrt() / max(self.distance_scale, 1e-12)
        padded_distance = pos.new_zeros(batch_size, self.max_atoms, self.max_atoms)
        padded_distance[:, :num_atoms, :num_atoms] = distance * valid_pair
        padded_mask = pos.new_zeros(batch_size, self.max_atoms)
        padded_mask[:, :num_atoms] = mask.to(pos.dtype)
        return torch.cat([padded_distance.reshape(batch_size, -1), padded_mask], dim=-1)

    def _initialize_prototype_descriptors(self, descriptor: torch.Tensor) -> None:
        if not self.use_prototype_memory or bool(self.prototype_descriptors_initialized):
            return
        with torch.no_grad():
            source = descriptor.detach().to(device=self.prototype_descriptors.device, dtype=self.prototype_descriptors.dtype)
            start = int(self.prototype_fill_count.item())
            if start < self.prototype_count:
                take = min(self.prototype_count - start, source.shape[0])
                self.prototype_descriptors[start : start + take].copy_(source[:take])
                self.prototype_fill_count.fill_(start + take)
            if int(self.prototype_fill_count.item()) >= self.prototype_count:
                self.prototype_descriptors_initialized.fill_(True)

    def prototype_coeff_matrix(self, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        if not self.use_prototype_memory or self.prototype_coefficients is None:
            return None
        descriptor = self.prototype_descriptor(pos, mask)
        self._initialize_prototype_descriptors(descriptor)
        query = descriptor.detach().to(dtype=self.prototype_descriptors.dtype)
        active_count = int(self.prototype_fill_count.item()) if self.prototype_count else 0
        active_count = max(1, min(active_count, self.prototype_count))
        prototypes = self.prototype_descriptors[:active_count].to(device=query.device, dtype=query.dtype)
        distances = ((query[:, None, :] - prototypes[None, :, :]) ** 2).mean(dim=-1)
        if self.prototype_assignment == "nearest":
            nearest = distances.argmin(dim=-1)
            coeff = self.prototype_coefficients[nearest]
        else:
            weights = torch.softmax(-self.prototype_temperature * distances, dim=-1)
            coeff = torch.einsum(
                "bp,pij->bij",
                weights.to(dtype=self.prototype_coefficients.dtype),
                self.prototype_coefficients[:active_count],
            )
        return coeff.to(device=pos.device, dtype=pos.dtype)

    def architecture_metadata(self) -> dict[str, object]:
        return {
            "actual_hidden_irreps": "global_invariant_scalars_to_vectors",
            "irreps_in": f"{self.atom_embedding_dim}x0e",
            "irreps_hidden": f"{self.global_context_dim}x0e",
            "irreps_out": "1x1o",
            "irreps_sh": "",
            "uses_non_scalar_hidden": False,
            "uses_spherical_harmonics_in_value": False,
            "force_head_type": "global_invariant_edge_coefficients",
            "force_head_irreps": "1x1o",
            "uses_relative_vector_fallback": True,
            "uses_pairwise_force_skip": True,
            "uses_atom_pair_embedding": bool(self.use_atom_pair_embedding),
            "atom_embedding_dim": int(self.atom_embedding_dim),
            "pair_embedding_dim": int(self.pair_embedding_dim),
            "edge_mlp_hidden_dim": int(self.edge_mlp_hidden_dim),
            "edge_mlp_layers": int(self.edge_mlp_layers),
            "graph_mode": self.graph_mode,
            "use_prototype_memory": bool(self.use_prototype_memory),
            "prototype_count": int(self.prototype_count),
            "prototype_assignment": self.prototype_assignment,
        }

    def forward(self, pos: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        if mask is None:
            mask = torch.ones(pos.shape[:2], dtype=torch.bool, device=pos.device)
        batch_size, num_atoms, _ = pos.shape
        graph = build_molecular_graph(
            pos,
            mask,
            cutoff_radius=self.cutoff_radius,
            max_neighbors=self.max_neighbors,
            graph_mode=self.graph_mode,
        )
        context = self.global_mlp(self.global_descriptor(pos, z, mask))
        coeff_matrix = self.coeff_matrix_head(context).view(batch_size, self.max_atoms, self.max_atoms)
        prototype_coeff = self.prototype_coeff_matrix(pos, mask)
        if prototype_coeff is not None:
            coeff_matrix = coeff_matrix + prototype_coeff
        if graph.edge_count == 0:
            scale = self.force_output_scale_value().to(device=pos.device, dtype=pos.dtype)
            return {
                "forces": pos.new_zeros(pos.shape),
                "energy": pos.new_zeros(batch_size, 1),
                "energy_raw": pos.new_zeros(batch_size, 1),
                "graph_stats": graph_stats(graph, self.cutoff_radius),
                "diagnostics": {"force_output_scale": scale, "last_hidden_norm": context.norm(dim=-1).mean()},
            }
        atom_h = self.atom_embedding(z.clamp(0, self.max_atomic_number)).to(dtype=pos.dtype)
        hi = atom_h[graph.batch, graph.dst]
        hj = atom_h[graph.batch, graph.src]
        pair_index = pair_index_from_z(z[graph.batch, graph.dst], z[graph.batch, graph.src], self.max_atomic_number)
        if self.pair_embedding is None:
            pair_h = pos.new_zeros(graph.edge_count, 0)
        else:
            pair_h = self.pair_embedding(pair_index).to(dtype=pos.dtype)
        radial = self.radial(graph.distances)
        edge_context = context[graph.batch]
        edge_features = torch.cat([edge_context, hi, hj, pair_h, radial], dim=-1)
        hidden = self.edge_mlp[:-1](edge_features)
        edge_coeff = self.edge_mlp[-1](hidden).squeeze(-1)
        matrix_coeff = coeff_matrix[graph.batch, graph.dst, graph.src]
        coeff = edge_coeff + matrix_coeff
        messages = coeff.unsqueeze(-1) * graph.edge_vec
        raw_forces = aggregate_to_padded_nodes(messages, graph) * mask.unsqueeze(-1)
        scale = self.force_output_scale_value().to(device=raw_forces.device, dtype=raw_forces.dtype)
        forces = raw_forces * scale
        return {
            "forces": forces,
            "energy": pos.new_zeros(batch_size, 1),
            "energy_raw": pos.new_zeros(batch_size, 1),
            "graph_stats": graph_stats(graph, self.cutoff_radius),
            "diagnostics": {
                "force_final_activation_norm": hidden.norm(dim=-1).mean(),
                "last_hidden_norm": context.norm(dim=-1).mean(),
                "message_norm_mean": messages.norm(dim=-1).mean(),
                "edge_message_norm_mean": messages.norm(dim=-1).mean(),
                "force_head_output_norm": coeff.norm(dim=-1).mean(),
                "force_output_scale": scale,
                "prototype_coeff_norm": (
                    prototype_coeff.norm(dim=(-1, -2)).mean() if prototype_coeff is not None else raw_forces.new_zeros(())
                ),
            },
        }


class InternalCoordinateEnergyMemorizer(nn.Module):
    """Invariant internal-coordinate energy network with autograd forces."""

    def __init__(
        self,
        max_atoms: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 4,
        atom_embedding_dim: int = 24,
        distance_scale: float = 5.0,
        use_angles: bool = False,
        cutoff_radius: float = 5.0,
        max_neighbors: int | None = None,
        graph_mode: str = "full",
        max_atomic_number: int = 100,
        training_mode: str = "direct_force",
        force_output_scale: float = 1.0,
        learnable_force_output_scale: bool = False,
        initial_force_output_scale: float | None = None,
        force_output_scale_regularization: float = 0.0,
        **_unused,
    ) -> None:
        super().__init__()
        self.max_atoms = int(max_atoms)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.atom_embedding_dim = int(atom_embedding_dim)
        self.distance_scale = float(distance_scale)
        self.use_angles = bool(use_angles)
        self.cutoff_radius = float(cutoff_radius)
        self.max_neighbors = max_neighbors
        self.graph_mode = str(graph_mode)
        self.max_atomic_number = int(max_atomic_number)
        self.training_mode = training_mode
        self.backbone_class = "internal_coordinate_energy"
        self.model_label = "internal_energy"
        self.lmax = 0
        self.hidden_irreps = "invariant_internal_coordinates"
        self.use_attention = False
        self.use_gate = False
        self.force_output_scale = float(force_output_scale)
        self.learnable_force_output_scale = bool(learnable_force_output_scale)
        self.initial_force_output_scale = float(initial_force_output_scale if initial_force_output_scale is not None else force_output_scale)
        self.force_output_scale_regularization = float(force_output_scale_regularization)
        self.atom_embedding = nn.Embedding(self.max_atomic_number + 1, self.atom_embedding_dim)
        angle_dim = self.max_atoms * self.max_atoms if self.use_angles else 0
        descriptor_dim = self.max_atoms * self.max_atoms + angle_dim + self.max_atoms * self.atom_embedding_dim + self.max_atoms
        self.energy_mlp = make_mlp(descriptor_dim, self.hidden_dim, 1, self.num_layers)
        if self.learnable_force_output_scale:
            initial = max(abs(self.initial_force_output_scale), 1e-12)
            self.force_output_log_scale = nn.Parameter(torch.tensor(math.log(initial), dtype=torch.float32))
        else:
            self.force_output_log_scale = None
        self.architecture_signature = "|".join(
            [
                "internal_energy",
                f"model_class={self.__class__.__name__}",
                f"backbone={self.backbone_class}",
                f"max_atoms={self.max_atoms}",
                f"hidden={self.hidden_dim}",
                f"layers={self.num_layers}",
                f"use_angles={self.use_angles}",
                f"graph_mode={self.graph_mode}",
                f"force_output_scale={self.force_output_scale:.6g}",
            ]
        )

    def force_output_scale_value(self) -> torch.Tensor:
        if self.force_output_log_scale is not None:
            return self.force_output_log_scale.exp()
        return torch.tensor(self.force_output_scale, dtype=torch.float32, device=self.atom_embedding.weight.device)

    def descriptor(self, pos: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_atoms, _ = pos.shape
        if num_atoms > self.max_atoms:
            raise ValueError(f"num_atoms={num_atoms} exceeds internal_energy max_atoms={self.max_atoms}")
        valid_pair = (mask[:, :, None] & mask[:, None, :]).to(pos.dtype)
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        distance = (diff * diff).sum(dim=-1).clamp_min(1e-24).sqrt() / max(self.distance_scale, 1e-12)
        padded_distance = pos.new_zeros(batch_size, self.max_atoms, self.max_atoms)
        padded_distance[:, :num_atoms, :num_atoms] = distance * valid_pair
        parts = [padded_distance.reshape(batch_size, -1)]
        if self.use_angles:
            centered = pos - (pos * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
            unit = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cosine = torch.einsum("bid,bjd->bij", unit, unit) * valid_pair
            padded_cosine = pos.new_zeros(batch_size, self.max_atoms, self.max_atoms)
            padded_cosine[:, :num_atoms, :num_atoms] = cosine
            parts.append(padded_cosine.reshape(batch_size, -1))
        atom_h = self.atom_embedding(z.clamp(0, self.max_atomic_number)).to(dtype=pos.dtype) * mask.unsqueeze(-1)
        padded_atom_h = pos.new_zeros(batch_size, self.max_atoms, self.atom_embedding_dim)
        padded_atom_h[:, :num_atoms] = atom_h
        padded_mask = pos.new_zeros(batch_size, self.max_atoms)
        padded_mask[:, :num_atoms] = mask.to(pos.dtype)
        parts.extend([padded_atom_h.reshape(batch_size, -1), padded_mask])
        return torch.cat(parts, dim=-1)

    def architecture_metadata(self) -> dict[str, object]:
        return {
            "actual_hidden_irreps": "invariant_internal_coordinates",
            "irreps_in": f"{self.atom_embedding_dim}x0e",
            "irreps_hidden": f"{self.hidden_dim}x0e",
            "irreps_out": "0e_energy_gradient_to_1o",
            "irreps_sh": "",
            "uses_non_scalar_hidden": False,
            "uses_spherical_harmonics_in_value": False,
            "force_head_type": "negative_energy_gradient",
            "force_head_irreps": "1x1o",
            "uses_relative_vector_fallback": False,
            "uses_pairwise_force_skip": False,
            "uses_atom_pair_embedding": False,
            "atom_embedding_dim": int(self.atom_embedding_dim),
            "pair_embedding_dim": 0,
            "edge_mlp_hidden_dim": int(self.hidden_dim),
            "edge_mlp_layers": int(self.num_layers),
            "graph_mode": self.graph_mode,
        }

    def forward(self, pos: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        if mask is None:
            mask = torch.ones(pos.shape[:2], dtype=torch.bool, device=pos.device)
        graph = build_molecular_graph(
            pos,
            mask,
            cutoff_radius=self.cutoff_radius,
            max_neighbors=self.max_neighbors,
            graph_mode=self.graph_mode,
        )
        work_pos = pos if pos.requires_grad else pos.detach().clone().requires_grad_(True)
        descriptor = self.descriptor(work_pos, z, mask)
        energy = self.energy_mlp(descriptor)
        grad = torch.autograd.grad(
            energy.sum(),
            work_pos,
            create_graph=self.training,
            retain_graph=True,
            allow_unused=False,
        )[0]
        scale = self.force_output_scale_value().to(device=work_pos.device, dtype=work_pos.dtype)
        forces = -grad * scale * mask.unsqueeze(-1)
        energy_raw = energy
        return {
            "forces": forces,
            "energy": energy,
            "energy_raw": energy_raw,
            "graph_stats": graph_stats(graph, self.cutoff_radius),
            "diagnostics": {
                "force_final_activation_norm": descriptor.norm(dim=-1).mean(),
                "last_hidden_norm": descriptor.norm(dim=-1).mean(),
                "message_norm_mean": forces.norm(dim=-1)[mask].mean() if mask.any() else forces.new_zeros(()),
                "edge_message_norm_mean": forces.norm(dim=-1)[mask].mean() if mask.any() else forces.new_zeros(()),
                "force_head_output_norm": forces.norm(dim=-1)[mask].mean() if mask.any() else forces.new_zeros(()),
                "force_output_scale": scale,
                "energy_output_norm": energy.norm(dim=-1).mean(),
                "energy_grad_norm": grad.norm(dim=-1)[mask].mean() if mask.any() else grad.new_zeros(()),
            },
        }


class PaiNNLiteForceModel(nn.Module):
    """Small scalar/vector message-passing diagnostic inspired by PaiNN."""

    def __init__(
        self,
        hidden_dim: int = 128,
        vector_channels: int = 16,
        num_layers: int = 3,
        radial_num_basis: int = 32,
        radial_hidden_dim: int = 192,
        pair_embedding_dim: int = 24,
        cutoff_radius: float = 5.0,
        max_neighbors: int | None = None,
        graph_mode: str = "full",
        max_atomic_number: int = 100,
        training_mode: str = "direct_force",
        output_head_init_scale: float = 1.0,
        force_output_scale: float = 1.0,
        learnable_force_output_scale: bool = False,
        initial_force_output_scale: float | None = None,
        force_output_scale_regularization: float = 0.0,
        **_unused,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.vector_channels = int(vector_channels)
        self.num_layers = int(num_layers)
        self.radial_num_basis = int(radial_num_basis)
        self.radial_hidden_dim = int(radial_hidden_dim)
        self.pair_embedding_dim = int(pair_embedding_dim)
        self.cutoff_radius = float(cutoff_radius)
        self.max_neighbors = max_neighbors
        self.graph_mode = str(graph_mode)
        self.max_atomic_number = int(max_atomic_number)
        self.training_mode = training_mode
        self.backbone_class = "painn_lite_vector_message_passing"
        self.model_label = "painn_lite"
        self.lmax = 1
        self.hidden_irreps = f"{self.hidden_dim}x0e+{self.vector_channels}x1o"
        self.use_attention = False
        self.use_gate = True
        self.output_head_init_scale = float(output_head_init_scale)
        self.force_output_scale = float(force_output_scale)
        self.learnable_force_output_scale = bool(learnable_force_output_scale)
        self.initial_force_output_scale = float(initial_force_output_scale if initial_force_output_scale is not None else force_output_scale)
        self.force_output_scale_regularization = float(force_output_scale_regularization)
        self.atom_embedding = nn.Embedding(self.max_atomic_number + 1, self.hidden_dim)
        self.pair_embedding = nn.Embedding((self.max_atomic_number + 1) * (self.max_atomic_number + 1), self.pair_embedding_dim)
        self.radial = GaussianRadialBasis(self.radial_num_basis)
        edge_input = 2 * self.hidden_dim + self.pair_embedding_dim + self.radial_num_basis
        self.edge_mlps = nn.ModuleList(
            [make_mlp(edge_input, self.radial_hidden_dim, self.hidden_dim + 2 * self.vector_channels, 2) for _ in range(self.num_layers)]
        )
        self.scalar_updates = nn.ModuleList(
            [make_mlp(2 * self.hidden_dim + self.vector_channels, self.radial_hidden_dim, self.hidden_dim, 1) for _ in range(self.num_layers)]
        )
        self.force_head = nn.Linear(self.vector_channels, 1, bias=False)
        if self.learnable_force_output_scale:
            initial = max(abs(self.initial_force_output_scale), 1e-12)
            self.force_output_log_scale = nn.Parameter(torch.tensor(math.log(initial), dtype=torch.float32))
        else:
            self.force_output_log_scale = None
        self._apply_output_head_init_scale()
        self.architecture_signature = "|".join(
            [
                "painn_lite",
                f"model_class={self.__class__.__name__}",
                f"backbone={self.backbone_class}",
                f"hidden={self.hidden_dim}",
                f"vector_channels={self.vector_channels}",
                f"layers={self.num_layers}",
                f"graph_mode={self.graph_mode}",
                f"force_output_scale={self.force_output_scale:.6g}",
            ]
        )

    def _apply_output_head_init_scale(self) -> None:
        if self.output_head_init_scale != 1.0:
            with torch.no_grad():
                self.force_head.weight.mul_(self.output_head_init_scale)
                if self.force_head.bias is not None:
                    self.force_head.bias.mul_(self.output_head_init_scale)

    def force_output_scale_value(self) -> torch.Tensor:
        if self.force_output_log_scale is not None:
            return self.force_output_log_scale.exp()
        return torch.tensor(self.force_output_scale, dtype=torch.float32, device=self.atom_embedding.weight.device)

    def architecture_metadata(self) -> dict[str, object]:
        return {
            "actual_hidden_irreps": self.hidden_irreps,
            "irreps_in": f"{self.hidden_dim}x0e",
            "irreps_hidden": self.hidden_irreps,
            "irreps_out": "1x1o",
            "irreps_sh": "1x1o",
            "uses_non_scalar_hidden": True,
            "uses_spherical_harmonics_in_value": False,
            "force_head_type": "vector_channel_linear",
            "force_head_irreps": "1x1o",
            "uses_relative_vector_fallback": False,
            "uses_pairwise_force_skip": False,
            "uses_atom_pair_embedding": True,
            "atom_embedding_dim": int(self.hidden_dim),
            "pair_embedding_dim": int(self.pair_embedding_dim),
            "edge_mlp_hidden_dim": int(self.radial_hidden_dim),
            "edge_mlp_layers": 2,
            "graph_mode": self.graph_mode,
        }

    def forward(self, pos: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        if mask is None:
            mask = torch.ones(pos.shape[:2], dtype=torch.bool, device=pos.device)
        graph = build_molecular_graph(
            pos,
            mask,
            cutoff_radius=self.cutoff_radius,
            max_neighbors=self.max_neighbors,
            graph_mode=self.graph_mode,
        )
        s = self.atom_embedding(z.clamp(0, self.max_atomic_number)).to(dtype=pos.dtype) * mask.unsqueeze(-1)
        v = pos.new_zeros(pos.shape[0], pos.shape[1], self.vector_channels, 3)
        if graph.edge_count:
            unit = graph.edge_vec / graph.distances.unsqueeze(-1).clamp_min(1e-12)
            pair_index = pair_index_from_z(z[graph.batch, graph.dst], z[graph.batch, graph.src], self.max_atomic_number)
            pair_h = self.pair_embedding(pair_index).to(dtype=pos.dtype)
            radial = self.radial(graph.distances)
            for edge_mlp, scalar_update in zip(self.edge_mlps, self.scalar_updates):
                si = s[graph.batch, graph.dst]
                sj = s[graph.batch, graph.src]
                edge_features = torch.cat([si, sj, pair_h, radial], dim=-1)
                raw = edge_mlp(edge_features)
                scalar_msg, vec_dir, vec_gate = torch.split(raw, [self.hidden_dim, self.vector_channels, self.vector_channels], dim=-1)
                agg_scalar = aggregate_to_padded_nodes(scalar_msg, graph)
                v_src = v[graph.batch, graph.src]
                vector_msg = v_src * vec_gate.unsqueeze(-1) + vec_dir.unsqueeze(-1) * unit.unsqueeze(1)
                agg_vector = aggregate_to_padded_nodes(vector_msg.reshape(graph.edge_count, -1), graph).view(
                    pos.shape[0], pos.shape[1], self.vector_channels, 3
                )
                v = v + agg_vector * mask[:, :, None, None]
                v_norm = v.norm(dim=-1)
                s = s + scalar_update(torch.cat([s, agg_scalar, v_norm], dim=-1)) * mask.unsqueeze(-1)
        weights = self.force_head(v.transpose(2, 3)).squeeze(-1)
        raw_forces = weights * mask.unsqueeze(-1)
        scale = self.force_output_scale_value().to(device=raw_forces.device, dtype=raw_forces.dtype)
        forces = raw_forces * scale
        return {
            "forces": forces,
            "energy": pos.new_zeros(pos.shape[0], 1),
            "energy_raw": pos.new_zeros(pos.shape[0], 1),
            "graph_stats": graph_stats(graph, self.cutoff_radius),
            "diagnostics": {
                "force_final_activation_norm": v.norm(dim=-1)[mask].mean() if mask.any() else v.new_zeros(()),
                "last_hidden_norm": s.norm(dim=-1)[mask].mean() if mask.any() else s.new_zeros(()),
                "message_norm_mean": v.norm(dim=-1)[mask].mean() if mask.any() else v.new_zeros(()),
                "edge_message_norm_mean": v.norm(dim=-1)[mask].mean() if mask.any() else v.new_zeros(()),
                "force_head_output_norm": raw_forces.norm(dim=-1)[mask].mean() if mask.any() else raw_forces.new_zeros(()),
                "force_output_scale": scale,
            },
        }


class MolecularCoordinateMLPMemorizer(nn.Module):
    """Non-equivariant coordinate memorizer for overfit diagnostics only."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        max_atoms: int = 64,
        training_mode: str = "direct_force",
        force_output_scale: float = 1.0,
        learnable_force_output_scale: bool = False,
        initial_force_output_scale: float | None = None,
        force_output_scale_regularization: float = 0.0,
        **_unused,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_atoms = int(max_atoms)
        self.training_mode = training_mode
        self.backbone_class = "non_equivariant_coordinate_mlp"
        self.lmax = None
        self.hidden_irreps = "non_equivariant"
        self.use_attention = False
        self.use_gate = False
        self.force_output_scale = float(force_output_scale)
        self.learnable_force_output_scale = bool(learnable_force_output_scale)
        self.initial_force_output_scale = float(initial_force_output_scale if initial_force_output_scale is not None else force_output_scale)
        self.force_output_scale_regularization = float(force_output_scale_regularization)
        input_dim = self.max_atoms * 5
        output_dim = self.max_atoms * 3
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(self.num_layers):
            layers.extend([nn.Linear(dim, self.hidden_dim), nn.SiLU()])
            dim = self.hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)
        if self.learnable_force_output_scale:
            initial = max(abs(self.initial_force_output_scale), 1e-12)
            self.force_output_log_scale = nn.Parameter(torch.tensor(math.log(initial), dtype=torch.float32))
        else:
            self.force_output_log_scale = None
        self.architecture_signature = "|".join(
            [
                "mlp_memorizer",
                f"model_class={self.__class__.__name__}",
                f"backbone={self.backbone_class}",
                f"hidden={self.hidden_dim}",
                f"layers={self.num_layers}",
                f"max_atoms={self.max_atoms}",
                f"force_output_scale={self.force_output_scale:.6g}",
                f"learnable_force_output_scale={self.learnable_force_output_scale}",
                f"initial_force_output_scale={self.initial_force_output_scale:.6g}",
            ]
        )

    def force_output_scale_value(self) -> torch.Tensor:
        if self.force_output_log_scale is not None:
            return self.force_output_log_scale.exp()
        first = next(self.parameters())
        return torch.tensor(self.force_output_scale, dtype=first.dtype, device=first.device)

    def forward(self, pos: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        if mask is None:
            mask = torch.ones(pos.shape[:2], dtype=torch.bool, device=pos.device)
        batch_size, num_atoms, _ = pos.shape
        if num_atoms > self.max_atoms:
            raise ValueError(f"num_atoms={num_atoms} exceeds mlp_memorizer max_atoms={self.max_atoms}")
        padded_pos = pos.new_zeros(batch_size, self.max_atoms, 3)
        padded_z = pos.new_zeros(batch_size, self.max_atoms, 1)
        padded_mask = pos.new_zeros(batch_size, self.max_atoms, 1)
        padded_pos[:, :num_atoms] = pos
        padded_z[:, :num_atoms, 0] = z.to(pos.dtype) / 100.0
        padded_mask[:, :num_atoms, 0] = mask.to(pos.dtype)
        features = torch.cat([padded_pos.reshape(batch_size, -1), padded_z.reshape(batch_size, -1), padded_mask.reshape(batch_size, -1)], dim=-1)
        hidden = features
        for layer in self.net[:-1]:
            hidden = layer(hidden)
        raw = self.net[-1](hidden).view(batch_size, self.max_atoms, 3)[:, :num_atoms]
        scale = self.force_output_scale_value().to(device=raw.device, dtype=raw.dtype)
        forces = raw * scale * mask.unsqueeze(-1)
        return {
            "forces": forces,
            "energy": pos.new_zeros(batch_size, 1),
            "energy_raw": pos.new_zeros(batch_size, 1),
            "graph_stats": {
                "average_neighbors": 0.0,
                "edge_count_mean": 0.0,
                "edge_count_max": 0.0,
                "graph_build_time_sec": 0.0,
            },
            "diagnostics": {
                "force_final_activation_norm": hidden.norm(dim=-1).mean(),
                "force_output_scale": scale,
            },
        }


def parameter_count_by_module(model: nn.Module) -> dict[str, int]:
    counts = {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()}
    counts["_total"] = sum(p.numel() for p in model.parameters())
    return counts


def molecular_model_identity(model: nn.Module) -> dict[str, object]:
    architecture = model.architecture_metadata() if hasattr(model, "architecture_metadata") else {}
    return {
        "model_class": model.__class__.__name__,
        "backbone_class": str(getattr(model, "backbone_class", model.__class__.__name__)),
        "architecture_signature": str(getattr(model, "architecture_signature", model.__class__.__name__)),
        "training_mode": str(getattr(model, "training_mode", "")),
        "lmax": getattr(model, "lmax", None),
        "hidden_irreps": str(getattr(model, "hidden_irreps", "")),
        "use_attention": bool(getattr(model, "use_attention", False)),
        "use_gate": bool(getattr(model, "use_gate", False)),
        "cutoff_radius": float(getattr(model, "cutoff_radius", float("nan"))),
        "force_output_scale": float(getattr(model, "force_output_scale", 1.0)),
        "learnable_force_output_scale": bool(getattr(model, "learnable_force_output_scale", False)),
        "initial_force_output_scale": float(getattr(model, "initial_force_output_scale", getattr(model, "force_output_scale", 1.0))),
        "graph_mode": str(getattr(model, "graph_mode", "")),
        "actual_hidden_irreps": str(architecture.get("actual_hidden_irreps", getattr(model, "hidden_irreps", ""))),
        "irreps_in": str(architecture.get("irreps_in", "")),
        "irreps_hidden": str(architecture.get("irreps_hidden", getattr(model, "hidden_irreps", ""))),
        "irreps_out": str(architecture.get("irreps_out", "")),
        "irreps_sh": str(architecture.get("irreps_sh", "")),
        "uses_non_scalar_hidden": bool(architecture.get("uses_non_scalar_hidden", False)),
        "uses_spherical_harmonics_in_value": bool(architecture.get("uses_spherical_harmonics_in_value", False)),
        "force_head_type": str(architecture.get("force_head_type", "")),
        "force_head_irreps": str(architecture.get("force_head_irreps", "")),
        "uses_relative_vector_fallback": bool(architecture.get("uses_relative_vector_fallback", False)),
        "uses_pairwise_force_skip": bool(architecture.get("uses_pairwise_force_skip", False)),
        "uses_atom_pair_embedding": bool(architecture.get("uses_atom_pair_embedding", False)),
        "atom_embedding_dim": architecture.get("atom_embedding_dim"),
        "pair_embedding_dim": architecture.get("pair_embedding_dim"),
        "edge_mlp_hidden_dim": architecture.get("edge_mlp_hidden_dim"),
        "edge_mlp_layers": architecture.get("edge_mlp_layers"),
        "uses_global_context": bool(architecture.get("uses_global_context", False)),
        "global_context_dim": architecture.get("global_context_dim"),
        "global_context_type": str(architecture.get("global_context_type", "")),
        "use_prototype_memory": bool(architecture.get("use_prototype_memory", False)),
        "prototype_count": architecture.get("prototype_count", 0),
        "prototype_assignment": str(architecture.get("prototype_assignment", "")),
        "parameter_count_by_module": parameter_count_by_module(model),
    }


def build_molecular_model(config: dict) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    dataset_cfg = config.get("dataset", {})
    name = model_cfg.pop("name", "se3_transformer")
    model_cfg.setdefault("cutoff_radius", dataset_cfg.get("cutoff_radius", 5.0))
    model_cfg.setdefault("max_neighbors", dataset_cfg.get("max_neighbors"))
    model_cfg.setdefault("graph_mode", dataset_cfg.get("graph_mode", "cutoff"))
    training_mode = str(config.get("training", {}).get("mode", model_cfg.pop("training_mode", "direct_force")))
    model_cfg["training_mode"] = training_mode
    if name in {"egnn", "molecular_egnn"}:
        return MolecularEGNN(**model_cfg)
    if name in {"tfn", "baseline_tfn", "molecular_tfn"} or model_cfg.get("use_attention") is False:
        return MolecularTFNConv(**model_cfg)
    if name in {"radial", "pairwise_radial", "radial_baseline"}:
        return MolecularRadialForceBaseline(**model_cfg)
    if name in {"radial_pair", "high_capacity_radial_pair"}:
        return HighCapacityRadialForceModel(**model_cfg)
    if name in {"global_context_radial", "global_context_radial_pair"}:
        return GlobalContextRadialForceModel(**model_cfg)
    if name in {"global_coeff", "global_invariant_coefficients"}:
        return GlobalInvariantCoefficientForceModel(**model_cfg)
    if name in {"internal_energy", "internal_coordinate_energy"}:
        return InternalCoordinateEnergyMemorizer(**model_cfg)
    if name in {"painn_lite", "painn"}:
        return PaiNNLiteForceModel(**model_cfg)
    if name in {"mlp_memorizer", "coordinate_mlp_memorizer"}:
        return MolecularCoordinateMLPMemorizer(**model_cfg)
    if name in {"se3_full", "se3_full_l1", "se3_full_l2", "molecular_se3_full_irrep"}:
        return MolecularFullIrrepSE3ForceTransformer(**model_cfg)
    if name in {"se3_transformer", "molecular_se3"}:
        return MolecularSE3ForceTransformer(**model_cfg)
    raise ValueError(f"unknown molecular model: {name}")
