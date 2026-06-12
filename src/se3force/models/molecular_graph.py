from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MolecularGraph:
    batch: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    edge_vec: torch.Tensor
    distances: torch.Tensor
    num_batches: int
    num_nodes: int
    mask: torch.Tensor
    build_time_sec: float

    @property
    def edge_count(self) -> int:
        return int(self.src.numel())


@dataclass(frozen=True)
class PackedMolecularGraph:
    batch: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    edge_vec: torch.Tensor
    distances: torch.Tensor
    num_batches: int
    num_nodes: int
    build_time_sec: float

    @property
    def edge_count(self) -> int:
        return int(self.src.numel())


def _empty_graph(pos: torch.Tensor, mask: torch.Tensor, start: float) -> MolecularGraph:
    empty_long = torch.empty(0, device=pos.device, dtype=torch.long)
    empty_vec = torch.empty(0, 3, device=pos.device, dtype=pos.dtype)
    empty_dist = torch.empty(0, device=pos.device, dtype=pos.dtype)
    return MolecularGraph(
        batch=empty_long,
        src=empty_long,
        dst=empty_long,
        edge_vec=empty_vec,
        distances=empty_dist,
        num_batches=pos.shape[0],
        num_nodes=pos.shape[1],
        mask=mask,
        build_time_sec=time.perf_counter() - start,
    )


def build_molecular_graph(
    pos: torch.Tensor,
    mask: torch.Tensor | None = None,
    cutoff_radius: float | None = None,
    max_neighbors: int | None = None,
    exclude_self: bool = True,
) -> MolecularGraph:
    """Build directed local-index edges for padded molecular batches."""
    start = time.perf_counter()
    B, N, _ = pos.shape
    device = pos.device
    if mask is None:
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
    cutoff = float(cutoff_radius) if cutoff_radius is not None else None
    batch_chunks = []
    src_chunks = []
    dst_chunks = []

    for b in range(B):
        valid = torch.nonzero(mask[b], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        local_pos = pos[b, valid]
        dmat = torch.cdist(local_pos, local_pos)
        keep = torch.ones_like(dmat, dtype=torch.bool)
        if exclude_self:
            keep.fill_diagonal_(False)
        if cutoff is not None:
            keep &= dmat <= cutoff
        if max_neighbors is not None and max_neighbors > 0:
            limited = torch.zeros_like(keep)
            for dst_local in range(valid.numel()):
                candidates = torch.nonzero(keep[dst_local], as_tuple=False).flatten()
                if candidates.numel() == 0:
                    continue
                order = torch.argsort(dmat[dst_local, candidates])[:max_neighbors]
                limited[dst_local, candidates[order]] = True
            keep = limited
        dst_local, src_local = torch.nonzero(keep, as_tuple=True)
        if src_local.numel() == 0:
            continue
        batch_chunks.append(torch.full_like(src_local, b))
        src_chunks.append(valid[src_local])
        dst_chunks.append(valid[dst_local])

    if not src_chunks:
        return _empty_graph(pos, mask, start)

    batch = torch.cat(batch_chunks)
    src = torch.cat(src_chunks)
    dst = torch.cat(dst_chunks)
    edge_vec = pos[batch, src] - pos[batch, dst]
    distances = edge_vec.norm(dim=-1).clamp_min(1e-12)
    return MolecularGraph(
        batch=batch,
        src=src,
        dst=dst,
        edge_vec=edge_vec,
        distances=distances,
        num_batches=B,
        num_nodes=N,
        mask=mask,
        build_time_sec=time.perf_counter() - start,
    )


def aggregate_to_padded_nodes(messages: torch.Tensor, graph: MolecularGraph) -> torch.Tensor:
    out = messages.new_zeros(graph.num_batches * graph.num_nodes, messages.shape[-1])
    if messages.numel() > 0:
        index = graph.batch * graph.num_nodes + graph.dst
        out.index_add_(0, index, messages)
    return out.view(graph.num_batches, graph.num_nodes, messages.shape[-1])


def build_packed_molecular_graph(
    pos: torch.Tensor,
    batch_index: torch.Tensor,
    cutoff_radius: float | None = None,
    max_neighbors: int | None = None,
    exclude_self: bool = True,
) -> PackedMolecularGraph:
    """Build directed cutoff edges for packed nodes with global node indices."""
    start = time.perf_counter()
    num_batches = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    src_chunks = []
    dst_chunks = []
    batch_chunks = []
    cutoff = float(cutoff_radius) if cutoff_radius is not None else None
    for b in range(num_batches):
        nodes = torch.nonzero(batch_index == b, as_tuple=False).flatten()
        if nodes.numel() == 0:
            continue
        local_pos = pos[nodes]
        dmat = torch.cdist(local_pos, local_pos)
        keep = torch.ones_like(dmat, dtype=torch.bool)
        if exclude_self:
            keep.fill_diagonal_(False)
        if cutoff is not None:
            keep &= dmat <= cutoff
        if max_neighbors is not None and max_neighbors > 0:
            limited = torch.zeros_like(keep)
            for dst_local in range(nodes.numel()):
                candidates = torch.nonzero(keep[dst_local], as_tuple=False).flatten()
                if candidates.numel() == 0:
                    continue
                order = torch.argsort(dmat[dst_local, candidates])[:max_neighbors]
                limited[dst_local, candidates[order]] = True
            keep = limited
        dst_local, src_local = torch.nonzero(keep, as_tuple=True)
        if src_local.numel() == 0:
            continue
        src_chunks.append(nodes[src_local])
        dst_chunks.append(nodes[dst_local])
        batch_chunks.append(torch.full_like(src_local, b))
    if not src_chunks:
        empty = torch.empty(0, dtype=torch.long, device=pos.device)
        return PackedMolecularGraph(
            batch=empty,
            src=empty,
            dst=empty,
            edge_vec=torch.empty(0, 3, dtype=pos.dtype, device=pos.device),
            distances=torch.empty(0, dtype=pos.dtype, device=pos.device),
            num_batches=num_batches,
            num_nodes=pos.shape[0],
            build_time_sec=time.perf_counter() - start,
        )
    src = torch.cat(src_chunks)
    dst = torch.cat(dst_chunks)
    batch = torch.cat(batch_chunks)
    edge_vec = pos[src] - pos[dst]
    return PackedMolecularGraph(
        batch=batch,
        src=src,
        dst=dst,
        edge_vec=edge_vec,
        distances=edge_vec.norm(dim=-1).clamp_min(1e-12),
        num_batches=num_batches,
        num_nodes=pos.shape[0],
        build_time_sec=time.perf_counter() - start,
    )


def aggregate_to_packed_nodes(messages: torch.Tensor, graph: PackedMolecularGraph) -> torch.Tensor:
    out = messages.new_zeros(graph.num_nodes, messages.shape[-1])
    if messages.numel() > 0:
        out.index_add_(0, graph.dst, messages)
    return out


def graph_stats(graph: MolecularGraph, cutoff_radius: float | None = None) -> dict[str, float]:
    counts = torch.zeros(graph.num_batches * graph.num_nodes, device=graph.distances.device)
    if graph.edge_count:
        index = graph.batch * graph.num_nodes + graph.dst
        counts.index_add_(0, index, torch.ones_like(graph.distances))
    valid_counts = counts.view(graph.num_batches, graph.num_nodes)[graph.mask]
    edge_counts = []
    for b in range(graph.num_batches):
        edge_counts.append(int((graph.batch == b).sum()))
    return {
        "average_neighbors": float(valid_counts.mean().item()) if valid_counts.numel() else 0.0,
        "max_neighbors": float(valid_counts.max().item()) if valid_counts.numel() else 0.0,
        "edge_count_mean": float(sum(edge_counts) / max(1, len(edge_counts))),
        "edge_count_max": float(max(edge_counts) if edge_counts else 0),
        "cutoff_radius": float(cutoff_radius) if cutoff_radius is not None else float("nan"),
        "graph_build_time_sec": float(graph.build_time_sec),
    }
