from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DenseEdges:
    batch: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    edge_vec: torch.Tensor
    distances: torch.Tensor
    num_batches: int
    num_nodes: int


def dense_edges(x: torch.Tensor, eps: float = 1e-12) -> DenseEdges:
    """Build dense directed edges j -> i, excluding self-edges."""
    B, N, _ = x.shape
    device = x.device
    nodes = torch.arange(N, device=device)
    dst_grid, src_grid = torch.meshgrid(nodes, nodes, indexing="ij")
    mask = dst_grid != src_grid
    src_one = src_grid[mask]
    dst_one = dst_grid[mask]
    batch = torch.arange(B, device=device).repeat_interleave(src_one.numel())
    src = src_one.repeat(B)
    dst = dst_one.repeat(B)
    edge_vec = x[batch, src] - x[batch, dst]
    distances = edge_vec.norm(dim=-1).clamp_min(eps)
    return DenseEdges(batch=batch, src=src, dst=dst, edge_vec=edge_vec, distances=distances, num_batches=B, num_nodes=N)


def aggregate_to_nodes(messages: torch.Tensor, edges: DenseEdges) -> torch.Tensor:
    out = messages.new_zeros(edges.num_batches * edges.num_nodes, messages.shape[-1])
    index = edges.batch * edges.num_nodes + edges.dst
    out.index_add_(0, index, messages)
    return out.view(edges.num_batches, edges.num_nodes, messages.shape[-1])


def edge_softmax(logits: torch.Tensor, edges: DenseEdges) -> torch.Tensor:
    """Softmax over incoming edges grouped by (batch, destination node)."""
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    out = torch.empty_like(logits)
    group_index = edges.batch * edges.num_nodes + edges.dst
    for group in range(edges.num_batches * edges.num_nodes):
        mask = group_index == group
        if mask.any():
            out[mask] = torch.softmax(logits[mask], dim=0)
    return out.squeeze(-1) if squeeze else out
