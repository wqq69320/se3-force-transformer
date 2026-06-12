import torch

from se3force.models.molecular_graph import build_molecular_graph, build_packed_molecular_graph, graph_stats


def test_cutoff_graph_excludes_self_edges_and_respects_cutoff():
    pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    graph = build_molecular_graph(pos, mask, cutoff_radius=1.5)
    pairs = set(zip(graph.dst.tolist(), graph.src.tolist()))
    assert pairs == {(0, 1), (1, 0)}
    assert all(src != dst for src, dst in zip(graph.src.tolist(), graph.dst.tolist()))
    stats = graph_stats(graph, cutoff_radius=1.5)
    assert stats["edge_count_mean"] == 2
    assert stats["average_neighbors"] > 0


def test_packed_cutoff_graph_supports_variable_size_molecules():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ]
    )
    batch = torch.tensor([0, 0, 0, 1, 1])
    graph = build_packed_molecular_graph(pos, batch, cutoff_radius=1.5)
    assert graph.num_batches == 2
    assert graph.num_nodes == 5
    assert set(zip(graph.dst.tolist(), graph.src.tolist())) == {(0, 1), (1, 0), (3, 4), (4, 3)}
