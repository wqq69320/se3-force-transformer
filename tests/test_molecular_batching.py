import torch

from se3force.data.molecular import molecular_collate


def test_variable_n_molecular_batching_preserves_masks_and_atomic_numbers():
    samples = [
        {"pos": torch.randn(3, 3), "z": torch.tensor([1, 6, 8]), "energy": torch.tensor([1.0]), "forces": torch.randn(3, 3), "molecule_id": "a", "frame_id": 0},
        {"pos": torch.randn(5, 3), "z": torch.tensor([1, 1, 6, 7, 8]), "energy": torch.tensor([2.0]), "forces": torch.randn(5, 3), "molecule_id": "b", "frame_id": 1},
    ]
    batch = molecular_collate(samples)
    assert batch["pos"].shape == (2, 5, 3)
    assert batch["mask"].sum(dim=1).tolist() == [3, 5]
    assert batch["z"][0, :3].tolist() == [1, 6, 8]
    assert batch["z"][0, 3:].tolist() == [0, 0]
