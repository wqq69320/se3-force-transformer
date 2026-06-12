import torch

from se3force.models.molecular import MolecularEGNN, MolecularSE3ForceTransformer


def test_molecular_direct_force_model_shapes():
    pos = torch.randn(2, 5, 3)
    z = torch.tensor([[1, 6, 8, 1, 0], [6, 6, 1, 8, 7]])
    mask = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
    for model in [
        MolecularEGNN(hidden_dim=16, num_layers=1, radial_num_basis=5, radial_hidden_dim=12, cutoff_radius=4.0),
        MolecularSE3ForceTransformer(channels_by_l={0: 16}, num_layers=1, radial_num_basis=5, radial_hidden_dim=12, cutoff_radius=4.0),
    ]:
        out = model(pos, z, mask)
        assert out["forces"].shape == pos.shape
        assert out["energy"].shape == (2, 1)
        assert torch.all(out["forces"][~mask] == 0)
