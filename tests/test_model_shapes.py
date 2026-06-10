import torch

from se3force.models.baselines import CoordMLP, EGNN, VanillaGraphTransformer
from se3force.models.equivariant import SE3ForceTransformer


def test_all_models_return_force_shape():
    B, N, C = 2, 5, 1
    x = torch.randn(B, N, 3)
    z = torch.randn(B, N, C)
    models = [
        CoordMLP(scalar_input_dim=C, num_nodes=N, hidden_dim=32, num_layers=2),
        VanillaGraphTransformer(scalar_input_dim=C, hidden_dim=32, num_layers=1, num_heads=4),
        EGNN(scalar_input_dim=C, hidden_dim=32, num_layers=2),
        SE3ForceTransformer(
            scalar_input_dim=C,
            lmax=2,
            channels_by_l={0: 8, 1: 4, 2: 2},
            num_layers=1,
            num_heads=2,
            num_query_channels=4,
            radial_num_basis=6,
            radial_hidden_dim=16,
        ),
    ]
    for model in models:
        assert model(x, z).shape == (B, N, 3)
