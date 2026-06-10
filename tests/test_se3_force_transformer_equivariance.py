import torch

from se3force.evaluation import model_equivariance_error
from se3force.models.equivariant import SE3ForceTransformer


def test_se3_force_transformer_force_output_is_equivariant():
    model = SE3ForceTransformer(
        scalar_input_dim=1,
        lmax=2,
        channels_by_l={0: 8, 1: 4, 2: 2},
        num_layers=2,
        num_heads=2,
        num_query_channels=4,
        radial_num_basis=6,
        radial_hidden_dim=16,
    )
    x = torch.randn(2, 5, 3)
    z = torch.randn(2, 5, 1)
    err = model_equivariance_error(model, x, z)
    assert err < 5e-4
