import torch

from se3force.geometry import apply_transform, random_rotation_matrix, random_translation, relative_error
from se3force.geometry.irreps import transform_features
from se3force.models.equivariant import TFNConv


def test_tfn_layer_is_equivariant():
    irreps_in = "3x0e+2x1o"
    irreps_out = "2x0e+2x1o+1x2e"
    layer = TFNConv(irreps_in, irreps_out, lmax=2, radial_num_basis=6, radial_hidden_dim=16)
    layer.eval()
    x = torch.randn(2, 5, 3)
    features = torch.randn(2, 5, layer.irreps_in.dim)
    R = random_rotation_matrix(2)
    t = random_translation(2)
    out = layer(x, features)
    out_rt = layer(apply_transform(x, R, t), transform_features(features, irreps_in, R))
    assert relative_error(out_rt, transform_features(out, irreps_out, R)) < 1e-4
