import torch

from se3force.geometry import apply_transform, random_rotation_matrix, random_translation, relative_error
from se3force.geometry.irreps import transform_features
from se3force.models.equivariant import SE3AttentionHead


def test_se3_attention_head_is_equivariant():
    irreps_in = "4x0e+2x1o+1x2e"
    irreps_value = "3x0e+2x1o"
    head = SE3AttentionHead(
        irreps_in=irreps_in,
        irreps_value=irreps_value,
        lmax=2,
        num_query_channels=4,
        radial_num_basis=6,
        radial_hidden_dim=16,
    )
    head.eval()
    x = torch.randn(2, 5, 3)
    features = torch.randn(2, 5, head.irreps_in.dim)
    R = random_rotation_matrix(2)
    t = random_translation(2)
    out = head(x, features)
    out_rt = head(apply_transform(x, R, t), transform_features(features, irreps_in, R))
    assert relative_error(out_rt, transform_features(out, irreps_value, R)) < 2e-4
