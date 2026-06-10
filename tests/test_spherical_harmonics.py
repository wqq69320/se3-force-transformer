import torch
from e3nn import o3

from se3force.geometry.irreps import spherical_harmonics_irreps, transform_features
from se3force.geometry.rotations import apply_rotation, random_rotation_matrix


def test_spherical_harmonics_transform_by_wigner_d():
    B, N = 3, 7
    irreps = spherical_harmonics_irreps(3)
    r = torch.randn(B, N, 3)
    R = random_rotation_matrix(B)
    y = o3.spherical_harmonics(irreps, r.reshape(-1, 3), normalize=True, normalization="component").view(B, N, -1)
    y_rot = o3.spherical_harmonics(
        irreps, apply_rotation(r, R).reshape(-1, 3), normalize=True, normalization="component"
    ).view(B, N, -1)
    assert torch.allclose(y_rot, transform_features(y, irreps, R), atol=1e-4, rtol=1e-4)
