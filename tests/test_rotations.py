import torch

from se3force.geometry import random_rotation_matrix


def test_rotation_matrices_are_orthogonal_with_positive_determinant():
    R = random_rotation_matrix(16)
    eye = torch.eye(3).expand(16, 3, 3)
    assert torch.allclose(R.transpose(-1, -2) @ R, eye, atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(16), atol=1e-5)
