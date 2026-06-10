import torch

from se3force.geometry import apply_transform, pairwise_dist, pairwise_relative, random_rotation_matrix, random_translation


def test_pairwise_relative_is_translation_invariant():
    x = torch.randn(3, 5, 3)
    t = random_translation(3)
    assert torch.allclose(pairwise_relative(x + t), pairwise_relative(x), atol=1e-6)


def test_pairwise_distances_are_rigid_motion_invariant():
    x = torch.randn(3, 5, 3)
    R = random_rotation_matrix(3)
    t = random_translation(3)
    assert torch.allclose(pairwise_dist(apply_transform(x, R, t)), pairwise_dist(x), atol=1e-5)
