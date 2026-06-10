import torch

from se3force.data.central_force import central_force
from se3force.geometry import apply_rotation, apply_transform, random_rotation_matrix, random_translation, relative_error


def test_central_force_target_is_equivariant():
    x = torch.randn(4, 6, 3)
    mass = 0.5 + torch.rand(4, 6)
    R = random_rotation_matrix(4)
    t = random_translation(4)
    force = central_force(x, mass)
    force_rt = central_force(apply_transform(x, R, t), mass)
    assert relative_error(force_rt, apply_rotation(force, R)) < 1e-5
