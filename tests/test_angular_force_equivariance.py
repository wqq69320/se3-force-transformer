import torch

from se3force.data.angular_potential import angular_energy, angular_force
from se3force.geometry import apply_rotation, apply_transform, random_rotation_matrix, random_translation, relative_error


def test_angular_energy_is_rigid_motion_invariant():
    x = torch.randn(5, 3)
    mass = 0.5 + torch.rand(5)
    R = random_rotation_matrix(1)
    t = random_translation(1)
    e = angular_energy(x, mass)
    e_rt = angular_energy(apply_transform(x.unsqueeze(0), R, t).squeeze(0), mass)
    assert torch.allclose(e, e_rt, atol=1e-5)


def test_angular_force_target_is_equivariant():
    x = torch.randn(5, 3)
    mass = 0.5 + torch.rand(5)
    R = random_rotation_matrix(1)
    t = random_translation(1)
    force = angular_force(x, mass)
    force_rt = angular_force(apply_transform(x.unsqueeze(0), R, t).squeeze(0), mass)
    assert relative_error(force_rt.unsqueeze(0), apply_rotation(force.unsqueeze(0), R)) < 5e-5
