import torch

from se3force.geometry import apply_rotation, apply_transform, random_rotation_matrix, random_translation, relative_error
from se3force.models.molecular import MolecularSE3ForceTransformer


def test_energy_force_mode_energy_invariant_and_force_equivariant():
    model = MolecularSE3ForceTransformer(
        channels_by_l={0: 16},
        num_layers=2,
        radial_num_basis=6,
        radial_hidden_dim=16,
        cutoff_radius=5.0,
        training_mode="energy_force",
    )
    pos = torch.randn(2, 4, 3, requires_grad=True)
    z = torch.tensor([[1, 6, 8, 1], [6, 6, 1, 8]])
    mask = torch.ones(2, 4, dtype=torch.bool)
    out = model(pos, z, mask)
    R = random_rotation_matrix(2)
    t = random_translation(2)
    pos_rt = apply_transform(pos.detach(), R, t).requires_grad_(True)
    out_rt = model(pos_rt, z, mask)
    assert torch.allclose(out["energy"].detach(), out_rt["energy"].detach(), atol=1e-5, rtol=1e-5)
    assert relative_error(out_rt["forces"].detach(), apply_rotation(out["forces"].detach(), R)) < 1e-4
