from __future__ import annotations

import torch

from se3force.geometry.metrics import relative_error
from se3force.geometry.rotations import apply_rotation, random_rotation_matrix, random_translation
from se3force.geometry.transforms import apply_transform


@torch.no_grad()
def model_equivariance_error(model, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    model.eval()
    R = random_rotation_matrix(x.shape[0], device=x.device, dtype=x.dtype)
    t = random_translation(x.shape[0], device=x.device, dtype=x.dtype)
    pred = model(x, z)
    pred_transformed = model(apply_transform(x, R, t), z)
    expected = apply_rotation(pred, R)
    return relative_error(pred_transformed, expected)
