from __future__ import annotations

from collections.abc import Mapping

import torch
from e3nn import o3


def parity_for_l(l: int) -> str:
    return "e" if l % 2 == 0 else "o"


def build_hidden_irreps(
    num_scalar_channels: int | None = None,
    num_vector_channels: int | None = None,
    lmax: int = 2,
    channels_by_l: Mapping[int | str, int] | None = None,
) -> o3.Irreps:
    if channels_by_l is None:
        channels_by_l = {0: num_scalar_channels or 16}
        if lmax >= 1:
            channels_by_l[1] = num_vector_channels or 8
        for l in range(2, lmax + 1):
            channels_by_l[l] = max(1, (num_vector_channels or 8) // (2 ** (l - 1)))

    parts = []
    for key, mul in sorted(((int(k), int(v)) for k, v in channels_by_l.items())):
        if key <= lmax and mul > 0:
            parts.append(f"{mul}x{key}{parity_for_l(key)}")
    if not parts:
        raise ValueError("hidden irreps cannot be empty")
    return o3.Irreps("+".join(parts)).simplify()


def spherical_harmonics_irreps(lmax: int) -> o3.Irreps:
    return o3.Irreps.spherical_harmonics(lmax)


def wigner_D(irreps: o3.Irreps | str, R: torch.Tensor) -> torch.Tensor:
    irreps = o3.Irreps(irreps)
    if R.ndim == 2:
        return irreps.D_from_matrix(R)
    return torch.stack([irreps.D_from_matrix(r) for r in R], dim=0)


def transform_features(features: torch.Tensor, irreps: o3.Irreps | str, R: torch.Tensor) -> torch.Tensor:
    """Transform row-vector feature tensors by an e3nn representation."""
    D = wigner_D(irreps, R).to(device=features.device, dtype=features.dtype)
    if D.ndim == 2:
        return features @ D.T
    return torch.einsum("bnd,bed->bne", features, D)
