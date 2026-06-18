from __future__ import annotations

import numpy as np

ATOM_COLORS: dict[int, tuple[float, float, float]] = {
    1: (0.86, 0.86, 0.86),
    6: (0.08, 0.08, 0.08),
    7: (0.12, 0.32, 0.95),
    8: (0.9, 0.05, 0.04),
}

ATOM_RADII: dict[int, float] = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
}


def atom_color(atomic_number: int) -> tuple[float, float, float]:
    return ATOM_COLORS.get(int(atomic_number), (0.55, 0.55, 0.55))


def atom_radius(atomic_number: int) -> float:
    return float(ATOM_RADII.get(int(atomic_number), 0.7))


def _positions_array(positions) -> np.ndarray:
    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"positions must have shape [N, 3], got {arr.shape}")
    return arr


def _vectors_array(vectors, name: str = "vectors") -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {arr.shape}")
    return arr


def infer_bonds(
    positions,
    atomic_numbers,
    *,
    covalent_scale: float = 1.25,
    tolerance: float = 0.15,
    min_distance: float = 0.15,
) -> np.ndarray:
    pos = _positions_array(positions)
    z = np.asarray(atomic_numbers, dtype=np.int64).reshape(-1)
    if z.shape[0] != pos.shape[0]:
        raise ValueError(f"atomic_numbers length {z.shape[0]} does not match positions {pos.shape[0]}")
    bonds: list[tuple[int, int]] = []
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            distance = float(np.linalg.norm(pos[j] - pos[i]))
            cutoff = covalent_scale * (atom_radius(int(z[i])) + atom_radius(int(z[j]))) + tolerance
            if min_distance < distance <= cutoff:
                bonds.append((i, j))
    return np.asarray(bonds, dtype=np.int64).reshape(-1, 2)


def center_positions(positions) -> np.ndarray:
    pos = _positions_array(positions)
    return pos - pos.mean(axis=0, keepdims=True)


def scale_positions_for_rendering(positions, *, target_radius: float = 3.0) -> tuple[np.ndarray, float]:
    centered = center_positions(positions)
    max_radius = float(np.linalg.norm(centered, axis=1).max(initial=0.0))
    if not np.isfinite(max_radius) or max_radius <= 1e-12:
        return centered, 1.0
    scale = float(target_radius) / max_radius
    return centered * scale, scale


def force_scale_factor(
    forces,
    *,
    percentile: float = 90.0,
    max_arrow_length: float = 1.1,
    eps: float = 1e-12,
) -> float:
    vec = _vectors_array(forces, "forces")
    norms = np.linalg.norm(vec, axis=1)
    norms = norms[np.isfinite(norms)]
    if norms.size == 0:
        return 1.0
    reference = float(np.percentile(norms, percentile))
    if not np.isfinite(reference) or reference <= eps:
        return 1.0
    return float(max_arrow_length) / reference


def scale_forces(
    forces,
    *,
    percentile: float = 90.0,
    max_arrow_length: float = 1.1,
) -> tuple[np.ndarray, float]:
    factor = force_scale_factor(forces, percentile=percentile, max_arrow_length=max_arrow_length)
    scaled = _vectors_array(forces, "forces") * factor
    scaled[~np.isfinite(scaled)] = 0.0
    return scaled, factor


def _rotation_matrix(rotation) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape [3, 3], got {matrix.shape}")
    return matrix


def se3_transform_positions(positions, rotation, translation) -> np.ndarray:
    pos = _positions_array(positions)
    rot = _rotation_matrix(rotation)
    trans = np.asarray(translation, dtype=np.float64).reshape(3)
    return pos @ rot.T + trans


def se3_transform_vectors(vectors, rotation) -> np.ndarray:
    vec = _vectors_array(vectors, "vectors")
    rot = _rotation_matrix(rotation)
    return vec @ rot.T
