#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from se3force.data.rmd17 import load_rmd17_npz
from se3force.visualization.geometry import se3_transform_positions, se3_transform_vectors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export one molecular frame for force-field visualizations.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--molecule", required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--se3-checkpoint")
    parser.add_argument("--se3-config")
    parser.add_argument("--painn-checkpoint")
    parser.add_argument("--painn-config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic-if-no-checkpoint", action="store_true")
    return parser


def output_paths(output: str | Path) -> tuple[Path, Path]:
    path = Path(output)
    if path.suffix == ".json":
        return path, path.with_suffix(".npz")
    if path.suffix == ".npz":
        return path.with_suffix(".json"), path
    return path.with_suffix(".json"), path.with_suffix(".npz")


def rotation_from_seed(seed: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray([0.35 + 0.01 * seed, -0.72, 0.59], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(37.0 + (seed % 11))
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    rotation = np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )
    translation = np.asarray([2.25, -1.1, 0.65], dtype=np.float64)
    return rotation, translation


def placeholder_forces(target: np.ndarray, seed: int, label: str) -> np.ndarray:
    rng = np.random.default_rng(seed + (17 if label == "se3" else 41))
    norms = np.linalg.norm(target, axis=1)
    scale = float(np.percentile(norms, 90)) if norms.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    noise = rng.normal(size=target.shape).astype(np.float64)
    noise -= noise.mean(axis=0, keepdims=True)
    strength = 0.035 if label == "se3" else 0.03
    return target.astype(np.float64) + strength * scale * noise


def predict_with_checkpoint(config_path: str | None, checkpoint_path: str | None, positions: np.ndarray, atomic_numbers: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    if not config_path or not checkpoint_path:
        return None, None
    try:
        import torch

        from se3force.models.molecular import build_molecular_model
        from se3force.training.checkpointing import load_checkpoint
        from se3force.training.molecular_trainer import load_molecular_config

        config = load_molecular_config(config_path)
        model = build_molecular_model(config)
        load_checkpoint(checkpoint_path, model=model, map_location="cpu")
        model.eval()
        pos = torch.as_tensor(positions, dtype=torch.float32).unsqueeze(0)
        z = torch.as_tensor(atomic_numbers, dtype=torch.long).unsqueeze(0)
        mask = torch.ones(z.shape, dtype=torch.bool)
        with torch.no_grad():
            out = model(pos, z, mask)
        return out["forces"].squeeze(0).detach().cpu().numpy().astype(np.float64), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def relative_error(predicted_rotated: np.ndarray, expected_rotated: np.ndarray) -> float:
    denom = float(np.linalg.norm(expected_rotated))
    if denom <= 1e-12:
        denom = 1.0
    return float(np.linalg.norm(predicted_rotated - expected_rotated) / denom)


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def build_sample(args: argparse.Namespace) -> dict:
    arrays = load_rmd17_npz(args.data_path)
    positions_all = arrays["positions"]
    forces_all = arrays["forces"]
    atomic_numbers = arrays["atomic_numbers"].astype(np.int64)
    if args.frame_index < 0 or args.frame_index >= positions_all.shape[0]:
        raise IndexError(f"frame-index {args.frame_index} outside available range 0..{positions_all.shape[0] - 1}")
    positions = positions_all[args.frame_index].astype(np.float64)
    target_forces = forces_all[args.frame_index].astype(np.float64)
    rotation, translation = rotation_from_seed(args.seed)

    se3_forces, se3_error = predict_with_checkpoint(args.se3_config, args.se3_checkpoint, positions, atomic_numbers)
    painn_forces, painn_error = predict_with_checkpoint(args.painn_config, args.painn_checkpoint, positions, atomic_numbers)
    se3_placeholder = se3_forces is None
    painn_placeholder = painn_forces is None
    if se3_forces is None:
        se3_forces = placeholder_forces(target_forces, args.seed, "se3")
    if painn_forces is None:
        painn_forces = placeholder_forces(target_forces, args.seed, "painn")

    rotated_positions = se3_transform_positions(positions, rotation, translation)
    rotated_target = se3_transform_vectors(target_forces, rotation)
    expected_rotated_se3 = se3_transform_vectors(se3_forces, rotation)
    expected_rotated_painn = se3_transform_vectors(painn_forces, rotation)
    rotated_se3 = expected_rotated_se3.copy()
    rotated_painn = expected_rotated_painn.copy()
    equivariance: dict[str, float] = {}

    if not se3_placeholder:
        rotated_pred, rotated_error = predict_with_checkpoint(args.se3_config, args.se3_checkpoint, rotated_positions, atomic_numbers)
        if rotated_pred is not None:
            rotated_se3 = rotated_pred
            equivariance["se3"] = relative_error(rotated_se3, expected_rotated_se3)
        elif rotated_error:
            se3_error = rotated_error
    if not painn_placeholder:
        rotated_pred, rotated_error = predict_with_checkpoint(args.painn_config, args.painn_checkpoint, rotated_positions, atomic_numbers)
        if rotated_pred is not None:
            rotated_painn = rotated_pred
            equivariance["painn_lite"] = relative_error(rotated_painn, expected_rotated_painn)
        elif rotated_error:
            painn_error = rotated_error

    metadata = {
        "source": "local_rmd17_npz",
        "data_path_basename": Path(args.data_path).name,
        "source_keys": arrays["source_keys"],
        "se3_prediction_type": "placeholder_non_scientific" if se3_placeholder else "checkpoint",
        "painn_prediction_type": "placeholder_non_scientific" if painn_placeholder else "checkpoint",
        "se3_checkpoint": args.se3_checkpoint or "",
        "painn_checkpoint": args.painn_checkpoint or "",
        "se3_config": args.se3_config or "",
        "painn_config": args.painn_config or "",
        "se3_prediction_error": se3_error or "",
        "painn_prediction_error": painn_error or "",
        "placeholder_note": "Placeholder predictions are deterministic visual aids and must not be used for quantitative claims."
        if se3_placeholder or painn_placeholder
        else "",
    }
    eq_value = max(equivariance.values()) if equivariance else None
    return {
        "molecule": args.molecule,
        "frame_index": int(args.frame_index),
        "unit_length": "Angstrom",
        "unit_force": "kcal/mol/Angstrom",
        "positions": positions,
        "atomic_numbers": atomic_numbers,
        "target_forces": target_forces,
        "se3_forces": se3_forces,
        "painn_forces": painn_forces,
        "rotation_matrix": rotation,
        "translation": translation,
        "rotated_positions": rotated_positions,
        "rotated_target_forces": rotated_target,
        "rotated_se3_forces": rotated_se3,
        "rotated_painn_forces": rotated_painn,
        "expected_rotated_se3_forces": expected_rotated_se3,
        "expected_rotated_painn_forces": expected_rotated_painn,
        "equivariance_error": eq_value,
        "equivariance_errors": equivariance,
        "metadata": metadata,
    }


def write_outputs(sample: dict, json_path: Path, npz_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(to_jsonable(sample), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez(
        npz_path,
        positions=sample["positions"],
        atomic_numbers=sample["atomic_numbers"],
        target_forces=sample["target_forces"],
        se3_forces=sample["se3_forces"],
        painn_forces=sample["painn_forces"],
        rotation_matrix=sample["rotation_matrix"],
        translation=sample["translation"],
        rotated_positions=sample["rotated_positions"],
        rotated_target_forces=sample["rotated_target_forces"],
        rotated_se3_forces=sample["rotated_se3_forces"],
        rotated_painn_forces=sample["rotated_painn_forces"],
        expected_rotated_se3_forces=sample["expected_rotated_se3_forces"],
        expected_rotated_painn_forces=sample["expected_rotated_painn_forces"],
        metadata_json=json.dumps(to_jsonable(sample["metadata"]), sort_keys=True),
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sample = build_sample(args)
    json_path, npz_path = output_paths(args.output)
    write_outputs(sample, json_path, npz_path)
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
