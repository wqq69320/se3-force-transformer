from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch

from se3force.data.transforms import rotate_translate_molecular_batch
from se3force.evaluation.metrics_schema import standard_metrics
from se3force.geometry.metrics import parameter_count
from se3force.geometry.rotations import apply_rotation
from se3force.models.molecular import build_molecular_model
from se3force.training.checkpointing import load_checkpoint
from se3force.training.logging import write_json


def batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask]


def force_errors(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff = pred - target
    masked = masked_values(diff, mask)
    mse = (masked * masked).mean()
    mae = masked.abs().mean()
    rmse = mse.sqrt()
    return mse, mae, rmse


def finite_energy_mask(energy: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(energy.squeeze(-1))


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rmse_or_none(squared_errors: list[float]) -> float | None:
    return math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else None


def evaluate_molecular_model(model, loader, config: dict, metadata: dict, device="cpu") -> dict:
    model.eval()
    force_sse = 0.0
    force_abs = 0.0
    force_count = 0
    rotated_abs = 0.0
    rotated_count = 0
    equiv_numer = 0.0
    equiv_denom = 0.0
    energy_abs: list[float] = []
    energy_sq: list[float] = []
    energy_inv_errors: list[float] = []
    runtime = 0.0
    graph_stat_rows: list[dict[str, float]] = []
    atom_counts: list[int] = []
    training_mode = str(config.get("training", {}).get("mode", "direct_force"))

    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        if training_mode == "energy_force":
            batch["pos"] = batch["pos"].detach().clone().requires_grad_(True)
        start = time.perf_counter()
        out = model(batch["pos"], batch["z"], batch["mask"])
        runtime += time.perf_counter() - start
        pred_force = out["forces"]
        diff = pred_force - batch["forces"]
        mask3 = batch["mask"].unsqueeze(-1).expand_as(diff)
        force_sse += float((diff[mask3] ** 2).sum())
        force_abs += float(diff[mask3].abs().sum())
        force_count += int(mask3.sum())
        graph_stat_rows.append(out["graph_stats"])
        atom_counts.extend(int(v) for v in batch["num_atoms"].tolist())

        e_mask = finite_energy_mask(batch["energy"])
        if e_mask.any() and out.get("energy") is not None:
            e_diff = (out["energy"].detach()[e_mask] - batch["energy"][e_mask]).reshape(-1)
            energy_abs.extend(float(v) for v in e_diff.abs())
            energy_sq.extend(float(v * v) for v in e_diff)

        rotated_batch, R = rotate_translate_molecular_batch(batch)
        if training_mode == "energy_force":
            rotated_batch["pos"] = rotated_batch["pos"].detach().clone().requires_grad_(True)
        out_rot = model(rotated_batch["pos"], rotated_batch["z"], rotated_batch["mask"])
        expected_force = apply_rotation(pred_force.detach(), R)
        rot_diff = out_rot["forces"].detach() - rotated_batch["forces"]
        rotated_abs += float(rot_diff[mask3].abs().sum())
        rotated_count += int(mask3.sum())
        equiv_diff = (out_rot["forces"].detach() - expected_force)[mask3]
        equiv_ref = expected_force[mask3]
        equiv_numer += float((equiv_diff * equiv_diff).sum())
        equiv_denom += float((equiv_ref * equiv_ref).sum())
        if out.get("energy") is not None and out_rot.get("energy") is not None:
            denom = out["energy"].detach().norm().clamp_min(1e-8)
            energy_inv_errors.append(float((out_rot["energy"].detach() - out["energy"].detach()).norm() / denom))

    force_mse = force_sse / max(1, force_count)
    force_mae = force_abs / max(1, force_count)
    force_rmse = math.sqrt(force_mse)
    rotated_force_mae = rotated_abs / max(1, rotated_count)
    graph_mean = {
        key: sum(row[key] for row in graph_stat_rows) / max(1, len(graph_stat_rows))
        for key in ["average_neighbors", "edge_count_mean", "edge_count_max", "graph_build_time_sec"]
    }
    graph_mean["edge_count_max"] = max(row["edge_count_max"] for row in graph_stat_rows) if graph_stat_rows else 0.0
    equiv_error = math.sqrt(equiv_numer) / max(math.sqrt(equiv_denom), 1e-8)
    num_atoms_mean = sum(atom_counts) / max(1, len(atom_counts))
    num_atoms_max = max(atom_counts) if atom_counts else 0

    return {
        "canonical_mse": force_mse,
        "rotated_translated_mse": force_mse,
        "equivariance_error": equiv_error,
        "parameter_count": parameter_count(model),
        "runtime_per_batch_sec": runtime / max(1, len(loader)),
        "num_atoms_mean": num_atoms_mean,
        "num_atoms_max": num_atoms_max,
        "configuration_dim_mean": 3.0 * num_atoms_mean,
        "force_dim_mean": 3.0 * num_atoms_mean,
        "cutoff_radius": float(metadata.get("cutoff_radius", config.get("dataset", {}).get("cutoff_radius", float("nan")))),
        "edge_count_mean": graph_mean["edge_count_mean"],
        "edge_count_max": graph_mean["edge_count_max"],
        "average_neighbors": graph_mean["average_neighbors"],
        "graph_build_time_sec": graph_mean["graph_build_time_sec"],
        "force_mae": force_mae,
        "force_rmse": force_rmse,
        "rotated_force_mae": rotated_force_mae,
        "energy_mae": _mean_or_none(energy_abs),
        "energy_rmse": _rmse_or_none(energy_sq),
        "energy_invariance_error": _mean_or_none(energy_inv_errors),
        "force_unit": metadata.get("unit_force"),
        "energy_unit": metadata.get("unit_energy"),
        "training_mode": training_mode,
        "molecule_name": metadata.get("molecule_name", config.get("dataset", {}).get("molecule", "")),
        "num_frames_total": metadata.get("num_frames", 0),
        "split_type": metadata.get("split_type", "random"),
    }


def molecular_standard_metrics(
    config: dict,
    loaders: dict,
    metadata: dict,
    best_checkpoint: str | Path | None,
    final_train_loss: float | None,
    best_val_mse: float | None,
    eval_metrics: dict,
    extra: dict | None = None,
) -> dict:
    metrics = standard_metrics(config, loaders, best_checkpoint, final_train_loss, best_val_mse, eval_metrics, extra)
    metrics.update({key: eval_metrics.get(key) for key in eval_metrics if key not in metrics})
    metrics["dataset_name"] = metadata.get("dataset_name", metrics["dataset_name"])
    return metrics


def evaluate_molecular_checkpoint(config: dict, checkpoint_path: str | Path, loaders, metadata, output_path=None) -> dict:
    device = torch.device(config.get("device", "cpu"))
    model = build_molecular_model(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)
    eval_metrics = evaluate_molecular_model(model, loaders["test"], config, metadata, device)
    train_row = checkpoint.get("metrics", {})
    metrics = molecular_standard_metrics(
        config,
        loaders,
        metadata,
        checkpoint_path,
        train_row.get("train", {}).get("loss"),
        train_row.get("val", {}).get("canonical_mse"),
        eval_metrics,
    )
    if output_path is not None:
        write_json(output_path, metrics)
    return metrics
