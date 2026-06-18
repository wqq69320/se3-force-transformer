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
from se3force.models.molecular import build_molecular_model, molecular_model_identity
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


def _std_from_sums(total: float, sq_total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    mean = total / count
    variance = max(0.0, sq_total / count - mean * mean)
    return math.sqrt(variance)


def _percentile_or_none(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def force_loss_weight(config: dict) -> float:
    training = config.get("training", {})
    return float(training.get("force_loss_weight", training.get("lambda_force", 1.0)))


def force_scale_settings(config: dict, metadata: dict) -> dict[str, float | str]:
    training = config.get("training", {})
    normalization = str(training.get("force_scale_normalization", "none"))
    train_rms = float(metadata.get("force_rms", metadata.get("force_std", 1.0)) or 1.0)
    component_rms = float(metadata.get("force_component_rms", train_rms) or train_rms)
    vector_rms = float(metadata.get("force_vector_rms", metadata.get("force_norm_mean", train_rms)) or train_rms)
    train_std = float(metadata.get("force_std", 1.0) or 1.0)
    if normalization == "none":
        scale = 1.0
    elif normalization == "train_force_rms":
        scale = train_rms
    elif normalization == "train_force_component_rms":
        scale = component_rms
    elif normalization == "train_force_vector_rms":
        scale = vector_rms
    elif normalization == "train_force_std":
        scale = train_std
    elif normalization == "fixed":
        scale = float(training.get("fixed_force_scale_value", training.get("force_scale_value", 1.0)))
    else:
        raise ValueError(f"unknown force_scale_normalization: {normalization}")
    scale = max(float(scale), 1e-12)
    return {
        "force_scale_normalization": normalization,
        "force_scale_value": scale,
        "force_train_rms": train_rms,
        "force_train_std": train_std,
        "force_train_component_std": train_std,
        "force_train_component_rms": component_rms,
        "force_train_vector_rms": vector_rms,
        "fixed_force_scale_value": float(training.get("fixed_force_scale_value", training.get("force_scale_value", scale))),
    }


def energy_loss_weight(config: dict) -> float:
    training = config.get("training", {})
    return float(training.get("energy_loss_weight", training.get("lambda_energy", 0.0)))


def energy_settings(config: dict, metadata: dict) -> dict[str, float | bool]:
    training = config.get("training", {})
    mean = training.get("energy_train_mean", metadata.get("energy_train_mean"))
    std = training.get("energy_train_std", metadata.get("energy_train_std"))
    mean_f = float(mean) if mean is not None else 0.0
    std_f = float(std) if std not in (None, 0, 0.0) else 1.0
    std_f = max(std_f, 1e-12)
    centering = bool(training.get("energy_centering", False))
    standardization = bool(training.get("energy_standardization", False))
    loss_on_centered = bool(training.get("energy_loss_on_centered", centering or standardization))
    return {
        "energy_centering": centering,
        "energy_standardization": standardization,
        "energy_loss_on_centered": loss_on_centered,
        "energy_train_mean": mean_f,
        "energy_train_std": std_f,
        "energy_output_scale": std_f if standardization else 1.0,
        "energy_output_shift": mean_f if (centering or standardization or loss_on_centered) else 0.0,
    }


def energy_target_for_loss(raw_energy: torch.Tensor, config: dict, metadata: dict) -> torch.Tensor:
    settings = energy_settings(config, metadata)
    target = raw_energy
    if settings["energy_loss_on_centered"]:
        target = target - float(settings["energy_train_mean"])
        if settings["energy_standardization"]:
            target = target / float(settings["energy_train_std"])
    return target


def energy_prediction_for_loss(out: dict, config: dict, metadata: dict) -> torch.Tensor:
    settings = energy_settings(config, metadata)
    if settings["energy_loss_on_centered"]:
        return out["energy"]
    return out.get("energy_raw", out["energy"])


def _pct_improvement(baseline: float, model: float) -> float | None:
    if baseline in (None, 0) or not math.isfinite(baseline):
        return None
    return 100.0 * (baseline - model) / baseline


def evaluate_molecular_model(model, loader, config: dict, metadata: dict, device="cpu") -> dict:
    model.eval()
    force_sse = 0.0
    force_abs = 0.0
    force_count = 0
    force_vector_l2_sum = 0.0
    force_vector_l2_sq = 0.0
    force_vector_count = 0
    rotated_abs = 0.0
    rotated_sse = 0.0
    rotated_count = 0
    rotated_vector_l2_sum = 0.0
    rotated_vector_l2_sq = 0.0
    zero_force_sse = 0.0
    zero_force_abs = 0.0
    zero_force_vector_l2_sum = 0.0
    zero_force_vector_l2_sq = 0.0
    mean_force_sse = 0.0
    mean_force_abs = 0.0
    mean_force_vector_l2_sum = 0.0
    mean_force_vector_l2_sq = 0.0
    target_force_norm_sum = 0.0
    target_force_norm_sq = 0.0
    target_force_norm_values: list[float] = []
    pred_force_norm_sum = 0.0
    pred_force_norm_sq = 0.0
    residual_force_norm_sum = 0.0
    force_cosine_sum = 0.0
    force_cosine_sq = 0.0
    force_cosine_count = 0
    pred_component_sum = 0.0
    pred_component_sq = 0.0
    target_component_sum = 0.0
    target_component_sq = 0.0
    equiv_numer = 0.0
    equiv_denom = 0.0
    energy_abs_raw: list[float] = []
    energy_sq_raw: list[float] = []
    energy_abs_centered: list[float] = []
    energy_sq_centered: list[float] = []
    energy_inv_errors: list[float] = []
    runtime = 0.0
    graph_stat_rows: list[dict[str, float]] = []
    atom_counts: list[int] = []
    training_mode = str(config.get("training", {}).get("mode", "direct_force"))
    settings = energy_settings(config, metadata)
    force_scale = force_scale_settings(config, metadata)
    mean_force = torch.tensor(
        [
            float(metadata.get("force_mean_x", 0.0) or 0.0),
            float(metadata.get("force_mean_y", 0.0) or 0.0),
            float(metadata.get("force_mean_z", 0.0) or 0.0),
        ],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3)

    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        if training_mode == "energy_force":
            batch["pos"] = batch["pos"].detach().clone().requires_grad_(True)
        start = time.perf_counter()
        out = model(batch["pos"], batch["z"], batch["mask"])
        runtime += time.perf_counter() - start
        pred_force = out["forces"]
        diff = pred_force - batch["forces"]
        mask_atoms = batch["mask"]
        mask3 = batch["mask"].unsqueeze(-1).expand_as(diff)
        force_sse += float((diff[mask3] ** 2).sum())
        force_abs += float(diff[mask3].abs().sum())
        force_count += int(mask3.sum())
        vector_l2 = diff.norm(dim=-1)[mask_atoms]
        force_vector_l2_sum += float(vector_l2.sum())
        force_vector_l2_sq += float((vector_l2 * vector_l2).sum())
        force_vector_count += int(mask_atoms.sum())
        target_norm = batch["forces"].norm(dim=-1)[mask_atoms]
        pred_norm = pred_force.norm(dim=-1)[mask_atoms]
        target_force_norm_sum += float(target_norm.sum())
        target_force_norm_sq += float((target_norm * target_norm).sum())
        target_force_norm_values.extend(float(v) for v in target_norm.detach().cpu())
        pred_force_norm_sum += float(pred_norm.sum())
        pred_force_norm_sq += float((pred_norm * pred_norm).sum())
        residual_force_norm_sum += float(vector_l2.sum())
        dot = (pred_force * batch["forces"]).sum(dim=-1)[mask_atoms]
        denom = pred_norm * target_norm
        cosine_mask = denom > 1e-12
        if cosine_mask.any():
            cosine = dot[cosine_mask] / denom[cosine_mask].clamp_min(1e-12)
            force_cosine_sum += float(cosine.sum())
            force_cosine_sq += float((cosine * cosine).sum())
            force_cosine_count += int(cosine.numel())
        pred_components = pred_force[mask3]
        target_components = batch["forces"][mask3]
        pred_component_sum += float(pred_components.sum())
        pred_component_sq += float((pred_components * pred_components).sum())
        target_component_sum += float(target_components.sum())
        target_component_sq += float((target_components * target_components).sum())
        zero_diff = -batch["forces"]
        zero_l2 = zero_diff.norm(dim=-1)[mask_atoms]
        zero_force_sse += float((zero_diff[mask3] ** 2).sum())
        zero_force_abs += float(zero_diff[mask3].abs().sum())
        zero_force_vector_l2_sum += float(zero_l2.sum())
        zero_force_vector_l2_sq += float((zero_l2 * zero_l2).sum())
        mean_diff = mean_force.to(batch["forces"].dtype) - batch["forces"]
        mean_l2 = mean_diff.norm(dim=-1)[mask_atoms]
        mean_force_sse += float((mean_diff[mask3] ** 2).sum())
        mean_force_abs += float(mean_diff[mask3].abs().sum())
        mean_force_vector_l2_sum += float(mean_l2.sum())
        mean_force_vector_l2_sq += float((mean_l2 * mean_l2).sum())
        graph_stat_rows.append(out["graph_stats"])
        atom_counts.extend(int(v) for v in batch["num_atoms"].tolist())

        e_mask = finite_energy_mask(batch["energy"])
        if e_mask.any() and out.get("energy") is not None:
            pred_raw = out.get("energy_raw", out["energy"]).detach()
            target_raw = batch["energy"]
            raw_diff = (pred_raw[e_mask] - target_raw[e_mask]).reshape(-1)
            centered_diff = ((pred_raw - float(settings["energy_train_mean"]))[e_mask] - (target_raw - float(settings["energy_train_mean"]))[e_mask]).reshape(-1)
            energy_abs_raw.extend(float(v) for v in raw_diff.abs())
            energy_sq_raw.extend(float(v * v) for v in raw_diff)
            energy_abs_centered.extend(float(v) for v in centered_diff.abs())
            energy_sq_centered.extend(float(v * v) for v in centered_diff)

        rotated_batch, R = rotate_translate_molecular_batch(batch)
        if training_mode == "energy_force":
            rotated_batch["pos"] = rotated_batch["pos"].detach().clone().requires_grad_(True)
        out_rot = model(rotated_batch["pos"], rotated_batch["z"], rotated_batch["mask"])
        expected_force = apply_rotation(pred_force.detach(), R)
        rot_diff = out_rot["forces"].detach() - rotated_batch["forces"]
        rot_l2 = rot_diff.norm(dim=-1)[mask_atoms]
        rotated_sse += float((rot_diff[mask3] ** 2).sum())
        rotated_abs += float(rot_diff[mask3].abs().sum())
        rotated_count += int(mask3.sum())
        rotated_vector_l2_sum += float(rot_l2.sum())
        rotated_vector_l2_sq += float((rot_l2 * rot_l2).sum())
        equiv_diff = (out_rot["forces"].detach() - expected_force)[mask3]
        equiv_ref = expected_force[mask3]
        equiv_numer += float((equiv_diff * equiv_diff).sum())
        equiv_denom += float((equiv_ref * equiv_ref).sum())
        if out.get("energy") is not None and out_rot.get("energy") is not None:
            raw_energy = out.get("energy_raw", out["energy"]).detach()
            raw_energy_rot = out_rot.get("energy_raw", out_rot["energy"]).detach()
            denom = raw_energy.norm().clamp_min(1e-8)
            energy_inv_errors.append(float((raw_energy_rot - raw_energy).norm() / denom))

    force_mse = force_sse / max(1, force_count)
    force_mae = force_abs / max(1, force_count)
    force_rmse = math.sqrt(force_mse)
    force_vector_l2_mae = force_vector_l2_sum / max(1, force_vector_count)
    force_vector_l2_rmse = math.sqrt(force_vector_l2_sq / max(1, force_vector_count))
    rotated_force_mae = rotated_abs / max(1, rotated_count)
    rotated_force_rmse = math.sqrt(rotated_sse / max(1, rotated_count))
    rotated_force_vector_l2_mae = rotated_vector_l2_sum / max(1, force_vector_count)
    rotated_force_vector_l2_rmse = math.sqrt(rotated_vector_l2_sq / max(1, force_vector_count))
    zero_force_mse = zero_force_sse / max(1, force_count)
    zero_force_mae = zero_force_abs / max(1, force_count)
    zero_force_rmse = math.sqrt(zero_force_mse)
    zero_force_vector_l2_mae = zero_force_vector_l2_sum / max(1, force_vector_count)
    zero_force_vector_l2_rmse = math.sqrt(zero_force_vector_l2_sq / max(1, force_vector_count))
    mean_force_mse = mean_force_sse / max(1, force_count)
    mean_force_mae = mean_force_abs / max(1, force_count)
    mean_force_rmse = math.sqrt(mean_force_mse)
    mean_force_vector_l2_mae = mean_force_vector_l2_sum / max(1, force_vector_count)
    mean_force_vector_l2_rmse = math.sqrt(mean_force_vector_l2_sq / max(1, force_vector_count))
    target_force_norm_mean = target_force_norm_sum / max(1, force_vector_count)
    pred_force_norm_mean = pred_force_norm_sum / max(1, force_vector_count)
    target_force_norm_std = _std_from_sums(target_force_norm_sum, target_force_norm_sq, force_vector_count)
    pred_force_norm_std = _std_from_sums(pred_force_norm_sum, pred_force_norm_sq, force_vector_count)
    force_cosine_similarity_mean = force_cosine_sum / max(1, force_cosine_count)
    force_cosine_similarity_std = _std_from_sums(force_cosine_sum, force_cosine_sq, force_cosine_count)
    force_component_mean_pred = pred_component_sum / max(1, force_count)
    force_component_std_pred = _std_from_sums(pred_component_sum, pred_component_sq, force_count)
    force_component_mean_target = target_component_sum / max(1, force_count)
    force_component_std_target = _std_from_sums(target_component_sum, target_component_sq, force_count)
    graph_mean = {
        key: sum(row[key] for row in graph_stat_rows) / max(1, len(graph_stat_rows))
        for key in ["average_neighbors", "edge_count_mean", "edge_count_max", "graph_build_time_sec"]
    }
    graph_mean["edge_count_max"] = max(row["edge_count_max"] for row in graph_stat_rows) if graph_stat_rows else 0.0
    equiv_error = math.sqrt(equiv_numer) / max(math.sqrt(equiv_denom), 1e-8)
    num_atoms_mean = sum(atom_counts) / max(1, len(atom_counts))
    num_atoms_max = max(atom_counts) if atom_counts else 0
    identity = molecular_model_identity(model)

    return {
        "canonical_mse": force_mse,
        "rotated_translated_mse": force_mse,
        "equivariance_error": equiv_error,
        "force_equivariance_error": equiv_error,
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
        "rotated_force_rmse": rotated_force_rmse,
        "force_vector_l2_mae": force_vector_l2_mae,
        "force_vector_l2_rmse": force_vector_l2_rmse,
        "rotated_force_vector_l2_mae": rotated_force_vector_l2_mae,
        "rotated_force_vector_l2_rmse": rotated_force_vector_l2_rmse,
        "zero_force_mae": zero_force_mae,
        "zero_force_rmse": zero_force_rmse,
        "zero_force_vector_l2_mae": zero_force_vector_l2_mae,
        "zero_force_vector_l2_rmse": zero_force_vector_l2_rmse,
        "mean_force_mae": mean_force_mae,
        "mean_force_rmse": mean_force_rmse,
        "mean_force_vector_l2_mae": mean_force_vector_l2_mae,
        "mean_force_vector_l2_rmse": mean_force_vector_l2_rmse,
        "force_mae_improvement_vs_zero_pct": _pct_improvement(zero_force_mae, force_mae),
        "force_mae_improvement_vs_mean_pct": _pct_improvement(mean_force_mae, force_mae),
        "force_vector_l2_mae_improvement_vs_zero_pct": _pct_improvement(zero_force_vector_l2_mae, force_vector_l2_mae),
        "force_vector_l2_mae_improvement_vs_mean_pct": _pct_improvement(mean_force_vector_l2_mae, force_vector_l2_mae),
        "target_force_norm_mean": target_force_norm_mean,
        "target_force_norm_std": target_force_norm_std,
        "target_force_norm_median": _percentile_or_none(target_force_norm_values, 0.5),
        "target_force_norm_p95": _percentile_or_none(target_force_norm_values, 0.95),
        "target_force_norm_max": max(target_force_norm_values) if target_force_norm_values else None,
        "pred_force_norm_mean": pred_force_norm_mean,
        "pred_force_norm_std": pred_force_norm_std,
        "pred_to_target_force_norm_ratio": pred_force_norm_mean / max(target_force_norm_mean, 1e-12),
        "residual_force_norm_mean": residual_force_norm_sum / max(1, force_vector_count),
        "force_cosine_similarity_mean": force_cosine_similarity_mean,
        "force_cosine_similarity_std": force_cosine_similarity_std,
        "force_component_mean_pred": force_component_mean_pred,
        "force_component_std_pred": force_component_std_pred,
        "force_component_mean_target": force_component_mean_target,
        "force_component_std_target": force_component_std_target,
        "energy_mae": _mean_or_none(energy_abs_raw),
        "energy_rmse": _rmse_or_none(energy_sq_raw),
        "energy_mae_raw": _mean_or_none(energy_abs_raw),
        "energy_rmse_raw": _rmse_or_none(energy_sq_raw),
        "energy_mae_centered": _mean_or_none(energy_abs_centered),
        "energy_rmse_centered": _rmse_or_none(energy_sq_centered),
        "energy_invariance_error": _mean_or_none(energy_inv_errors),
        "force_unit": metadata.get("unit_force"),
        "energy_unit": metadata.get("unit_energy"),
        "training_mode": training_mode,
        "model_class": identity["model_class"],
        "backbone_class": identity["backbone_class"],
        "architecture_signature": identity["architecture_signature"],
        "lmax": identity["lmax"],
        "hidden_irreps": identity["hidden_irreps"],
        "actual_hidden_irreps": identity["actual_hidden_irreps"],
        "irreps_in": identity["irreps_in"],
        "irreps_hidden": identity["irreps_hidden"],
        "irreps_out": identity["irreps_out"],
        "irreps_sh": identity["irreps_sh"],
        "uses_non_scalar_hidden": identity["uses_non_scalar_hidden"],
        "uses_spherical_harmonics_in_value": identity["uses_spherical_harmonics_in_value"],
        "force_head_type": identity["force_head_type"],
        "force_head_irreps": identity["force_head_irreps"],
        "uses_relative_vector_fallback": identity["uses_relative_vector_fallback"],
        "uses_pairwise_force_skip": identity["uses_pairwise_force_skip"],
        "uses_atom_pair_embedding": identity["uses_atom_pair_embedding"],
        "atom_embedding_dim": identity["atom_embedding_dim"],
        "pair_embedding_dim": identity["pair_embedding_dim"],
        "edge_mlp_hidden_dim": identity["edge_mlp_hidden_dim"],
        "edge_mlp_layers": identity["edge_mlp_layers"],
        "uses_global_context": identity["uses_global_context"],
        "global_context_dim": identity["global_context_dim"],
        "global_context_type": identity["global_context_type"],
        "use_prototype_memory": identity["use_prototype_memory"],
        "prototype_count": identity["prototype_count"],
        "prototype_assignment": identity["prototype_assignment"],
        "graph_mode": identity["graph_mode"],
        "use_attention": identity["use_attention"],
        "use_gate": identity["use_gate"],
        "force_output_scale": identity["force_output_scale"],
        "learnable_force_output_scale": identity["learnable_force_output_scale"],
        "initial_force_output_scale": identity["initial_force_output_scale"],
        "parameter_count_by_module": identity["parameter_count_by_module"],
        "energy_train_mean": float(settings["energy_train_mean"]),
        "energy_train_std": float(settings["energy_train_std"]),
        "energy_centering": bool(settings["energy_centering"]),
        "energy_standardization": bool(settings["energy_standardization"]),
        "energy_loss_on_centered": bool(settings["energy_loss_on_centered"]),
        **force_scale,
        "force_loss_type": str(config.get("training", {}).get("force_loss_type", "mse")),
        "huber_delta": float(config.get("training", {}).get("huber_delta", 1.0)),
        "force_loss_weight": force_loss_weight(config),
        "energy_loss_weight": energy_loss_weight(config),
        "gradient_clip_norm": _gradient_clip_norm(config),
        "output_head_init_scale": float(config.get("model", {}).get("output_head_init_scale", 1.0)),
        "force_output_scale_regularization": float(config.get("model", {}).get("force_output_scale_regularization", 0.0)),
        "weight_decay": float(config.get("training", {}).get("weight_decay", 0.0)),
        "diagnostic_type": str(config.get("diagnostic_type", config.get("training", {}).get("diagnostic_type", ""))),
        "loss_mode": _loss_mode(config),
        "molecule_name": metadata.get("molecule_name", config.get("dataset", {}).get("molecule", "")),
        "data_source_type": metadata.get("data_source_type", ""),
        "is_fake_or_synthetic": bool(metadata.get("is_fake_or_synthetic", False)),
        "is_real_rmd17": bool(metadata.get("is_real_rmd17", False)),
        "dataset_path_basename": metadata.get("dataset_path_basename", ""),
        "num_frames_total": metadata.get("num_frames_total", metadata.get("num_frames", 0)),
        "num_frames_used": metadata.get("num_frames_used", metadata.get("num_frames", 0)),
        "split_type": metadata.get("split_type", "random"),
    }


def _loss_mode(config: dict) -> str:
    force = force_loss_weight(config)
    energy = energy_loss_weight(config)
    if force > 0 and energy > 0:
        return "combined"
    if force > 0:
        return "force"
    if energy > 0:
        return "energy"
    return "unspecified"


def _gradient_clip_norm(config: dict) -> float | None:
    training = config.get("training", {})
    value = training.get("gradient_clip_norm", training.get("gradient_clip"))
    return None if value is None else float(value)


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
    metrics.update(
        {
            "num_train_frames": len(loaders["train"].dataset),
            "num_val_frames": len(loaders["val"].dataset),
            "num_test_frames": len(loaders["test"].dataset),
            "num_train_batches": len(loaders["train"]),
            "num_val_batches": len(loaders["val"]),
            "num_test_batches": len(loaders["test"]),
            "batch_size": getattr(loaders["train"], "batch_size", None),
        }
    )
    for key in [
        "training_curve",
        "val_force_mae_epoch1",
        "val_force_mae_final",
        "val_force_rmse_final",
        "val_force_vector_l2_mae_epoch1",
        "val_force_vector_l2_mae_final",
        "val_force_mae_decreased",
        "learning_established",
    ]:
        metrics.setdefault(key, None)
    return metrics


def evaluate_molecular_checkpoint(config: dict, checkpoint_path: str | Path, loaders, metadata, output_path=None) -> dict:
    device = torch.device(config.get("device", "cpu"))
    settings = energy_settings(config, metadata)
    config.setdefault("model", {})["energy_output_scale"] = float(settings["energy_output_scale"])
    config.setdefault("model", {})["energy_output_shift"] = float(settings["energy_output_shift"])
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
