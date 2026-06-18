from __future__ import annotations

import copy
import csv
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from se3force.data.dataset_registry import build_molecular_dataloaders
from se3force.evaluation.molecular_evaluate import (
    batch_to_device,
    energy_loss_weight,
    energy_prediction_for_loss,
    energy_settings,
    energy_target_for_loss,
    evaluate_molecular_model,
    force_scale_settings,
    force_loss_weight,
    molecular_standard_metrics,
)
from se3force.models.molecular import build_molecular_model, molecular_model_identity
from se3force.training.checkpointing import load_checkpoint, save_checkpoint
from se3force.training.logging import write_json
from se3force.training.seed import set_seed


def load_molecular_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(path)
    config.setdefault("config_name", Path(path).name)
    return config


def _redacted_resolved_config(config: dict) -> dict:
    out = copy.deepcopy(config)
    dataset = out.get("dataset")
    if isinstance(dataset, dict) and dataset.get("path"):
        dataset["dataset_path_basename"] = Path(str(dataset["path"])).name
        dataset["path"] = dataset["dataset_path_basename"]
        dataset["path_redacted_from_absolute"] = True
    return out


def write_run_audit_artifacts(output_dir: Path, config: dict, model, metadata: dict) -> dict:
    identity = molecular_model_identity(model)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(_redacted_resolved_config(config), f, sort_keys=True)
    (output_dir / "model_repr.txt").write_text(repr(model) + "\n", encoding="utf-8")
    write_json(output_dir / "parameter_count_by_module.json", identity["parameter_count_by_module"])
    write_json(output_dir / "data_metadata.json", metadata)
    return identity


def configure_energy_normalization(config: dict, metadata: dict) -> None:
    training = config.setdefault("training", {})
    if "force_loss_weight" not in training:
        training["force_loss_weight"] = float(training.get("lambda_force", 1.0))
    if "energy_loss_weight" not in training:
        training["energy_loss_weight"] = float(training.get("lambda_energy", 0.0))
    if metadata.get("energy_train_mean") is not None:
        training["energy_train_mean"] = float(metadata["energy_train_mean"])
    if metadata.get("energy_train_std") is not None:
        training["energy_train_std"] = float(metadata["energy_train_std"])
    force_scale = force_scale_settings(config, metadata)
    training["force_scale_value"] = float(force_scale["force_scale_value"])
    training["force_train_rms"] = float(force_scale["force_train_rms"])
    training["force_train_std"] = float(force_scale["force_train_std"])
    training["force_train_component_std"] = float(force_scale["force_train_component_std"])
    training["force_train_component_rms"] = float(force_scale["force_train_component_rms"])
    training["force_train_vector_rms"] = float(force_scale["force_train_vector_rms"])
    if "fixed_force_scale_value" in force_scale:
        training["fixed_force_scale_value"] = float(force_scale["fixed_force_scale_value"])
    settings = energy_settings(config, metadata)
    model_cfg = config.setdefault("model", {})
    model_cfg["energy_output_scale"] = float(settings["energy_output_scale"])
    model_cfg["energy_output_shift"] = float(settings["energy_output_shift"])


def write_training_curve(path: Path, history: list[dict]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "train_force_loss",
        "train_force_mse",
        "train_force_mae",
        "train_force_vector_l2_mae",
        "train_force_vector_l2_rmse",
        "train_pred_force_norm_mean",
        "train_target_force_norm_mean",
        "train_pred_to_target_force_norm_ratio",
        "train_force_cosine_similarity_mean",
        "train_force_final_activation_norm",
        "train_last_hidden_norm",
        "train_message_norm_mean",
        "train_edge_message_norm_mean",
        "train_force_head_output_norm",
        "train_force_head_weight_norm",
        "train_force_head_bias_mean",
        "train_force_head_bias_norm",
        "train_force_output_scale",
        "train_force_head_grad_norm",
        "train_message_passing_grad_norm",
        "train_backbone_grad_norm",
        "train_edge_mlp_grad_norm",
        "train_total_grad_norm_before_clip",
        "train_total_grad_norm_after_clip",
        "train_max_grad_norm_before_clip",
        "train_max_grad_norm_after_clip",
        "train_learnable_force_output_scale_value",
        "train_learnable_force_output_scale_grad",
        "train_energy_output_norm",
        "train_energy_grad_norm",
        "train_energy_mse",
        "val_force_mae",
        "val_force_rmse",
        "val_force_vector_l2_mae",
        "val_force_vector_l2_rmse",
        "val_pred_to_target_force_norm_ratio",
        "val_force_cosine_similarity_mean",
        "val_energy_mae_raw",
        "val_energy_mae_centered",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            train = row["train"]
            val = row["val"]
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_loss": train.get("loss"),
                    "train_force_loss": train.get("force_loss"),
                    "train_force_mse": train.get("force_mse"),
                    "train_force_mae": train.get("force_mae"),
                    "train_force_vector_l2_mae": train.get("force_vector_l2_mae"),
                    "train_force_vector_l2_rmse": train.get("force_vector_l2_rmse"),
                    "train_pred_force_norm_mean": train.get("pred_force_norm_mean"),
                    "train_target_force_norm_mean": train.get("target_force_norm_mean"),
                    "train_pred_to_target_force_norm_ratio": train.get("pred_to_target_force_norm_ratio"),
                    "train_force_cosine_similarity_mean": train.get("force_cosine_similarity_mean"),
                    "train_force_final_activation_norm": train.get("force_final_activation_norm"),
                    "train_last_hidden_norm": train.get("last_hidden_norm"),
                    "train_message_norm_mean": train.get("message_norm_mean"),
                    "train_edge_message_norm_mean": train.get("edge_message_norm_mean"),
                    "train_force_head_output_norm": train.get("force_head_output_norm"),
                    "train_force_head_weight_norm": train.get("force_head_weight_norm"),
                    "train_force_head_bias_mean": train.get("force_head_bias_mean"),
                    "train_force_head_bias_norm": train.get("force_head_bias_norm"),
                    "train_force_output_scale": train.get("force_output_scale"),
                    "train_force_head_grad_norm": train.get("force_head_grad_norm"),
                    "train_message_passing_grad_norm": train.get("message_passing_grad_norm"),
                    "train_backbone_grad_norm": train.get("backbone_grad_norm"),
                    "train_edge_mlp_grad_norm": train.get("edge_mlp_grad_norm"),
                    "train_total_grad_norm_before_clip": train.get("total_grad_norm_before_clip"),
                    "train_total_grad_norm_after_clip": train.get("total_grad_norm_after_clip"),
                    "train_max_grad_norm_before_clip": train.get("max_grad_norm_before_clip"),
                    "train_max_grad_norm_after_clip": train.get("max_grad_norm_after_clip"),
                    "train_learnable_force_output_scale_value": train.get("learnable_force_output_scale_value"),
                    "train_learnable_force_output_scale_grad": train.get("learnable_force_output_scale_grad"),
                    "train_energy_output_norm": train.get("energy_output_norm"),
                    "train_energy_grad_norm": train.get("energy_grad_norm"),
                    "train_energy_mse": train.get("energy_mse"),
                    "val_force_mae": val.get("force_mae"),
                    "val_force_rmse": val.get("force_rmse"),
                    "val_force_vector_l2_mae": val.get("force_vector_l2_mae"),
                    "val_force_vector_l2_rmse": val.get("force_vector_l2_rmse"),
                    "val_pred_to_target_force_norm_ratio": val.get("pred_to_target_force_norm_ratio"),
                    "val_force_cosine_similarity_mean": val.get("force_cosine_similarity_mean"),
                    "val_energy_mae_raw": val.get("energy_mae_raw"),
                    "val_energy_mae_centered": val.get("energy_mae_centered"),
                }
            )


def learning_diagnostics(history: list[dict]) -> dict:
    first = history[0]["val"]
    final = history[-1]["val"]
    final_force = float(final.get("force_mae", float("inf")))
    baseline = min(float(final.get("zero_force_mae", float("inf"))), float(final.get("mean_force_mae", float("inf"))))
    decreased = final_force < float(first.get("force_mae", float("inf")))
    return {
        "val_force_mae_epoch1": first.get("force_mae"),
        "val_force_mae_final": final.get("force_mae"),
        "val_force_rmse_final": final.get("force_rmse"),
        "val_force_vector_l2_mae_epoch1": first.get("force_vector_l2_mae"),
        "val_force_vector_l2_mae_final": final.get("force_vector_l2_mae"),
        "val_pred_to_target_force_norm_ratio_final": final.get("pred_to_target_force_norm_ratio"),
        "val_force_cosine_similarity_mean_final": final.get("force_cosine_similarity_mean"),
        "val_force_mae_decreased": decreased,
        "learning_established": bool(decreased and final_force < baseline),
    }


def training_curve_diagnostic_summary(history: list[dict]) -> dict:
    def final(key: str):
        return history[-1]["train"].get(key) if history else None

    def max_value(key: str):
        values = [row["train"].get(key) for row in history if row["train"].get(key) is not None]
        return max(float(value) for value in values) if values else None

    return {
        "train_force_head_weight_norm_final": final("force_head_weight_norm"),
        "train_force_head_bias_norm_final": final("force_head_bias_norm"),
        "train_force_output_scale_final": final("force_output_scale"),
        "train_force_final_activation_norm_final": final("force_final_activation_norm"),
        "train_last_hidden_norm_final": final("last_hidden_norm"),
        "train_message_norm_mean_final": final("message_norm_mean"),
        "train_edge_message_norm_mean_final": final("edge_message_norm_mean"),
        "train_force_head_output_norm_final": final("force_head_output_norm"),
        "train_force_head_grad_norm_max": max_value("force_head_grad_norm"),
        "train_message_passing_grad_norm_max": max_value("message_passing_grad_norm"),
        "train_backbone_grad_norm_max": max_value("backbone_grad_norm"),
        "train_edge_mlp_grad_norm_max": max_value("edge_mlp_grad_norm"),
        "train_total_grad_norm_before_clip_max": max_value("total_grad_norm_before_clip"),
        "train_total_grad_norm_after_clip_max": max_value("total_grad_norm_after_clip"),
        "train_max_grad_norm_before_clip_max": max_value("max_grad_norm_before_clip"),
        "train_max_grad_norm_after_clip_max": max_value("max_grad_norm_after_clip"),
        "train_learnable_force_output_scale_value_final": final("learnable_force_output_scale_value"),
        "train_learnable_force_output_scale_grad_max": max_value("learnable_force_output_scale_grad"),
        "train_energy_output_norm_final": final("energy_output_norm"),
        "train_energy_grad_norm_max": max_value("energy_grad_norm"),
    }


def _tensor_norm(value: torch.Tensor | None) -> float:
    if value is None:
        return 0.0
    return float(value.detach().norm())


def _grad_norm(named_parameters, predicate=None) -> float:
    total = 0.0
    for name, param in named_parameters:
        if predicate is not None and not predicate(name):
            continue
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float((grad * grad).sum())
    return total**0.5


def _max_grad_norm(named_parameters, predicate=None) -> float:
    max_value = 0.0
    for name, param in named_parameters:
        if predicate is not None and not predicate(name):
            continue
        if param.grad is None:
            continue
        max_value = max(max_value, float(param.grad.detach().norm()))
    return max_value


def _is_force_head_parameter(name: str) -> bool:
    return (
        "force_mlp" in name
        or "pair_skip_mlp" in name
        or "pair_residual_mlp" in name
        or "force_output_log_scale" in name
        or "backbone.force_head" in name
        or "coeff_matrix_head" in name
        or "edge_mlp" in name
        or "energy_mlp" in name
        or name.startswith("force_head")
        or name.startswith("net.")
    )


def _is_message_parameter(name: str) -> bool:
    return (
        name.startswith("atom_embedding")
        or name.startswith("pair_embedding")
        or name.startswith("layers")
        or name.startswith("radial")
        or name.startswith("attention_mlp")
        or name.startswith("force_gate")
        or name.startswith("backbone.blocks")
        or name.startswith("backbone.scalar_embed")
        or name.startswith("global_mlp")
        or name.startswith("edge_mlps")
        or name.startswith("scalar_updates")
    )


def _is_edge_mlp_parameter(name: str) -> bool:
    return (
        "force_mlp" in name
        or "pair_residual_mlp" in name
        or "pair_embedding" in name
        or "attention_mlp" in name
        or "key_radial" in name
        or "value_radial" in name
        or "bias_radial" in name
        or "edge_mlps" in name
        or "global_mlp" in name
        or "coeff_matrix_head" in name
    )


def _force_head_stats(model) -> dict[str, float | None]:
    layer = None
    if hasattr(model, "force_mlp") and len(model.force_mlp) > 0:
        candidate = model.force_mlp[-1]
        if hasattr(candidate, "weight"):
            layer = candidate
    elif hasattr(model, "edge_mlp") and len(model.edge_mlp) > 0:
        candidate = model.edge_mlp[-1]
        if hasattr(candidate, "weight"):
            layer = candidate
    elif hasattr(model, "energy_mlp") and len(model.energy_mlp) > 0:
        candidate = model.energy_mlp[-1]
        if hasattr(candidate, "weight"):
            layer = candidate
    elif hasattr(model, "force_head"):
        candidate = model.force_head
        if hasattr(candidate, "weight"):
            layer = candidate
    elif hasattr(model, "backbone") and hasattr(model.backbone, "force_head"):
        candidate = model.backbone.force_head
        if hasattr(candidate, "weight"):
            layer = candidate
    elif hasattr(model, "net") and len(model.net) > 0:
        candidate = model.net[-1]
        if hasattr(candidate, "weight"):
            layer = candidate
    scale = None
    if hasattr(model, "force_output_scale_value"):
        try:
            scale = float(model.force_output_scale_value().detach())
        except Exception:  # noqa: BLE001
            scale = None
    if layer is None:
        return {
            "force_head_weight_norm": None,
            "force_head_bias_mean": None,
            "force_head_bias_norm": None,
            "force_output_scale": scale,
        }
    bias = getattr(layer, "bias", None)
    return {
        "force_head_weight_norm": _tensor_norm(layer.weight),
        "force_head_bias_mean": None if bias is None else float(bias.detach().mean()),
        "force_head_bias_norm": None if bias is None else _tensor_norm(bias),
        "force_output_scale": scale,
    }


def _force_output_scale_regularization(model, config: dict, device: torch.device) -> torch.Tensor:
    weight = float(config.get("model", {}).get("force_output_scale_regularization", 0.0))
    log_scale = getattr(model, "force_output_log_scale", None)
    if weight <= 0.0 or log_scale is None:
        return torch.zeros((), device=device)
    return weight * (log_scale * log_scale)


def _diagnostic_float(diagnostics: dict, key: str) -> float | None:
    value = diagnostics.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach())
    return float(value)


def molecular_losses(out: dict, batch: dict, config: dict, metadata: dict, model=None) -> tuple[torch.Tensor, dict]:
    mask3 = batch["mask"].unsqueeze(-1).expand_as(batch["forces"])
    diff = out["forces"] - batch["forces"]
    force_mse = (diff[mask3] ** 2).mean()
    force_mae = diff[mask3].abs().mean()
    mask_atoms = batch["mask"]
    vector_l2 = diff.norm(dim=-1)[mask_atoms]
    force_vector_l2_mae = vector_l2.mean()
    force_vector_l2_rmse = (vector_l2 * vector_l2).mean().sqrt()
    pred_norm = out["forces"].norm(dim=-1)[mask_atoms]
    target_norm = batch["forces"].norm(dim=-1)[mask_atoms]
    pred_force_norm_mean = pred_norm.mean()
    target_force_norm_mean = target_norm.mean()
    norm_ratio = pred_norm.mean() / target_norm.mean().clamp_min(1e-12)
    denom = pred_norm * target_norm
    cosine_mask = denom > 1e-12
    if cosine_mask.any():
        cosine = ((out["forces"] * batch["forces"]).sum(dim=-1)[mask_atoms][cosine_mask] / denom[cosine_mask].clamp_min(1e-12)).mean()
    else:
        cosine = diff.new_zeros(())
    scale = float(force_scale_settings(config, metadata)["force_scale_value"])
    scaled = diff[mask3] / scale
    loss_type = str(config.get("training", {}).get("force_loss_type", "mse"))
    if loss_type == "mse":
        force_loss = (scaled * scaled).mean()
    elif loss_type == "mae":
        force_loss = scaled.abs().mean()
    elif loss_type == "huber":
        delta = float(config.get("training", {}).get("huber_delta", 1.0))
        abs_scaled = scaled.abs()
        quadratic = torch.minimum(abs_scaled, scaled.new_full((), delta))
        linear = abs_scaled - quadratic
        force_loss = (0.5 * quadratic * quadratic + delta * linear).mean()
    elif loss_type == "vector_l2":
        force_loss = (vector_l2 / scale).mean()
    elif loss_type == "normalized_vector_l2":
        force_loss = (vector_l2 / target_norm.clamp_min(1e-12)).mean()
    else:
        raise ValueError(f"unknown force_loss_type: {loss_type}")
    loss = force_loss_weight(config) * force_loss
    energy_mse = None
    e_mask = torch.isfinite(batch["energy"].squeeze(-1))
    if e_mask.any() and out.get("energy") is not None:
        pred_energy = energy_prediction_for_loss(out, config, metadata)
        target_energy = energy_target_for_loss(batch["energy"], config, metadata)
        e_diff = pred_energy[e_mask] - target_energy[e_mask]
        energy_mse = (e_diff * e_diff).mean()
        loss = loss + energy_loss_weight(config) * energy_mse
    regularization = _force_output_scale_regularization(model, config, batch["forces"].device) if model is not None else batch["forces"].new_zeros(())
    loss = loss + regularization
    return loss, {
        "loss": float(loss.detach()),
        "force_loss": float(force_loss.detach()),
        "force_mse": float(force_mse.detach()),
        "force_mae": float(force_mae.detach()),
        "force_vector_l2_mae": float(force_vector_l2_mae.detach()),
        "force_vector_l2_rmse": float(force_vector_l2_rmse.detach()),
        "pred_force_norm_mean": float(pred_force_norm_mean.detach()),
        "target_force_norm_mean": float(target_force_norm_mean.detach()),
        "pred_to_target_force_norm_ratio": float(norm_ratio.detach()),
        "force_cosine_similarity_mean": float(cosine.detach()),
        "force_output_scale_regularization_loss": float(regularization.detach()),
        "energy_mse": None if energy_mse is None else float(energy_mse.detach()),
    }


def train_one_epoch_molecular(model, loader, optimizer, config, metadata, device) -> dict:
    model.train()
    total = {
        "loss": 0.0,
        "force_loss": 0.0,
        "force_mse": 0.0,
        "force_mae": 0.0,
        "force_vector_l2_mae": 0.0,
        "force_vector_l2_rmse": 0.0,
        "pred_force_norm_mean": 0.0,
        "target_force_norm_mean": 0.0,
        "pred_to_target_force_norm_ratio": 0.0,
        "force_cosine_similarity_mean": 0.0,
        "force_output_scale_regularization_loss": 0.0,
        "force_final_activation_norm": 0.0,
        "last_hidden_norm": 0.0,
        "message_norm_mean": 0.0,
        "edge_message_norm_mean": 0.0,
        "force_head_output_norm": 0.0,
        "energy_output_norm": 0.0,
        "energy_grad_norm": 0.0,
        "force_head_weight_norm": 0.0,
        "force_head_bias_mean": 0.0,
        "force_head_bias_norm": 0.0,
        "force_output_scale": 0.0,
        "force_head_grad_norm": 0.0,
        "message_passing_grad_norm": 0.0,
        "backbone_grad_norm": 0.0,
        "edge_mlp_grad_norm": 0.0,
        "total_grad_norm_before_clip": 0.0,
        "total_grad_norm_after_clip": 0.0,
        "max_grad_norm_before_clip": 0.0,
        "max_grad_norm_after_clip": 0.0,
    }
    count = 0
    max_steps = config.get("training", {}).get("max_steps_per_epoch")
    max_steps = int(max_steps) if max_steps is not None else None
    iterator = tqdm(loader, desc="molecular-train", leave=False)
    for step, raw_batch in enumerate(iterator, start=1):
        batch = batch_to_device(raw_batch, device)
        if config.get("training", {}).get("mode", "direct_force") == "energy_force":
            batch["pos"] = batch["pos"].detach().clone().requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["pos"], batch["z"], batch["mask"])
        loss, row = molecular_losses(out, batch, config, metadata, model=model)
        loss.backward()
        training = config.get("training", {})
        grad_clip = training.get("gradient_clip_norm", training.get("gradient_clip"))
        named_params = list(model.named_parameters())
        total_grad_before = _grad_norm(named_params)
        max_grad_before = _max_grad_norm(named_params)
        force_head_grad = _grad_norm(named_params, _is_force_head_parameter)
        message_grad = _grad_norm(named_params, _is_message_parameter)
        backbone_grad = _grad_norm(named_params, lambda name: name.startswith("backbone"))
        edge_mlp_grad = _grad_norm(named_params, _is_edge_mlp_parameter)
        if grad_clip is not None:
            clip_grad_norm_(model.parameters(), float(grad_clip))
        total_grad_after = _grad_norm(named_params)
        max_grad_after = _max_grad_norm(named_params)
        optimizer.step()
        diagnostics = out.get("diagnostics", {})
        log_scale = getattr(model, "force_output_log_scale", None)
        learnable_scale_value = float(log_scale.detach().exp()) if log_scale is not None else None
        learnable_scale_grad = None
        if log_scale is not None and log_scale.grad is not None:
            learnable_scale_grad = float(log_scale.grad.detach().abs())
        row.update(
            {
                "force_final_activation_norm": _diagnostic_float(diagnostics, "force_final_activation_norm"),
                "last_hidden_norm": _diagnostic_float(diagnostics, "last_hidden_norm"),
                "message_norm_mean": _diagnostic_float(diagnostics, "message_norm_mean"),
                "edge_message_norm_mean": _diagnostic_float(diagnostics, "edge_message_norm_mean"),
                "force_head_output_norm": _diagnostic_float(diagnostics, "force_head_output_norm"),
                "energy_output_norm": _diagnostic_float(diagnostics, "energy_output_norm"),
                "energy_grad_norm": _diagnostic_float(diagnostics, "energy_grad_norm"),
                "force_head_grad_norm": force_head_grad,
                "message_passing_grad_norm": message_grad,
                "backbone_grad_norm": backbone_grad,
                "edge_mlp_grad_norm": edge_mlp_grad,
                "total_grad_norm_before_clip": total_grad_before,
                "total_grad_norm_after_clip": total_grad_after,
                "max_grad_norm_before_clip": max_grad_before,
                "max_grad_norm_after_clip": max_grad_after,
                "learnable_force_output_scale_value": learnable_scale_value,
                "learnable_force_output_scale_grad": learnable_scale_grad,
                **_force_head_stats(model),
            }
        )
        for key in total:
            value = row.get(key)
            total[key] += 0.0 if value is None else float(value)
        count += 1
        iterator.set_postfix(loss=total["loss"] / count, force_mae=total["force_mae"] / count)
        if max_steps is not None and step >= max_steps:
            break
    averaged = {key: value / max(1, count) for key, value in total.items()}
    log_scale = getattr(model, "force_output_log_scale", None)
    if log_scale is not None:
        averaged["learnable_force_output_scale_value"] = float(log_scale.detach().exp())
        averaged["learnable_force_output_scale_grad"] = (
            None if log_scale.grad is None else float(log_scale.grad.detach().abs())
        )
    else:
        averaged["learnable_force_output_scale_value"] = None
        averaged["learnable_force_output_scale_grad"] = None
    return averaged


def initial_oracle_linear_scalar_baseline(model, loader, config: dict, device) -> dict:
    if str(config.get("training", {}).get("mode", "direct_force")) != "direct_force":
        return {}
    model.eval()
    preds = []
    targets = []
    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        out = model(batch["pos"], batch["z"], batch["mask"])
        mask3 = batch["mask"].unsqueeze(-1).expand_as(batch["forces"])
        preds.append(out["forces"].detach()[mask3].reshape(-1, 3))
        targets.append(batch["forces"][mask3].reshape(-1, 3))
    if not preds:
        return {}
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    denom = (pred * pred).sum().clamp_min(1e-12)
    scalar = (pred * target).sum() / denom
    corrected = scalar * pred
    diff = corrected - target
    vector_l2 = diff.norm(dim=-1)
    pred_norm = corrected.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    cosine_denom = pred_norm * target_norm
    cosine_mask = cosine_denom > 1e-12
    cosine = (
        (corrected * target).sum(dim=-1)[cosine_mask] / cosine_denom[cosine_mask].clamp_min(1e-12)
        if cosine_mask.any()
        else target.new_zeros(1)
    )
    return {
        "initial_oracle_scalar_c": float(scalar),
        "initial_oracle_force_vector_l2_mae": float(vector_l2.mean()),
        "initial_oracle_pred_to_target_force_norm_ratio": float(pred_norm.mean() / target_norm.mean().clamp_min(1e-12)),
        "initial_oracle_force_cosine_similarity_mean": float(cosine.mean()),
    }


def write_force_norm_distribution(path: Path, dataset) -> None:
    fields = [
        "index",
        "frame_id",
        "target_force_norm_mean",
        "target_force_norm_median",
        "target_force_norm_p95",
        "target_force_norm_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for index in range(len(dataset)):
            item = dataset[index]
            norms = item["forces"].norm(dim=-1).to(torch.float32)
            quantiles = torch.quantile(norms, torch.tensor([0.5, 0.95], dtype=norms.dtype, device=norms.device))
            writer.writerow(
                {
                    "index": index,
                    "frame_id": int(item.get("frame_id", index)),
                    "target_force_norm_mean": float(norms.mean()),
                    "target_force_norm_median": float(quantiles[0]),
                    "target_force_norm_p95": float(quantiles[1]),
                    "target_force_norm_max": float(norms.max()),
                }
            )


def train_molecular_from_config(config: dict) -> dict:
    set_seed(int(config.get("seed", 0)))
    device = torch.device(config.get("device", "cpu"))
    output_dir = Path(config.get("output_dir", "outputs/molecular_run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders, metadata = build_molecular_dataloaders(config)
    configure_energy_normalization(config, metadata)
    model = build_molecular_model(config).to(device)
    write_run_audit_artifacts(output_dir, config, model, metadata)
    initial_oracle = initial_oracle_linear_scalar_baseline(model, loaders["train"], config, device)
    force_norm_distribution_path = output_dir / "train_force_norm_distribution.csv"
    if bool(config.get("training", {}).get("diagnostic_logging", False)) or str(config.get("diagnostic_type", "")):
        write_force_norm_distribution(force_norm_distribution_path, loaders["train"].dataset)
    train_cfg = config.get("training", {})
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 1))
    history = []
    best_val = float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch_molecular(model, loaders["train"], optimizer, config, metadata, device)
        val_metrics = evaluate_molecular_model(model, loaders["val"], config, metadata, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        save_checkpoint(last_path, model, optimizer, epoch, row, config)
        if val_metrics["force_mae"] <= best_val:
            best_val = val_metrics["force_mae"]
            save_checkpoint(best_path, model, optimizer, epoch, row, config)
        print(f"epoch={epoch} loss={train_metrics['loss']:.6g} val_force_mae={val_metrics['force_mae']:.6g}")

    write_training_curve(output_dir / "training_curve.csv", history)
    load_checkpoint(best_path, model=model, map_location=device)
    train_eval_metrics = evaluate_molecular_model(model, loaders["train"], config, metadata, device)
    eval_metrics = evaluate_molecular_model(model, loaders["test"], config, metadata, device)
    diagnostics = learning_diagnostics(history)
    curve_diagnostics = training_curve_diagnostic_summary(history)
    train_eval_keys = [
        "force_mae",
        "force_rmse",
        "force_vector_l2_mae",
        "force_vector_l2_rmse",
        "zero_force_mae",
        "zero_force_vector_l2_mae",
        "mean_force_mae",
        "mean_force_vector_l2_mae",
        "force_mae_improvement_vs_zero_pct",
        "force_mae_improvement_vs_mean_pct",
        "force_vector_l2_mae_improvement_vs_zero_pct",
        "force_vector_l2_mae_improvement_vs_mean_pct",
        "pred_to_target_force_norm_ratio",
        "residual_force_norm_mean",
        "force_cosine_similarity_mean",
        "force_cosine_similarity_std",
        "target_force_norm_mean",
        "target_force_norm_median",
        "target_force_norm_p95",
        "target_force_norm_max",
        "pred_force_norm_mean",
    ]
    train_eval = {f"train_eval_{key}": train_eval_metrics.get(key) for key in train_eval_keys}
    metrics = molecular_standard_metrics(
        config,
        loaders,
        metadata,
        best_path,
        history[-1]["train"]["loss"],
        history[-1]["val"]["force_rmse"] ** 2,
        eval_metrics,
        extra={
            "history": history,
            "last_checkpoint": str(last_path),
            "metadata": metadata,
            "training_curve": str(output_dir / "training_curve.csv"),
            "train_force_norm_distribution": str(force_norm_distribution_path) if force_norm_distribution_path.exists() else None,
            "epochs": int(train_cfg.get("epochs", epochs)),
            "max_steps_per_epoch": train_cfg.get("max_steps_per_epoch"),
            **diagnostics,
            **curve_diagnostics,
            **initial_oracle,
            **train_eval,
        },
    )
    write_json(output_dir / "metrics.json", metrics)
    return metrics
