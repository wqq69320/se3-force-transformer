from __future__ import annotations

from pathlib import Path
from typing import Any

from .display import display_name_for_config

REQUIRED_METRIC_FIELDS = [
    "run_name",
    "model_name",
    "config_name",
    "dataset_name",
    "seed",
    "device",
    "num_train_samples",
    "num_val_samples",
    "num_test_samples",
    "best_checkpoint",
    "final_train_loss",
    "best_val_mse",
    "canonical_mse",
    "rotated_translated_mse",
    "equivariance_error",
    "parameter_count",
    "runtime_per_batch_sec",
]


def config_name(config: dict[str, Any]) -> str:
    if config.get("config_name"):
        return str(config["config_name"])
    if config.get("_config_path"):
        return Path(config["_config_path"]).name
    return "inline_config"


def run_name(config: dict[str, Any]) -> str:
    if config.get("run_name"):
        return str(config["run_name"])
    return Path(str(config.get("output_dir", "run"))).name


def sample_counts(loaders: dict[str, Any]) -> dict[str, int]:
    return {
        "num_train_samples": len(loaders["train"].dataset),
        "num_val_samples": len(loaders["val"].dataset),
        "num_test_samples": len(loaders["test"].dataset),
    }


def metadata_from_config(config: dict[str, Any], loaders: dict[str, Any], best_checkpoint: str | Path | None) -> dict[str, Any]:
    cfg_name = config_name(config)
    return {
        "run_name": run_name(config),
        "model_name": str(config.get("model", {}).get("name", "se3_transformer")),
        "config_name": cfg_name,
        "display_name": display_name_for_config(cfg_name),
        "dataset_name": str(config.get("dataset", {}).get("name", "angular")),
        "seed": int(config.get("seed", 0)),
        "device": str(config.get("device", "cpu")),
        **sample_counts(loaders),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else "",
    }


def standard_metrics(
    config: dict[str, Any],
    loaders: dict[str, Any],
    best_checkpoint: str | Path | None,
    final_train_loss: float | None,
    best_val_mse: float | None,
    eval_metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = metadata_from_config(config, loaders, best_checkpoint)
    metrics.update(
        {
            "final_train_loss": final_train_loss,
            "best_val_mse": best_val_mse,
            "canonical_mse": eval_metrics.get("canonical_mse"),
            "rotated_translated_mse": eval_metrics.get("rotated_translated_mse"),
            "equivariance_error": eval_metrics.get("equivariance_error"),
            "parameter_count": eval_metrics.get("parameter_count"),
            "runtime_per_batch_sec": eval_metrics.get("runtime_per_batch_sec"),
        }
    )
    if extra:
        metrics.update(extra)
    return metrics
