#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import traceback
from pathlib import Path
from statistics import mean, stdev

from se3force.evaluation.metrics_schema import REQUIRED_METRIC_FIELDS
from se3force.evaluation.molecular_metrics import MOLECULAR_REQUIRED_FIELDS
from se3force.training.molecular_overrides import add_molecular_override_args, apply_molecular_overrides
from se3force.training.molecular_trainer import load_molecular_config, train_molecular_from_config

GROUP_FIELDS = [
    "config_name",
    "model_name",
    "model_class",
    "backbone_class",
    "dataset_name",
    "molecule_name",
    "data_source_type",
    "is_fake_or_synthetic",
    "is_real_rmd17",
    "training_mode",
    "split_type",
    "force_scale_normalization",
    "force_loss_type",
    "diagnostic_type",
    "gradient_clip_norm",
    "output_head_init_scale",
    "force_output_scale",
    "learnable_force_output_scale",
    "initial_force_output_scale",
    "force_output_scale_regularization",
    "fixed_force_scale_value",
    "weight_decay",
    "learning_rate",
]

NUMERIC_FIELDS = [
    "seed",
    "num_train_frames",
    "num_val_frames",
    "num_test_frames",
    "num_train_batches",
    "num_val_batches",
    "num_test_batches",
    "batch_size",
    "num_frames_used",
    "num_frames_total",
    "force_mae",
    "force_rmse",
    "force_vector_l2_mae",
    "force_vector_l2_rmse",
    "rotated_force_mae",
    "rotated_force_vector_l2_mae",
    "zero_force_mae",
    "zero_force_rmse",
    "zero_force_vector_l2_mae",
    "zero_force_vector_l2_rmse",
    "mean_force_mae",
    "mean_force_rmse",
    "mean_force_vector_l2_mae",
    "mean_force_vector_l2_rmse",
    "force_mae_improvement_vs_zero_pct",
    "force_mae_improvement_vs_mean_pct",
    "force_vector_l2_mae_improvement_vs_zero_pct",
    "force_vector_l2_mae_improvement_vs_mean_pct",
    "target_force_norm_mean",
    "target_force_norm_std",
    "target_force_norm_median",
    "target_force_norm_p95",
    "target_force_norm_max",
    "pred_force_norm_mean",
    "pred_force_norm_std",
    "pred_to_target_force_norm_ratio",
    "residual_force_norm_mean",
    "force_cosine_similarity_mean",
    "force_cosine_similarity_std",
    "force_component_mean_pred",
    "force_component_std_pred",
    "force_component_mean_target",
    "force_component_std_target",
    "equivariance_error",
    "force_equivariance_error",
    "parameter_count",
    "runtime_per_batch_sec",
    "force_scale_value",
    "force_train_rms",
    "force_train_std",
    "force_train_component_rms",
    "force_train_vector_rms",
    "fixed_force_scale_value",
    "huber_delta",
    "force_output_scale",
    "initial_force_output_scale",
    "force_output_scale_regularization",
    "train_eval_force_mae",
    "train_eval_force_vector_l2_mae",
    "train_eval_zero_force_vector_l2_mae",
    "train_eval_mean_force_vector_l2_mae",
    "train_eval_force_mae_improvement_vs_zero_pct",
    "train_eval_force_vector_l2_mae_improvement_vs_zero_pct",
    "train_eval_pred_to_target_force_norm_ratio",
    "train_eval_force_cosine_similarity_mean",
    "val_force_mae_epoch1",
    "val_force_mae_final",
    "val_force_vector_l2_mae_epoch1",
    "val_force_vector_l2_mae_final",
    "val_force_mae_decreased",
    "learning_established",
]


def csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def label_value(value) -> str:
    if value is None:
        return "none"
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fieldnames})


def complete(path: Path, run_name: str, seed: int) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if not all(field in data for field in REQUIRED_METRIC_FIELDS + MOLECULAR_REQUIRED_FIELDS):
        return False
    return data.get("run_name") == run_name and int(data.get("seed", -1)) == seed


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in GROUP_FIELDS)
        groups.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(groups.items()):
        summary = {field: value for field, value in zip(GROUP_FIELDS, key)}
        summary["n"] = len(group)
        for field in NUMERIC_FIELDS:
            values = []
            for row in group:
                value = row.get(field)
                if value in (None, "", "nan"):
                    continue
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
            summary[f"{field}_mean"] = mean(values) if values else ""
            summary[f"{field}_std"] = stdev(values) if len(values) > 1 else (0.0 if values else "")
        out.append(summary)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--lrs", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--force-loss-types", nargs="+", choices=["mse", "mae", "huber"], default=None)
    parser.add_argument("--force-scale-normalizations", nargs="+", choices=["none", "train_force_rms", "train_force_std", "fixed"], default=None)
    parser.add_argument("--gradient-clip-norms", nargs="+", type=float, default=None)
    parser.add_argument("--output-head-init-scales", nargs="+", type=float, default=None)
    parser.add_argument("--weight-decays", nargs="+", type=float, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    add_molecular_override_args(parser)
    args = parser.parse_args()

    output_root = Path(args.output)
    rows = []
    failures = []
    for config_arg in args.configs:
        config_path = Path(config_arg)
        base_config = load_molecular_config(config_path)
        base_training = base_config.get("training", {})
        base_model = base_config.get("model", {})
        base_loss_types = args.force_loss_types or [str(base_training.get("force_loss_type", "mse"))]
        base_norms = args.force_scale_normalizations or [str(base_training.get("force_scale_normalization", "train_force_rms"))]
        base_grad = base_training.get("gradient_clip_norm", base_training.get("gradient_clip"))
        grad_values = args.gradient_clip_norms if args.gradient_clip_norms is not None else [base_grad]
        head_values = args.output_head_init_scales if args.output_head_init_scales is not None else [base_model.get("output_head_init_scale", 1.0) or 1.0]
        wd_values = args.weight_decays if args.weight_decays is not None else [base_training.get("weight_decay", 0.0) or 0.0]
        extra_suffix = any(
            value is not None
            for value in [args.gradient_clip_norms, args.output_head_init_scales, args.weight_decays]
        )
        for lr in args.lrs:
            lr_label = f"{lr:g}".replace(".", "p").replace("-", "m")
            for loss_type in base_loss_types:
                for norm in base_norms:
                    for grad_clip in grad_values:
                        for head_scale in head_values:
                            for weight_decay in wd_values:
                                suffix = (
                                    f"_gc{label_value(grad_clip)}_head{label_value(head_scale)}_wd{label_value(weight_decay)}"
                                    if extra_suffix
                                    else ""
                                )
                                for seed in args.seeds:
                                    run_name = f"{config_path.stem}_lr{lr_label}_{loss_type}_{norm}{suffix}_seed{seed}"
                                    run_dir = output_root / "runs" / run_name
                                    metrics_path = run_dir / "metrics.json"
                                    try:
                                        if not args.force and complete(metrics_path, run_name, seed):
                                            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                                            print(f"skipping completed run: {run_name}")
                                        else:
                                            config = copy.deepcopy(base_config)
                                            config["seed"] = seed
                                            config["run_name"] = run_name
                                            config["output_dir"] = str(run_dir)
                                            config.setdefault("dataset", {})["seed"] = seed
                                            config = apply_molecular_overrides(config, args)
                                            config["seed"] = seed
                                            config["run_name"] = run_name
                                            config["output_dir"] = str(run_dir)
                                            config.setdefault("dataset", {})["seed"] = seed
                                            training = config.setdefault("training", {})
                                            model_cfg = config.setdefault("model", {})
                                            training["lr"] = float(lr)
                                            training["force_loss_type"] = loss_type
                                            training["force_scale_normalization"] = norm
                                            if grad_clip is not None:
                                                training["gradient_clip_norm"] = float(grad_clip)
                                            model_cfg["output_head_init_scale"] = float(head_scale)
                                            training["weight_decay"] = float(weight_decay)
                                            metrics = train_molecular_from_config(config)
                                        metrics["learning_rate"] = float(lr)
                                        rows.append(metrics)
                                        print(
                                            f"run={run_name} force_mae={metrics['force_mae']:.6g} "
                                            f"vec_l2={metrics['force_vector_l2_mae']:.6g} "
                                            f"improve_zero={metrics['force_mae_improvement_vs_zero_pct']:.3g}"
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        failures.append(
                                            {
                                                "run_name": run_name,
                                                "config_name": config_path.name,
                                                "learning_rate": lr,
                                                "force_loss_type": loss_type,
                                                "force_scale_normalization": norm,
                                                "seed": seed,
                                                "error_type": type(exc).__name__,
                                                "error_message": str(exc),
                                                "traceback": traceback.format_exc(),
                                            }
                                        )
                                        print(f"FAILED run={run_name} error={type(exc).__name__}: {exc}")

    per_fields = REQUIRED_METRIC_FIELDS + MOLECULAR_REQUIRED_FIELDS + ["learning_rate", "last_checkpoint"]
    write_csv(output_root / "per_run_metrics.csv", rows, per_fields)
    summary = summarize(rows)
    summary_fields = GROUP_FIELDS + ["n"]
    for field in NUMERIC_FIELDS:
        summary_fields.extend([f"{field}_mean", f"{field}_std"])
    write_csv(output_root / "summary_mean_std.csv", summary, summary_fields)
    failure_path = output_root / "failed_runs.csv"
    if failures:
        write_csv(
            failure_path,
            failures,
            ["run_name", "config_name", "learning_rate", "force_loss_type", "force_scale_normalization", "seed", "error_type", "error_message", "traceback"],
        )
    elif failure_path.exists():
        failure_path.unlink()
    print(f"wrote {output_root / 'per_run_metrics.csv'}")
    print(f"wrote {output_root / 'summary_mean_std.csv'}")


if __name__ == "__main__":
    main()
