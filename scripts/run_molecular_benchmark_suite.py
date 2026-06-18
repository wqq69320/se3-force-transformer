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
from se3force.training.molecular_trainer import load_molecular_config, train_molecular_from_config
from se3force.training.molecular_overrides import add_molecular_override_args, apply_molecular_overrides

NUMERIC_FIELDS = [
    "seed",
    "num_train_samples",
    "num_val_samples",
    "num_test_samples",
    "force_mae",
    "force_rmse",
    "rotated_force_mae",
    "rotated_force_rmse",
    "force_vector_l2_mae",
    "force_vector_l2_rmse",
    "rotated_force_vector_l2_mae",
    "rotated_force_vector_l2_rmse",
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
    "energy_mae",
    "energy_rmse",
    "energy_mae_raw",
    "energy_rmse_raw",
    "energy_mae_centered",
    "energy_rmse_centered",
    "energy_invariance_error",
    "energy_train_mean",
    "energy_train_std",
    "equivariance_error",
    "force_equivariance_error",
    "parameter_count",
    "runtime_per_batch_sec",
    "num_atoms_mean",
    "num_atoms_max",
    "configuration_dim_mean",
    "force_dim_mean",
    "num_frames_total",
    "num_frames_used",
    "num_train_frames",
    "num_val_frames",
    "num_test_frames",
    "num_train_batches",
    "num_val_batches",
    "num_test_batches",
    "batch_size",
    "cutoff_radius",
    "average_neighbors",
    "edge_count_mean",
    "edge_count_max",
    "graph_build_time_sec",
    "force_loss_weight",
    "energy_loss_weight",
    "force_scale_value",
    "force_train_rms",
    "force_train_std",
    "force_train_component_std",
    "force_train_component_rms",
    "force_train_vector_rms",
    "fixed_force_scale_value",
    "huber_delta",
    "gradient_clip_norm",
    "output_head_init_scale",
    "force_output_scale",
    "initial_force_output_scale",
    "force_output_scale_regularization",
    "weight_decay",
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
    "val_force_rmse_final",
    "val_force_vector_l2_mae_epoch1",
    "val_force_vector_l2_mae_final",
    "val_force_mae_decreased",
    "learning_established",
]
GROUP_FIELDS = [
    "config_name",
    "model_name",
    "model_class",
    "backbone_class",
    "architecture_signature",
    "lmax",
    "hidden_irreps",
    "use_attention",
    "use_gate",
    "energy_centering",
    "energy_standardization",
    "energy_loss_on_centered",
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
    "dataset_name",
    "molecule_name",
    "data_source_type",
    "is_fake_or_synthetic",
    "is_real_rmd17",
    "dataset_path_basename",
    "training_mode",
    "split_type",
    "force_unit",
    "energy_unit",
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fieldnames})


def csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def complete(path: Path, *, config_name: str, run_name: str, seed: int) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if not all(field in data for field in REQUIRED_METRIC_FIELDS + MOLECULAR_REQUIRED_FIELDS):
        return False
    return data.get("config_name") == config_name and data.get("run_name") == run_name and int(data.get("seed", -1)) == seed


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in GROUP_FIELDS)
        groups.setdefault(key, []).append(row)
    out_rows = []
    for key, group in sorted(groups.items()):
        out = {field: value for field, value in zip(GROUP_FIELDS, key)}
        out["n"] = len(group)
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
            out[f"{field}_mean"] = mean(values) if values else ""
            out[f"{field}_std"] = stdev(values) if len(values) > 1 else (0.0 if values else "")
        out_rows.append(out)
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    add_molecular_override_args(parser)
    args = parser.parse_args()

    output_root = Path(args.output)
    rows = []
    failures = []
    for config_arg in args.configs:
        config_path = Path(config_arg)
        for seed in args.seeds:
            run_name = f"{config_path.stem}_seed{seed}"
            run_dir = output_root / "runs" / run_name
            metrics_path = run_dir / "metrics.json"
            try:
                if not args.force and complete(metrics_path, config_name=config_path.name, run_name=run_name, seed=seed):
                    print(f"skipping completed run: {run_name}")
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    config = copy.deepcopy(load_molecular_config(config_path))
                    config["seed"] = seed
                    config["run_name"] = run_name
                    config["output_dir"] = str(run_dir)
                    config.setdefault("dataset", {})["seed"] = seed
                    config = apply_molecular_overrides(config, args)
                    config["seed"] = seed
                    config["run_name"] = run_name
                    config["output_dir"] = str(run_dir)
                    config.setdefault("dataset", {})["seed"] = seed
                    metrics = train_molecular_from_config(config)
                rows.append(metrics)
                print(f"run={run_name} force_mae={metrics['force_mae']:.6g} equivariance_error={metrics['equivariance_error']:.3e}")
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "run_name": run_name,
                        "config_name": config_path.name,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"FAILED run={run_name} error={type(exc).__name__}: {exc}")

    per_fields = REQUIRED_METRIC_FIELDS + MOLECULAR_REQUIRED_FIELDS + ["last_checkpoint"]
    write_csv(output_root / "per_run_metrics.csv", rows, per_fields)
    summary = summarize(rows)
    summary_fields = GROUP_FIELDS + ["n"]
    for field in NUMERIC_FIELDS:
        summary_fields.extend([f"{field}_mean", f"{field}_std"])
    write_csv(output_root / "summary_mean_std.csv", summary, summary_fields)
    failure_path = output_root / "failed_runs.csv"
    if failures:
        write_csv(failure_path, failures, ["run_name", "config_name", "seed", "error_type", "error_message", "traceback"])
        print(f"wrote {failure_path}")
    elif failure_path.exists():
        failure_path.unlink()
    print(f"wrote {output_root / 'per_run_metrics.csv'}")
    print(f"wrote {output_root / 'summary_mean_std.csv'}")


if __name__ == "__main__":
    main()
