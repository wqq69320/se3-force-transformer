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
    "diagnostic_type",
    "training_mode",
    "split_type",
    "force_scale_normalization",
    "force_loss_type",
    "learning_rate",
    "gradient_clip_norm",
    "output_head_init_scale",
    "force_output_scale",
    "learnable_force_output_scale",
    "initial_force_output_scale",
    "force_output_scale_regularization",
    "graph_mode",
    "actual_hidden_irreps",
    "uses_non_scalar_hidden",
    "force_head_type",
    "uses_atom_pair_embedding",
    "uses_global_context",
    "global_context_type",
    "use_prototype_memory",
    "prototype_assignment",
    "weight_decay",
]

NUMERIC_FIELDS = [
    "seed",
    "num_frames_total",
    "num_frames_used",
    "num_train_frames",
    "num_val_frames",
    "num_test_frames",
    "num_train_batches",
    "num_val_batches",
    "num_test_batches",
    "batch_size",
    "epochs",
    "max_steps_per_epoch",
    "cutoff_radius",
    "average_neighbors",
    "edge_count_mean",
    "edge_count_max",
    "graph_build_time_sec",
    "force_mae",
    "force_vector_l2_mae",
    "zero_force_mae",
    "zero_force_vector_l2_mae",
    "mean_force_mae",
    "mean_force_vector_l2_mae",
    "force_mae_improvement_vs_zero_pct",
    "force_mae_improvement_vs_mean_pct",
    "force_vector_l2_mae_improvement_vs_zero_pct",
    "force_vector_l2_mae_improvement_vs_mean_pct",
    "target_force_norm_mean",
    "target_force_norm_median",
    "target_force_norm_p95",
    "target_force_norm_max",
    "pred_force_norm_mean",
    "pred_to_target_force_norm_ratio",
    "residual_force_norm_mean",
    "force_cosine_similarity_mean",
    "force_cosine_similarity_std",
    "force_component_mean_pred",
    "force_component_std_pred",
    "force_component_mean_target",
    "force_component_std_target",
    "train_eval_force_mae",
    "train_eval_force_vector_l2_mae",
    "train_eval_zero_force_mae",
    "train_eval_zero_force_vector_l2_mae",
    "train_eval_mean_force_mae",
    "train_eval_mean_force_vector_l2_mae",
    "train_eval_force_mae_improvement_vs_zero_pct",
    "train_eval_force_mae_improvement_vs_mean_pct",
    "train_eval_force_vector_l2_mae_improvement_vs_zero_pct",
    "train_eval_force_vector_l2_mae_improvement_vs_mean_pct",
    "train_eval_pred_to_target_force_norm_ratio",
    "train_eval_force_cosine_similarity_mean",
    "equivariance_error",
    "force_equivariance_error",
    "parameter_count",
    "runtime_per_batch_sec",
    "global_context_dim",
    "prototype_count",
    "force_scale_value",
    "force_train_rms",
    "force_train_std",
    "force_train_component_rms",
    "force_train_vector_rms",
    "fixed_force_scale_value",
    "force_loss_weight",
    "huber_delta",
    "force_output_scale",
    "initial_force_output_scale",
    "force_output_scale_regularization",
    "initial_oracle_scalar_c",
    "initial_oracle_force_vector_l2_mae",
    "initial_oracle_pred_to_target_force_norm_ratio",
    "initial_oracle_force_cosine_similarity_mean",
    "train_eval_target_force_norm_median",
    "train_eval_target_force_norm_p95",
    "train_eval_target_force_norm_max",
    "train_force_head_weight_norm_final",
    "train_force_output_scale_final",
    "train_force_head_grad_norm_max",
    "train_backbone_grad_norm_max",
    "train_edge_mlp_grad_norm_max",
    "train_total_grad_norm_before_clip_max",
    "train_total_grad_norm_after_clip_max",
    "train_learnable_force_output_scale_value_final",
    "train_learnable_force_output_scale_grad_max",
    "val_force_mae_epoch1",
    "val_force_mae_final",
    "val_force_vector_l2_mae_epoch1",
    "val_force_vector_l2_mae_final",
    "val_pred_to_target_force_norm_ratio_final",
    "val_force_cosine_similarity_mean_final",
    "learning_established",
]

PER_EXTRA_FIELDS = [
    "learning_rate",
    "epochs",
    "max_steps_per_epoch",
    "last_checkpoint",
    "train_eval_force_mae",
    "train_eval_force_rmse",
    "train_eval_force_vector_l2_mae",
    "train_eval_force_vector_l2_rmse",
    "train_eval_zero_force_mae",
    "train_eval_zero_force_vector_l2_mae",
    "train_eval_mean_force_mae",
    "train_eval_mean_force_vector_l2_mae",
    "train_eval_force_mae_improvement_vs_zero_pct",
    "train_eval_force_mae_improvement_vs_mean_pct",
    "train_eval_force_vector_l2_mae_improvement_vs_zero_pct",
    "train_eval_force_vector_l2_mae_improvement_vs_mean_pct",
    "train_eval_pred_to_target_force_norm_ratio",
    "train_eval_residual_force_norm_mean",
    "train_eval_force_cosine_similarity_mean",
    "train_eval_force_cosine_similarity_std",
    "train_eval_target_force_norm_mean",
    "train_eval_pred_force_norm_mean",
    "initial_oracle_scalar_c",
    "initial_oracle_force_vector_l2_mae",
    "initial_oracle_pred_to_target_force_norm_ratio",
    "initial_oracle_force_cosine_similarity_mean",
    "train_force_norm_distribution",
    "train_force_head_weight_norm_final",
    "train_force_head_bias_norm_final",
    "train_force_output_scale_final",
    "train_force_final_activation_norm_final",
    "train_last_hidden_norm_final",
    "train_message_norm_mean_final",
    "train_edge_message_norm_mean_final",
    "train_force_head_output_norm_final",
    "train_force_head_grad_norm_max",
    "train_message_passing_grad_norm_max",
    "train_backbone_grad_norm_max",
    "train_edge_mlp_grad_norm_max",
    "train_total_grad_norm_before_clip_max",
    "train_total_grad_norm_after_clip_max",
    "train_max_grad_norm_before_clip_max",
    "train_max_grad_norm_after_clip_max",
    "train_learnable_force_output_scale_value_final",
    "train_learnable_force_output_scale_grad_max",
]


def csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def label_value(value) -> str:
    if value is None:
        return "none"
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def parse_optional_float(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null", "off", "no"}:
        return None
    return float(value)


def parse_bool(value) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"expected boolean value, got {value!r}")


def printable_float(value) -> float:
    if value in (None, "", "nan"):
        return float("nan")
    return float(value)


def model_label(value: str) -> str:
    return str(value).replace("se3_transformer", "se3").replace("molecular_", "")


def apply_model_choice(config: dict, model_choice: str) -> None:
    model_cfg = config.setdefault("model", {})
    choice = model_choice.lower()
    if choice in {"se3", "se3_scalar", "se3_transformer", "molecular_se3"}:
        model_cfg["name"] = "se3_transformer"
    elif choice in {"se3_full", "se3_full_l1"}:
        model_cfg["name"] = "se3_full_l1"
        model_cfg.setdefault("lmax", 1)
        model_cfg.setdefault("channels_by_l", {0: 64, 1: 16})
        model_cfg.setdefault("atom_embedding_dim", 16)
    elif choice in {"se3_full_l2"}:
        model_cfg["name"] = "se3_full_l2"
        model_cfg.setdefault("lmax", 2)
        model_cfg.setdefault("channels_by_l", {0: 64, 1: 16, 2: 8})
        model_cfg.setdefault("atom_embedding_dim", 16)
    elif choice in {"egnn", "molecular_egnn"}:
        model_cfg["name"] = "egnn"
    elif choice in {"tfn", "baseline_tfn", "molecular_tfn"}:
        model_cfg["name"] = "tfn"
    elif choice in {"radial", "pairwise_radial", "radial_baseline"}:
        model_cfg["name"] = "radial"
        model_cfg.setdefault("hidden_dim", 192)
        model_cfg.setdefault("radial_hidden_dim", 256)
    elif choice in {"radial_pair", "high_capacity_radial_pair"}:
        model_cfg["name"] = "radial_pair"
        model_cfg.setdefault("hidden_dim", 256)
        model_cfg.setdefault("radial_num_basis", 32)
        model_cfg.setdefault("radial_hidden_dim", 384)
        model_cfg.setdefault("edge_mlp_hidden_dim", 384)
        model_cfg.setdefault("edge_mlp_layers", 4)
        model_cfg.setdefault("use_atom_pair_embedding", True)
        model_cfg.setdefault("pair_embedding_dim", 32)
    elif choice in {"global_context_radial", "global_context_radial_pair"}:
        model_cfg["name"] = "global_context_radial"
        model_cfg.setdefault("max_atoms", 32)
        model_cfg.setdefault("hidden_dim", 192)
        model_cfg.setdefault("radial_num_basis", 32)
        model_cfg.setdefault("radial_hidden_dim", 256)
        model_cfg.setdefault("edge_mlp_hidden_dim", 384)
        model_cfg.setdefault("edge_mlp_layers", 4)
        model_cfg.setdefault("global_context_dim", 128)
        model_cfg.setdefault("global_layers", 2)
        model_cfg.setdefault("use_atom_pair_embedding", True)
        model_cfg.setdefault("pair_embedding_dim", 32)
        model_cfg.setdefault("graph_mode", "full")
    elif choice in {"global_coeff", "global_invariant_coefficients"}:
        model_cfg["name"] = "global_coeff"
        model_cfg.setdefault("max_atoms", 32)
        model_cfg.setdefault("hidden_dim", 512)
        model_cfg.setdefault("num_layers", 4)
        model_cfg.setdefault("global_context_dim", 256)
        model_cfg.setdefault("edge_mlp_hidden_dim", 512)
        model_cfg.setdefault("edge_mlp_layers", 2)
        model_cfg.setdefault("graph_mode", "full")
    elif choice in {"internal_energy", "internal_coordinate_energy"}:
        model_cfg["name"] = "internal_energy"
        model_cfg.setdefault("max_atoms", 32)
        model_cfg.setdefault("hidden_dim", 512)
        model_cfg.setdefault("num_layers", 4)
        model_cfg.setdefault("graph_mode", "full")
    elif choice in {"painn_lite", "painn"}:
        model_cfg["name"] = "painn_lite"
        model_cfg.setdefault("hidden_dim", 128)
        model_cfg.setdefault("vector_channels", 16)
        model_cfg.setdefault("num_layers", 3)
        model_cfg.setdefault("radial_hidden_dim", 192)
        model_cfg.setdefault("graph_mode", "full")
    elif choice in {"mlp", "mlp_memorizer", "coordinate_mlp_memorizer"}:
        model_cfg["name"] = "mlp_memorizer"
        model_cfg.setdefault("hidden_dim", 512)
        model_cfg.setdefault("num_layers", 4)
        model_cfg.setdefault("max_atoms", 64)
    else:
        raise ValueError(f"unknown diagnostic model: {model_choice}")


def diagnostic_type_for(config: dict) -> str:
    training = config.get("training", {})
    model_name = str(config.get("model", {}).get("name", ""))
    configured = str(training.get("diagnostic_type") or config.get("diagnostic_type") or "")
    if configured and configured != "overfit":
        return configured
    if model_name == "global_coeff":
        return "equivariant_global_memorizer"
    if model_name == "global_context_radial":
        return "equivariant_global_context_radial"
    if model_name == "internal_energy":
        return "equivariant_internal_energy_memorizer"
    if model_name == "painn_lite":
        return "equivariant_painn_lite"
    return configured or "overfit"


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
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
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
    parser.add_argument("--losses", nargs="+", choices=["mse", "mae", "huber", "vector_l2", "normalized_vector_l2"], default=None)
    parser.add_argument("--huber-deltas", nargs="+", type=float, default=None)
    parser.add_argument("--gradient-clip-norms", nargs="+", type=float, default=None)
    parser.add_argument("--gradient-clips", nargs="+", default=None)
    parser.add_argument("--output-head-init-scales", nargs="+", type=float, default=None)
    parser.add_argument("--force-output-scales", nargs="+", type=float, default=None)
    parser.add_argument("--learnable-force-output-scale", action="store_true")
    parser.add_argument("--learnable-force-output-scales", nargs="+", default=None)
    parser.add_argument("--force-scale-normalizations", nargs="+", choices=["none", "train_force_rms", "train_force_component_rms", "train_force_vector_rms", "fixed"], default=None)
    parser.add_argument("--fixed-force-scales", nargs="+", type=float, default=None)
    parser.add_argument("--graph-modes", nargs="+", choices=["cutoff", "full"], default=None)
    parser.add_argument("--use-angles", choices=["true", "false"], default=None)
    parser.add_argument("--global-hidden-dim", type=int, default=None)
    parser.add_argument("--global-layers", type=int, default=None)
    parser.add_argument("--vector-channels", type=int, default=None)
    parser.add_argument("--weight-decays", nargs="+", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    add_molecular_override_args(parser)
    args = parser.parse_args()

    output_root = Path(args.output)
    rows = []
    failures = []
    attempted_runs = 0
    for config_arg in args.configs:
        config_path = Path(config_arg)
        base_config = load_molecular_config(config_path)
        base_training = base_config.get("training", {})
        base_model = base_config.get("model", {})
        loss_types = args.losses or args.force_loss_types or [str(base_training.get("force_loss_type", "mse"))]
        huber_values = args.huber_deltas if args.huber_deltas is not None else [float(base_training.get("huber_delta", 1.0))]
        if args.gradient_clips is not None:
            grad_values = [parse_optional_float(value) for value in args.gradient_clips]
        elif args.gradient_clip_norms is not None:
            grad_values = list(args.gradient_clip_norms)
        else:
            grad_values = [base_training.get("gradient_clip_norm", base_training.get("gradient_clip"))]
        head_values = args.output_head_init_scales if args.output_head_init_scales is not None else [base_model.get("output_head_init_scale", 1.0) or 1.0]
        output_scale_values = args.force_output_scales if args.force_output_scales is not None else [base_model.get("force_output_scale", 1.0) or 1.0]
        if args.learnable_force_output_scales is not None:
            learnable_values = [parse_bool(value) for value in args.learnable_force_output_scales]
        elif args.learnable_force_output_scale:
            learnable_values = [True]
        else:
            learnable_values = [bool(base_model.get("learnable_force_output_scale", False))]
        normalizations = args.force_scale_normalizations if args.force_scale_normalizations is not None else [str(base_training.get("force_scale_normalization", "train_force_rms"))]
        fixed_scale_values = args.fixed_force_scales if args.fixed_force_scales is not None else [float(base_training.get("fixed_force_scale_value", base_training.get("force_scale_value", 1.0)))]
        graph_modes = args.graph_modes if args.graph_modes is not None else [str(base_model.get("graph_mode", base_config.get("dataset", {}).get("graph_mode", "cutoff")))]
        wd_values = args.weight_decays if args.weight_decays is not None else [base_training.get("weight_decay", 0.0) or 0.0]
        model_choices = args.models or [str(base_model.get("name", "se3_transformer"))]
        for model_choice in model_choices:
            for lr in args.lrs:
                lr_label = label_value(lr)
                for loss_type in loss_types:
                    active_huber_values = huber_values if loss_type == "huber" else [float(base_training.get("huber_delta", 1.0))]
                    for huber_delta in active_huber_values:
                        for grad_clip in grad_values:
                            for head_scale in head_values:
                                for output_scale in output_scale_values:
                                    for learnable_scale in learnable_values:
                                        for graph_mode in graph_modes:
                                            for normalization in normalizations:
                                                active_fixed_values = fixed_scale_values if normalization == "fixed" else [None]
                                                for fixed_scale in active_fixed_values:
                                                    for weight_decay in wd_values:
                                                        for seed in args.seeds:
                                                            if args.max_runs is not None and attempted_runs >= args.max_runs:
                                                                continue
                                                            attempted_runs += 1
                                                            huber_suffix = f"_huber{label_value(huber_delta)}" if loss_type == "huber" else ""
                                                            fixed_suffix = f"_fixed{label_value(fixed_scale)}" if fixed_scale is not None else ""
                                                            learn_label = "learn" if learnable_scale else "static"
                                                            model_suffix = model_label(model_choice)
                                                            run_name = (
                                                                f"{config_path.stem}_{model_suffix}_lr{lr_label}_{loss_type}{huber_suffix}_{normalization}{fixed_suffix}"
                                                                f"_gc{label_value(grad_clip)}_graph{graph_mode}_head{label_value(head_scale)}"
                                                                f"_out{label_value(output_scale)}_{learn_label}_wd{label_value(weight_decay)}_seed{seed}"
                                                            )
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
                                                                    apply_model_choice(config, model_choice)
                                                                    diagnostic_type = diagnostic_type_for(config)
                                                                    config["diagnostic_type"] = diagnostic_type
                                                                    training["diagnostic_type"] = diagnostic_type
                                                                    training["diagnostic_logging"] = True
                                                                    training["lr"] = float(lr)
                                                                    training["force_loss_type"] = loss_type
                                                                    training["huber_delta"] = float(huber_delta)
                                                                    training["force_scale_normalization"] = normalization
                                                                    if fixed_scale is not None:
                                                                        training["fixed_force_scale_value"] = float(fixed_scale)
                                                                        training["force_scale_value"] = float(fixed_scale)
                                                                    if grad_clip is None:
                                                                        training.pop("gradient_clip_norm", None)
                                                                        training.pop("gradient_clip", None)
                                                                    else:
                                                                        training["gradient_clip_norm"] = float(grad_clip)
                                                                    if args.epochs is not None:
                                                                        training["epochs"] = int(args.epochs)
                                                                    model_cfg["graph_mode"] = graph_mode
                                                                    if args.use_angles is not None:
                                                                        model_cfg["use_angles"] = parse_bool(args.use_angles)
                                                                    if args.global_hidden_dim is not None:
                                                                        model_cfg["hidden_dim"] = int(args.global_hidden_dim)
                                                                        model_cfg["edge_mlp_hidden_dim"] = int(args.global_hidden_dim)
                                                                    if args.global_layers is not None:
                                                                        model_cfg["num_layers"] = int(args.global_layers)
                                                                    if args.vector_channels is not None:
                                                                        model_cfg["vector_channels"] = int(args.vector_channels)
                                                                    model_cfg["output_head_init_scale"] = float(head_scale)
                                                                    model_cfg["force_output_scale"] = float(output_scale)
                                                                    model_cfg["initial_force_output_scale"] = float(output_scale)
                                                                    model_cfg["learnable_force_output_scale"] = bool(learnable_scale)
                                                                    training["weight_decay"] = float(weight_decay)
                                                                    metrics = train_molecular_from_config(config)
                                                                metrics["learning_rate"] = float(lr)
                                                                rows.append(metrics)
                                                                print(
                                                                    f"run={run_name} train_vec_l2={printable_float(metrics.get('train_eval_force_vector_l2_mae')):.6g} "
                                                                    f"train_improve_zero={printable_float(metrics.get('train_eval_force_vector_l2_mae_improvement_vs_zero_pct')):.3g} "
                                                                    f"ratio={printable_float(metrics.get('train_eval_pred_to_target_force_norm_ratio')):.3g} "
                                                                    f"cos={printable_float(metrics.get('train_eval_force_cosine_similarity_mean')):.3g}"
                                                                )
                                                            except Exception as exc:  # noqa: BLE001
                                                                failures.append(
                                                                    {
                                                                        "run_name": run_name,
                                                                        "config_name": config_path.name,
                                                                        "model": model_choice,
                                                                        "learning_rate": lr,
                                                                        "force_loss_type": loss_type,
                                                                        "huber_delta": huber_delta,
                                                                        "gradient_clip_norm": grad_clip,
                                                                        "force_output_scale": output_scale,
                                                                        "learnable_force_output_scale": learnable_scale,
                                                                        "graph_mode": graph_mode,
                                                                        "force_scale_normalization": normalization,
                                                                        "fixed_force_scale_value": fixed_scale,
                                                                        "seed": seed,
                                                                        "error_type": type(exc).__name__,
                                                                        "error_message": str(exc),
                                                                        "traceback": traceback.format_exc(),
                                                                    }
                                                                )
                                                                print(f"FAILED run={run_name} error={type(exc).__name__}: {exc}")

    per_fields = list(dict.fromkeys(REQUIRED_METRIC_FIELDS + MOLECULAR_REQUIRED_FIELDS + PER_EXTRA_FIELDS))
    write_csv(output_root / "per_run_metrics.csv", rows, per_fields)
    summary = summarize(rows)
    summary_fields = GROUP_FIELDS + ["n"]
    for field in NUMERIC_FIELDS:
        summary_fields.extend([f"{field}_mean", f"{field}_std"])
    write_csv(output_root / "summary_mean_std.csv", summary, summary_fields)
    best_rows = sorted(
        rows,
        key=lambda row: printable_float(row.get("train_eval_force_vector_l2_mae")),
    )[: min(20, len(rows))]
    write_csv(output_root / "best_runs.csv", best_rows, per_fields)
    failure_path = output_root / "failed_runs.csv"
    if failures:
        write_csv(
            failure_path,
            failures,
            [
                "run_name",
                "config_name",
                "model",
                "learning_rate",
                "force_loss_type",
                "huber_delta",
                "gradient_clip_norm",
                "force_output_scale",
                "learnable_force_output_scale",
                "graph_mode",
                "force_scale_normalization",
                "fixed_force_scale_value",
                "seed",
                "error_type",
                "error_message",
                "traceback",
            ],
        )
    elif failure_path.exists():
        failure_path.unlink()
    print(f"wrote {output_root / 'per_run_metrics.csv'}")
    print(f"wrote {output_root / 'summary_mean_std.csv'}")
    print(f"wrote {output_root / 'best_runs.csv'}")


if __name__ == "__main__":
    main()
