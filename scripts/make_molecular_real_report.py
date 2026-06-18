#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: str | None) -> str:
    if value in (None, ""):
        return "n/a"
    value_f = float(value)
    return f"{value_f:.4e}" if abs(value_f) < 1e-3 or abs(value_f) > 1e4 else f"{value_f:.4g}"


def redact_path(path: str | None) -> str:
    if not path:
        return "not recorded"
    p = Path(path)
    parts = p.parts
    if "data" in parts:
        return str(Path(*parts[parts.index("data") :]))
    if "datasets" in parts:
        return str(Path(*parts[parts.index("datasets") :]))
    return p.name


def table(rows: list[dict]) -> str:
    lines = [
        "| config | mode | n | frames train/val/test | force MAE | vector L2 MAE | zero MAE | mean MAE | improvement vs zero | energy raw MAE | centered energy | equivariance | runtime |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["config_name"].replace(".yaml", ""),
                    row.get("training_mode", ""),
                    row.get("n", ""),
                    f"{fmt(row.get('num_train_frames_mean'))}/{fmt(row.get('num_val_frames_mean'))}/{fmt(row.get('num_test_frames_mean'))}",
                    fmt(row.get("force_mae_mean")),
                    fmt(row.get("force_vector_l2_mae_mean")),
                    fmt(row.get("zero_force_mae_mean")),
                    fmt(row.get("mean_force_mae_mean")),
                    fmt(row.get("force_mae_improvement_vs_zero_pct_mean")),
                    fmt(row.get("energy_mae_raw_mean")),
                    row.get("energy_centering", ""),
                    fmt(row.get("equivariance_error_mean")),
                    fmt(row.get("runtime_per_batch_sec_mean")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def cutoff_table(rows: list[dict]) -> str:
    cutoff_rows = [row for row in rows if row.get("cutoff_radius_mean") not in (None, "")]
    if not cutoff_rows:
        return "No cutoff-specific rows were present in this summary."
    lines = [
        "| config | cutoff | force MAE | force RMSE | vector L2 MAE | avg neighbors | graph build sec | runtime sec/batch | equivariance |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in cutoff_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["config_name"].replace(".yaml", ""),
                    fmt(row.get("cutoff_radius_mean")),
                    fmt(row.get("force_mae_mean")),
                    fmt(row.get("force_rmse_mean")),
                    fmt(row.get("force_vector_l2_mae_mean")),
                    fmt(row.get("average_neighbors_mean")),
                    fmt(row.get("graph_build_time_sec_mean")),
                    fmt(row.get("runtime_per_batch_sec_mean")),
                    fmt(row.get("equivariance_error_mean")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def unique_fmt(rows: list[dict], field: str) -> str:
    values = sorted({fmt(row.get(field)) for row in rows if row.get(field) not in (None, "")})
    return ", ".join(values) if values else "not recorded"


def unique_text(rows: list[dict], field: str) -> str:
    values = sorted({str(row.get(field)) for row in rows if row.get(field) not in (None, "")})
    return ", ".join(values) if values else "not recorded"


def _family_comparison_status(rows: list[dict]) -> str:
    names = " ".join(row.get("config_name", "").lower() for row in rows)
    has_egnn = "egnn" in names
    has_tfn = "tfn" in names
    has_se3 = "se3" in names
    return "yes" if has_egnn and has_tfn and has_se3 else "no"


def truthy(value: str | None) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def validation_level(rows: list[dict]) -> tuple[str, str]:
    source_types = {row.get("data_source_type", "") for row in rows}
    seed_ok = all(int(float(row.get("n", 0))) >= 3 for row in rows)
    has_families = _family_comparison_status(rows) == "yes"
    if "synthetic_molecular" in source_types:
        return "Pipeline smoke", "Synthetic molecular data; use only for pipeline and symmetry checks."
    if "fake_rmd17_npz_smoke" in source_types:
        return "Fake rMD17-style local file", "This is a fake rMD17-style smoke file, not a real rMD17 benchmark."
    if all(truthy(row.get("is_real_rmd17")) for row in rows):
        if seed_ok and has_families:
            return "Real rMD17 multi-seed", "Real rMD17-style local file with enough seeds for cautious within-run comparison."
        return "Real rMD17 single molecule", "Real rMD17-style local file, but seed/model coverage is incomplete for ranking claims."
    if "local_md22_npz" in source_types:
        return "MD22 scale", "Local MD22-style file; inspect molecule and seed coverage before claims."
    return "Pipeline smoke", "Data source is not sufficiently classified for benchmark claims."


def distinct_model_identities(rows: list[dict]) -> bool:
    signatures = {row.get("architecture_signature", "") for row in rows if row.get("architecture_signature")}
    families = [row for row in rows if any(token in row.get("config_name", "").lower() for token in ("egnn", "tfn", "se3"))]
    return len(signatures) >= min(3, len(families))


def field_present(rows: list[dict], field: str) -> bool:
    return bool(rows) and field in rows[0] and any(row.get(field) not in (None, "", "nan") for row in rows)


def direct_family_present(rows: list[dict]) -> bool:
    direct = [row for row in rows if row.get("training_mode") == "direct_force"]
    names = " ".join(row.get("config_name", "").lower() for row in direct)
    return all(token in names for token in ("egnn", "tfn", "se3"))


def baselines_present(rows: list[dict]) -> tuple[bool, bool]:
    return field_present(rows, "zero_force_mae_mean"), field_present(rows, "mean_force_mae_mean")


def vector_metrics_present(rows: list[dict]) -> bool:
    return field_present(rows, "force_vector_l2_mae_mean") and field_present(rows, "force_vector_l2_rmse_mean")


def model_improves_over_baselines(rows: list[dict]) -> bool:
    if not baselines_present(rows) == (True, True):
        return False
    for row in rows:
        if row.get("training_mode") != "direct_force":
            continue
        try:
            model = float(row["force_mae_mean"])
            zero = float(row["zero_force_mae_mean"])
            mean = float(row["mean_force_mae_mean"])
        except (KeyError, TypeError, ValueError):
            return False
        if not (model < zero and model < mean):
            return False
    return True


def energy_normalization_ok(rows: list[dict]) -> bool:
    energy_rows = [row for row in rows if row.get("training_mode") == "energy_force"]
    return all(truthy(row.get("energy_centering")) or truthy(row.get("energy_standardization")) for row in energy_rows)


def comparable_frame_counts(rows: list[dict]) -> bool:
    fields = ["num_train_frames_mean", "num_val_frames_mean", "num_test_frames_mean", "batch_size_mean"]
    for field in fields:
        values = {row.get(field) for row in rows if row.get(field) not in (None, "")}
        if len(values) > 1:
            return False
    return True


def cutoff_schema_complete(rows: list[dict]) -> bool:
    required = [
        "config_name",
        "model_name",
        "model_class",
        "backbone_class",
        "architecture_signature",
        "dataset_name",
        "molecule_name",
        "data_source_type",
        "is_fake_or_synthetic",
        "is_real_rmd17",
        "training_mode",
        "split_type",
        "n",
        "cutoff_radius_mean",
        "cutoff_radius_std",
        "force_mae_mean",
        "force_mae_std",
        "force_rmse_mean",
        "force_rmse_std",
        "rotated_force_mae_mean",
        "rotated_force_mae_std",
        "force_vector_l2_mae_mean",
        "force_vector_l2_mae_std",
        "force_vector_l2_rmse_mean",
        "force_vector_l2_rmse_std",
        "equivariance_error_mean",
        "equivariance_error_std",
        "force_equivariance_error_mean",
        "force_equivariance_error_std",
        "average_neighbors_mean",
        "average_neighbors_std",
        "edge_count_mean_mean",
        "edge_count_mean_std",
        "edge_count_max_mean",
        "edge_count_max_std",
        "graph_build_time_sec_mean",
        "graph_build_time_sec_std",
        "runtime_per_batch_sec_mean",
        "runtime_per_batch_sec_std",
    ]
    return bool(rows) and all(field in rows[0] for field in required)


def ranking_claim_status(rows: list[dict], input_path: Path) -> tuple[bool, list[str]]:
    reasons = []
    if not rows or not all(truthy(row.get("is_real_rmd17")) for row in rows):
        reasons.append("data is not confirmed real rMD17")
    if any(truthy(row.get("is_fake_or_synthetic")) for row in rows):
        reasons.append("fake or synthetic data present")
    if not rows or not all(int(float(row.get("n", 0))) >= 3 for row in rows):
        reasons.append("fewer than 3 seeds")
    zero_ok, mean_ok = baselines_present(rows)
    if not zero_ok:
        reasons.append("zero baseline missing")
    if not mean_ok:
        reasons.append("mean baseline missing")
    if not direct_family_present(rows):
        reasons.append("EGNN, TFN, and SE3 direct-force models are not all present")
    if not vector_metrics_present(rows):
        reasons.append("vector L2 metrics missing")
    if not model_improves_over_baselines(rows):
        reasons.append("models do not all improve over zero and mean baselines")
    if not energy_normalization_ok(rows):
        reasons.append("energy-force run lacks energy centering/standardization")
    if not distinct_model_identities(rows):
        reasons.append("model identity signatures are not distinct enough")
    if not comparable_frame_counts(rows):
        reasons.append("frame counts or batch sizes are not comparable")
    if (input_path.parent / "failed_runs.csv").exists():
        reasons.append("failure log exists")
    return not reasons, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--molecule", default=None)
    args = parser.parse_args()
    input_path = Path(args.input)
    rows = read_rows(input_path)
    molecule = args.molecule or (rows[0].get("molecule_name") if rows else "unknown")
    split_types = sorted({row.get("split_type", "") for row in rows if row.get("split_type")})
    units = sorted({row.get("force_unit", "") for row in rows if row.get("force_unit")})
    energy_units = sorted({row.get("energy_unit", "") for row in rows if row.get("energy_unit")})
    max_equiv = max((float(row["equivariance_error_mean"]) for row in rows if row.get("equivariance_error_mean")), default=float("nan"))
    level, level_note = validation_level(rows)
    ranking_allowed, ranking_reasons = ranking_claim_status(rows, input_path)
    basename = Path(args.data_path).name if args.data_path else unique_text(rows, "dataset_path_basename")
    text = f"""# Real Molecular Benchmark Report

This report summarizes local molecular benchmark outputs. It does not claim SOTA performance or transferable chemistry.

## Validation Level

- Level: {level}
- Evidence note: {level_note}
- Architecture ranking claims allowed: {'yes' if ranking_allowed else 'no'}
- Ranking claim blockers: {', '.join(ranking_reasons) if ranking_reasons else 'none'}

{'Architecture superiority is not established by this run.' if not ranking_allowed else 'Architecture ranking gates passed for this run; interpret differences cautiously.'}

## Dataset

- Molecule: {molecule}
- Dataset path basename: {basename}
- Data source type(s): {unique_text(rows, 'data_source_type')}
- Fake or synthetic: {unique_text(rows, 'is_fake_or_synthetic')}
- Real rMD17: {unique_text(rows, 'is_real_rmd17')}
- Num frames: {unique_fmt(rows, 'num_frames_total_mean')}
- Num frames used: {unique_fmt(rows, 'num_frames_used_mean')}
- Train/val/test frames: {unique_fmt(rows, 'num_train_frames_mean')} / {unique_fmt(rows, 'num_val_frames_mean')} / {unique_fmt(rows, 'num_test_frames_mean')}
- Train/val/test batches: {unique_fmt(rows, 'num_train_batches_mean')} / {unique_fmt(rows, 'num_val_batches_mean')} / {unique_fmt(rows, 'num_test_batches_mean')}
- Batch size(s): {unique_fmt(rows, 'batch_size_mean')}
- Num atoms: mean {unique_fmt(rows, 'num_atoms_mean_mean')}, max {unique_fmt(rows, 'num_atoms_max_mean')}
- Split type(s): {', '.join(split_types) if split_types else 'not recorded'}
- Force unit(s): {', '.join(units) if units else 'not recorded'}
- Energy unit(s): {', '.join(energy_units) if energy_units else 'not recorded'}

## Model Comparison

{table(rows)}

## Direct-Force vs Energy-Force

Rows with `training_mode=energy_force` use invariant scalar energy and forces from `-grad_pos E`. Compare force MAE/RMSE and runtime against direct-force rows before making any physical-consistency claim.

Energy-force rows should use centered or standardized energy targets when the raw energy offset is large. Uncentered raw-energy losses are hard to interpret.

## Baselines And Vector Metrics

Zero-force and train-mean-force baselines are computed on the same test split. Component-wise force MAE/RMSE remains useful for comparison, but it is axis-component based; vector L2 error is more geometrically natural for force-vector evaluation.

## Cutoff Sweep

Cutoff rows should be interpreted through force error, average neighbor count, graph build time, and runtime together. Larger cutoffs are not automatically better.

Current 1k-frame cutoff results are preliminary.

{cutoff_table(rows)}

## Parameter/Runtime Trade-off

Use the model comparison table to read parameter count and runtime beside force error. Small local runs are primarily smoke checks; repeat with more seeds and frames before drawing ranking conclusions.

## Learning Diagnostics

- Validation force MAE epoch 1: {unique_fmt(rows, 'val_force_mae_epoch1_mean')}
- Validation force MAE final: {unique_fmt(rows, 'val_force_mae_final_mean')}
- Validation force vector L2 MAE epoch 1: {unique_fmt(rows, 'val_force_vector_l2_mae_epoch1_mean')}
- Validation force vector L2 MAE final: {unique_fmt(rows, 'val_force_vector_l2_mae_final_mean')}
- Learning established: {unique_fmt(rows, 'learning_established_mean')}

## Claim Checklist

- Real rMD17 local molecular file used: {'yes' if rows and all(truthy(row.get('is_real_rmd17')) for row in rows) else 'no'}
- Fake/synthetic false: {'yes' if rows and not any(truthy(row.get('is_fake_or_synthetic')) for row in rows) else 'no'}
- At least 3 seeds: {'yes' if all(int(float(row.get('n', 0))) >= 3 for row in rows) else 'no'}
- Zero baseline included: {'yes' if baselines_present(rows)[0] else 'no'}
- Mean baseline included: {'yes' if baselines_present(rows)[1] else 'no'}
- EGNN/TFN/SE3 direct-force compared: {'yes' if direct_family_present(rows) else 'no'}
- Energy normalization used for energy-force mode: {'yes' if energy_normalization_ok(rows) else 'no'}
- Vector L2 metrics included: {'yes' if vector_metrics_present(rows) else 'no'}
- Cutoff sweep schema complete: {'yes' if cutoff_schema_complete(rows) else 'no'}
- Distinct model identity signatures: {'yes' if distinct_model_identities(rows) else 'no'}
- Models improve over zero/mean baselines: {'yes' if model_improves_over_baselines(rows) else 'no'}
- Comparable frame counts and batch sizes: {'yes' if comparable_frame_counts(rows) else 'no'}
- Equivariant mean error <= 1e-5: {'yes' if max_equiv <= 1e-5 else 'no'}
- No failure log: {'yes' if not (input_path.parent / 'failed_runs.csv').exists() else 'no'}
- Parameter/runtime reported: yes
- Architecture claim allowed: {'yes' if ranking_allowed else 'no'}

## Limitations

These runs are only as strong as the local data source, split policy, and seed count. Do not extrapolate to MD22/OC20-scale claims without larger benchmarks.
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
