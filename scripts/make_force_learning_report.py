#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

BASELINE_IMPROVEMENT_MARGIN_PCT = 1.0


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: str | float | None) -> str:
    if value in (None, "", "nan"):
        return "n/a"
    value_f = float(value)
    return f"{value_f:.4e}" if abs(value_f) < 1e-3 or abs(value_f) > 1e4 else f"{value_f:.4g}"


def truthy(value: str | None) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def unique(rows: list[dict], field: str) -> str:
    values = sorted({str(row.get(field)) for row in rows if row.get(field) not in (None, "")})
    return ", ".join(values) if values else "not recorded"


def field_present(rows: list[dict], field: str) -> bool:
    return bool(rows) and field in rows[0] and any(row.get(field) not in (None, "", "nan") for row in rows)


def direct_families_present(rows: list[dict]) -> bool:
    names = " ".join(row.get("config_name", "").lower() for row in rows if row.get("training_mode") == "direct_force")
    return all(token in names for token in ("egnn", "tfn", "se3"))


def comparable_budgets(rows: list[dict]) -> bool:
    for field in ["num_train_frames_mean", "num_val_frames_mean", "num_test_frames_mean", "batch_size_mean"]:
        values = {row.get(field) for row in rows if row.get(field) not in (None, "")}
        if len(values) > 1:
            return False
    return True


def improvement_pct(model: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0
    return 100.0 * (baseline - model) / baseline


def row_beats_baselines(row: dict, margin_pct: float = BASELINE_IMPROVEMENT_MARGIN_PCT) -> bool:
    try:
        force_mae = float(row["force_mae_mean"])
        zero_force_mae = float(row["zero_force_mae_mean"])
        mean_force_mae = float(row["mean_force_mae_mean"])
        vector_l2 = float(row["force_vector_l2_mae_mean"])
        zero_vector_l2 = float(row["zero_force_vector_l2_mae_mean"])
        mean_vector_l2 = float(row["mean_force_vector_l2_mae_mean"])
    except (KeyError, TypeError, ValueError):
        return False

    component_vs_zero = improvement_pct(force_mae, zero_force_mae)
    component_vs_mean = improvement_pct(force_mae, mean_force_mae)
    vector_vs_zero = improvement_pct(vector_l2, zero_vector_l2)
    vector_vs_mean = improvement_pct(vector_l2, mean_vector_l2)
    return min(component_vs_zero, component_vs_mean, vector_vs_zero, vector_vs_mean) >= margin_pct


def any_model_beats_baselines(rows: list[dict]) -> bool:
    return any(row_beats_baselines(row) for row in rows)


def claim_gate(rows: list[dict], output_dir: Path) -> tuple[bool, list[str]]:
    reasons = []
    if not rows or not all(truthy(row.get("is_real_rmd17")) for row in rows):
        reasons.append("local real rMD17 data is not confirmed")
    if any(truthy(row.get("is_fake_or_synthetic")) for row in rows):
        reasons.append("fake or synthetic data present")
    if not rows or not all(int(float(row.get("n", 0))) >= 3 for row in rows):
        reasons.append("fewer than 3 seeds")
    if not field_present(rows, "zero_force_mae_mean"):
        reasons.append("zero baseline missing")
    if not field_present(rows, "mean_force_mae_mean"):
        reasons.append("mean baseline missing")
    if not field_present(rows, "zero_force_vector_l2_mae_mean"):
        reasons.append("zero vector-L2 baseline missing")
    if not field_present(rows, "mean_force_vector_l2_mae_mean"):
        reasons.append("mean vector-L2 baseline missing")
    if not any_model_beats_baselines(rows):
        reasons.append(f"no model beats zero and mean baselines by >= {BASELINE_IMPROVEMENT_MARGIN_PCT:.1f}% on component and vector-L2 MAE")
    if not direct_families_present(rows):
        reasons.append("EGNN, TFN, and SE3 direct-force models are not all present")
    if (output_dir / "failed_runs.csv").exists():
        reasons.append("failure log exists")
    if not comparable_budgets(rows):
        reasons.append("train budgets are not comparable")
    try:
        max_equiv = max(float(row.get("equivariance_error_mean", "nan")) for row in rows)
        if max_equiv > 1e-5:
            reasons.append("equivariance error exceeds 1e-5")
    except ValueError:
        reasons.append("equivariance metrics missing")
    return not reasons, reasons


def best_row(rows: list[dict]) -> dict:
    def key(row: dict) -> float:
        value = row.get("val_force_vector_l2_mae_final_mean") or row.get("force_vector_l2_mae_mean") or "inf"
        try:
            return float(value)
        except ValueError:
            return float("inf")

    return min(rows, key=key) if rows else {}


def model_table(rows: list[dict]) -> str:
    lines = [
        "| config | lr | loss | norm | n | val vector L2 | test vector L2 | force MAE | zero MAE | mean MAE | improvement vs zero | equivariance | runtime |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("config_name", "").replace(".yaml", ""),
                    fmt(row.get("learning_rate")),
                    row.get("force_loss_type", ""),
                    row.get("force_scale_normalization", ""),
                    row.get("n", ""),
                    fmt(row.get("val_force_vector_l2_mae_final_mean")),
                    fmt(row.get("force_vector_l2_mae_mean")),
                    fmt(row.get("force_mae_mean")),
                    fmt(row.get("zero_force_mae_mean")),
                    fmt(row.get("mean_force_mae_mean")),
                    fmt(row.get("force_mae_improvement_vs_zero_pct_mean")),
                    fmt(row.get("equivariance_error_mean")),
                    fmt(row.get("runtime_per_batch_sec_mean")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--per-run", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = read_rows(input_path)
    per_rows = read_rows(Path(args.per_run)) if args.per_run else []
    gate_ok, blockers = claim_gate(rows, input_path.parent)
    best = best_row(rows)
    text = f"""# rMD17 Force Learning Report

This report focuses on whether direct-force molecular models learn beyond trivial force baselines. It does not claim SOTA performance or transferable chemistry.

## Dataset And Split

- Dataset: {unique(rows, 'dataset_name')}
- Molecule: {unique(rows, 'molecule_name')}
- Data source type: {unique(rows, 'data_source_type')}
- Real rMD17: {unique(rows, 'is_real_rmd17')}
- Fake/synthetic: {unique(rows, 'is_fake_or_synthetic')}
- Train/val/test frames: {unique(rows, 'num_train_frames_mean')} / {unique(rows, 'num_val_frames_mean')} / {unique(rows, 'num_test_frames_mean')}
- Batch size: {unique(rows, 'batch_size_mean')}

## Baselines

Zero-force and train-mean-force baselines are evaluated on the same test split as each model.

## Model Results

{model_table(rows)}

## Winning Config

- Best by validation vector-L2 force MAE: {best.get('config_name', 'n/a')} at lr={best.get('learning_rate', 'n/a')}, loss={best.get('force_loss_type', 'n/a')}, normalization={best.get('force_scale_normalization', 'n/a')}
- Validation vector-L2 force MAE: {fmt(best.get('val_force_vector_l2_mae_final_mean'))}
- Test vector-L2 force MAE: {fmt(best.get('force_vector_l2_mae_mean'))}

## Learning Curves

- Per-run rows: {len(per_rows)}
- Training curve paths are recorded in `training_curve` in the per-run metrics CSV.

## Claim Gate

- Architecture comparison allowed: {'yes' if gate_ok else 'no'}
- Baseline improvement margin required: >= {BASELINE_IMPROVEMENT_MARGIN_PCT:.1f}% on component and vector-L2 MAE
- Gate blockers: {', '.join(blockers) if blockers else 'none'}

{'Architecture superiority is not established by this run.' if not gate_ok else 'Architecture comparison gates passed; interpret any differences cautiously.'}

## Checklist

- Local real rMD17 file used: {'yes' if rows and all(truthy(row.get('is_real_rmd17')) for row in rows) else 'no'}
- Fake/synthetic false: {'yes' if rows and not any(truthy(row.get('is_fake_or_synthetic')) for row in rows) else 'no'}
- At least 3 seeds: {'yes' if rows and all(int(float(row.get('n', 0))) >= 3 for row in rows) else 'no'}
- Zero baseline included: {'yes' if field_present(rows, 'zero_force_mae_mean') else 'no'}
- Mean baseline included: {'yes' if field_present(rows, 'mean_force_mae_mean') else 'no'}
- Vector-L2 baselines included: {'yes' if field_present(rows, 'zero_force_vector_l2_mae_mean') and field_present(rows, 'mean_force_vector_l2_mae_mean') else 'no'}
- At least one model beats both baselines by margin: {'yes' if any_model_beats_baselines(rows) else 'no'}
- EGNN/TFN/SE3 direct-force present: {'yes' if direct_families_present(rows) else 'no'}
- Comparable train budgets: {'yes' if comparable_budgets(rows) else 'no'}
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
