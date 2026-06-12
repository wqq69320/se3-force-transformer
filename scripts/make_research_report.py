#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from se3force.evaluation.display import config_stem, display_name_for_config


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: str, precision: int = 4) -> str:
    if value == "":
        return ""
    number = float(value)
    if number == 0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e4:
        return f"{number:.{precision}e}"
    return f"{number:.{precision}g}"


def row_label(row: dict) -> str:
    return row.get("display_name") or display_name_for_config(row["config_name"])


def row_stems(rows: list[dict]) -> set[str]:
    return {config_stem(row["config_name"]) for row in rows}


def lmax_values(rows: list[dict]) -> set[int]:
    values = set()
    for row in rows:
        stem = config_stem(row["config_name"])
        if not stem.startswith("ablation_lmax"):
            continue
        suffix = stem.replace("ablation_lmax", "")
        if suffix.isdigit():
            values.add(int(suffix))
    return values


def min_seed_count(rows: list[dict]) -> int:
    return min(int(float(row.get("n", 0))) for row in rows)


def claim_readiness(rows: list[dict]) -> dict[str, bool]:
    stems = row_stems(rows)
    lmax = lmax_values(rows)
    return {
        "non_equivariant_baselines": bool({"baseline_mlp", "baseline_vanilla_gt"} & stems),
        "egnn_and_tfn_baselines": {"baseline_egnn", "baseline_tfn"}.issubset(stems),
        "lmax_0_1_2": {0, 1, 2}.issubset(lmax),
        "at_least_3_seeds": min_seed_count(rows) >= 3,
    }


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def markdown_table(rows: list[dict]) -> str:
    headers = [
        "config",
        "model",
        "n",
        "canonical MSE",
        "rotated MSE",
        "equivariance error",
        "params",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row_label(row),
                    row.get("model_name", ""),
                    row["n"],
                    f"{fmt(row['canonical_mse_mean'])} +/- {fmt(row['canonical_mse_std'])}",
                    f"{fmt(row['rotated_translated_mse_mean'])} +/- {fmt(row['rotated_translated_mse_std'])}",
                    f"{fmt(row['equivariance_error_mean'])} +/- {fmt(row['equivariance_error_std'])}",
                    fmt(row["parameter_count_mean"], precision=3),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def equivariance_statement(rows: list[dict]) -> str:
    non_equivariant = {"baseline_mlp", "baseline_vanilla_gt"}
    equivariant_rows = [row for row in rows if config_stem(row["config_name"]) not in non_equivariant]
    nonequiv_rows = [row for row in rows if config_stem(row["config_name"]) in non_equivariant]
    worst_equiv = max(float(row["equivariance_error_mean"]) for row in equivariant_rows) if equivariant_rows else None
    worst_nonequiv = max(float(row["equivariance_error_mean"]) for row in nonequiv_rows) if nonequiv_rows else None

    if worst_equiv is None:
        return "No explicitly equivariant model rows are present, so equivariant-model symmetry correctness cannot be assessed from this summary."
    if worst_equiv <= 1e-5:
        statement = f"Equivariant model rows have mean equivariance error <= 1e-5; the largest equivariant-model mean is {worst_equiv:.3e}."
    else:
        statement = (
            f"At least one equivariant model row has mean equivariance error above 1e-5; "
            f"the largest equivariant-model mean is {worst_equiv:.3e}. Investigate before making symmetry claims."
        )
    if worst_nonequiv is not None:
        statement += (
            f" Non-equivariant baselines are reported separately and may have large errors by design; "
            f"their largest mean here is {worst_nonequiv:.3e}."
        )
    return statement


def best_by_metric(rows: list[dict], metric: str) -> dict:
    return min(rows, key=lambda row: float(row[f"{metric}_mean"]))


def make_report(rows: list[dict]) -> str:
    if not rows:
        raise ValueError("cannot build report from an empty summary")

    best_canonical = best_by_metric(rows, "canonical_mse")
    best_rotated = best_by_metric(rows, "rotated_translated_mse")
    best_equiv = best_by_metric(rows, "equivariance_error")
    readiness = claim_readiness(rows)
    seed_counts = sorted({row.get("n", "") for row in rows if row.get("n", "") != ""})
    lmax = sorted(lmax_values(rows))
    can_claim_architecture_advantage = all(readiness.values())
    fastest = best_by_metric(rows, "runtime_per_batch_sec")
    smallest = best_by_metric(rows, "parameter_count")

    return f"""# SE3-ForceTransformer Benchmark Report

This report is generated from benchmark summary metrics. It is evidence for the runs listed below, not a claim about broad molecular-force-field performance.

## Summary Table

{markdown_table(rows)}

## Implementation Correctness

The repository includes executable training, evaluation, benchmark, plotting, and report scripts. Passing this benchmark report generation shows that the configured runs produced schema-complete metrics files and aggregate CSV summaries. This does not by itself prove numerical optimality.

## Equivariance Correctness

{equivariance_statement(rows)}

Equivariance correctness should be interpreted together with the unit tests for rotations, spherical harmonics, TFN convolution, SE(3) attention, and the full force head. Do not loosen those tests to make benchmark results pass.

## Prediction Performance

The lowest mean canonical MSE in this summary is `{fmt(best_canonical['canonical_mse_mean'])}` from `{row_label(best_canonical)}`. The lowest mean rotated-translated MSE is `{fmt(best_rotated['rotated_translated_mse_mean'])}` from `{row_label(best_rotated)}`.

These are small synthetic datasets and short benchmark runs unless the input CSV came from a larger sweep. Treat the values as validation evidence for the configured experiment scale, not as final research claims.

## Baseline Comparison And Architecture Advantage

Architecture-advantage claims are gated by the following evidence:

| requirement | satisfied |
|---|---|
| non-equivariant baselines included | {yes_no(readiness['non_equivariant_baselines'])} |
| EGNN and TFN baselines included | {yes_no(readiness['egnn_and_tfn_baselines'])} |
| lmax ablation includes at least 0, 1, 2 | {yes_no(readiness['lmax_0_1_2'])} |
| at least 3 seeds per summarized claim | {yes_no(readiness['at_least_3_seeds'])} |

Architecture advantage status: {'the benchmark has the minimum comparison structure needed for a cautious claim, but effect sizes and uncertainty should still be inspected before wording any conclusion.' if can_claim_architecture_advantage else 'not established by this summary. The report must be read as benchmark evidence, not as proof that SE3 is superior.'}

The strongest equivariance result in this summary is `{fmt(best_equiv['equivariance_error_mean'])}` from `{row_label(best_equiv)}`. Baseline comparison is meaningful only when the summary contains multiple model families trained with comparable dataset sizes, seeds, and budgets.

## Parameter And Runtime Trade-Off

The smallest mean parameter count is `{fmt(smallest['parameter_count_mean'], precision=3)}` from `{row_label(smallest)}`. The fastest mean runtime per batch is `{fmt(fastest['runtime_per_batch_sec_mean'])}` seconds from `{row_label(fastest)}`.

Interpret lower prediction error together with parameter count and runtime. A model with better MSE but substantially larger size or cost should be described as a trade-off, not a strict win.

## Ablation Evidence

lmax values present from `ablation_lmax*` configs: {', '.join(str(value) for value in lmax) if lmax else 'none'}.

Ablation completeness: {'complete enough for a first lmax trend check.' if readiness['lmax_0_1_2'] else 'incomplete. Include at least ablation_lmax0, ablation_lmax1, and ablation_lmax2 before describing lmax trends.'} Attention and gate ablation evidence requires both `ablation_no_attention` and `ablation_no_gate` rows.

## Reproducibility Notes

- Number of summary rows: {len(rows)}
- Seed counts represented in grouped rows: {', '.join(seed_counts) if seed_counts else 'not recorded'}
- Metrics are mean +/- sample standard deviation across completed runs in each group.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/benchmark_suite/research_report.md")
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    report = make_report(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
