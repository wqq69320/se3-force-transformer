#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


LOCAL_REQUIRED = {"egnn", "tfn"}
LOCAL_ONE_OF = {"se3", "painn_lite"}


def read_csv(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(row: dict, key: str, default: float | None = None) -> float | None:
    value = row.get(key)
    if value in (None, "", "nan"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def text_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def model_family(row: dict) -> str:
    config = str(row.get("config_name", "")).lower()
    model = str(row.get("model_name", "")).lower()
    backbone = str(row.get("backbone_class", "")).lower()
    if "global_context_radial" in config or model == "global_context_radial":
        if "_no_global" in config or not text_bool(row.get("uses_global_context", "")):
            return "global_context_radial_no_global"
        return "global_context_radial"
    if "global_coeff" in config or model == "global_coeff":
        return "global_coeff"
    if "radial_pair" in config or model == "radial_pair" or "radial_pair" in backbone:
        return "radial_pair"
    if "painn" in config or model == "painn_lite":
        return "painn_lite"
    if "egnn" in config or model == "egnn":
        return "egnn"
    if "tfn" in config or model == "tfn":
        return "tfn"
    if "se3" in config or model in {"se3", "se3_transformer"}:
        return "se3"
    return model or config or "unknown"


def split_rows(rows: list[dict], split: str) -> list[dict]:
    return [row for row in rows if str(row.get("split_type", "")).lower() == split]


def best_row(rows: list[dict]) -> dict | None:
    scored = [(number(row, "force_vector_l2_mae_mean"), row) for row in rows]
    scored = [(score, row) for score, row in scored if score is not None]
    return min(scored, key=lambda item: item[0])[1] if scored else None


def required_rows(rows: list[dict]) -> list[dict]:
    families = {model_family(row) for row in rows}
    required = set(LOCAL_REQUIRED)
    if "se3" in families:
        required.add("se3")
    elif "painn_lite" in families:
        required.add("painn_lite")
    required.update({"global_coeff"})
    return [row for row in rows if model_family(row) in required]


def budget_values(row: dict) -> tuple:
    return (
        row.get("learning_rate", ""),
        row.get("force_loss_type", ""),
        row.get("force_scale_normalization", ""),
        row.get("num_frames_used_mean", ""),
        row.get("num_train_frames_mean", ""),
        row.get("num_val_frames_mean", ""),
        row.get("num_test_frames_mean", ""),
        row.get("epochs_mean", ""),
        row.get("max_steps_per_epoch_mean", ""),
    )


def budgets_present(rows: list[dict]) -> bool:
    def has(row: dict, *keys: str) -> bool:
        return any(row.get(key) not in (None, "") for key in keys)

    return all(
        has(row, "learning_rate")
        and has(row, "force_loss_type")
        and has(row, "force_scale_normalization")
        and (
            has(row, "num_frames_used_mean")
            or (
                has(row, "num_train_frames_mean")
                and has(row, "num_val_frames_mean")
                and has(row, "num_test_frames_mean")
            )
        )
        and has(row, "batch_size_mean")
        and has(row, "epochs_mean")
        and has(row, "max_steps_per_epoch_mean")
        and has(row, "parameter_count_mean")
        and has(row, "runtime_per_batch_sec_mean")
        and has(row, "graph_mode")
        and has(row, "cutoff_radius_mean")
        and has(row, "average_neighbors_mean")
        for row in rows
    )


def failed_required(failed_rows: list[dict], rows: list[dict]) -> list[dict]:
    required_configs = {row.get("config_name") for row in required_rows(rows)}
    return [row for row in failed_rows if row.get("config_name") in required_configs]


def claim_gate(rows: list[dict], failed_rows: list[dict], split: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    families = {model_family(row) for row in rows}
    if not all(text_bool(row.get("is_real_rmd17", "")) for row in rows):
        reasons.append("real local rMD17 file used: no")
    if any(text_bool(row.get("is_fake_or_synthetic", "")) for row in rows):
        reasons.append("fake/synthetic false: no")
    if any(number(row, "n", 0) < 3 for row in rows):
        reasons.append("n >= 3 for compared rows: no")
    if not all(row.get("zero_force_vector_l2_mae_mean") not in (None, "") and row.get("mean_force_vector_l2_mae_mean") not in (None, "") for row in rows):
        reasons.append("zero/mean baselines included: no")
    if not LOCAL_REQUIRED.issubset(families):
        reasons.append("EGNN and TFN rows present: no")
    if not (families & LOCAL_ONE_OF):
        reasons.append("local SE3 or PaiNN-lite row present: no")
    if not ({"global_coeff", "global_context_radial"} & families):
        reasons.append("global row present: no")
    required_failed = failed_required(failed_rows, rows)
    if required_failed:
        reasons.append(f"failed required rows: {len(required_failed)}")
    if not budgets_present(required_rows(rows)):
        reasons.append("budget metadata present/listed: no")

    local = [row for row in rows if model_family(row) in {"egnn", "tfn", "se3", "radial_pair", "painn_lite", "global_context_radial_no_global"}]
    global_rows = [row for row in rows if model_family(row) in {"global_coeff", "global_context_radial"}]
    best_local = best_row(local)
    best_global = best_row(global_rows)
    if best_global is None or best_local is None:
        reasons.append("best local/global rows available: no")
    else:
        zero = number(best_global, "force_vector_l2_mae_improvement_vs_zero_pct_mean", -1e9)
        mean = number(best_global, "force_vector_l2_mae_improvement_vs_mean_pct_mean", -1e9)
        if zero is None or zero < 5 or mean is None or mean < 5:
            reasons.append("global beats zero/mean by >=5% vector-L2: no")
        local_mae = number(best_local, "force_vector_l2_mae_mean")
        global_mae = number(best_global, "force_vector_l2_mae_mean")
        margin = 5.0 if split == "random" else 1.0
        if local_mae is None or global_mae is None or local_mae <= 0:
            reasons.append("global vs local margin computable: no")
        elif ((local_mae - global_mae) / local_mae * 100.0) < margin:
            reasons.append(f"global beats best local by >= {margin:g}% vector-L2: no")
        if number(best_global, "force_equivariance_error_mean", 1e9) > 1e-5:
            reasons.append("equivariance error <= 1e-5: no")
    return (not reasons), reasons


def fmt(value, precision: int = 4) -> str:
    num = number({"x": value}, "x")
    if num is None:
        return ""
    return f"{num:.{precision}g}"


def comparison_table(rows: list[dict]) -> list[str]:
    lines = [
        "| model | config | n | frames | batch | vector-L2 | zero % | mean % | component std pred | equiv | params | runtime/batch | graph | cutoff | neighbors | epochs | max steps |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (model_family(r), r.get("config_name", ""))):
        lines.append(
            "| "
            + " | ".join(
                [
                    model_family(row),
                    Path(row.get("config_name", "")).stem,
                    fmt(row.get("n"), 0),
                    fmt(row.get("num_frames_used_mean"), 0),
                    fmt(row.get("batch_size_mean"), 0),
                    fmt(row.get("force_vector_l2_mae_mean")),
                    fmt(row.get("force_vector_l2_mae_improvement_vs_zero_pct_mean")),
                    fmt(row.get("force_vector_l2_mae_improvement_vs_mean_pct_mean")),
                    fmt(row.get("force_component_std_pred_mean")),
                    fmt(row.get("force_equivariance_error_mean")),
                    fmt(row.get("parameter_count_mean"), 0),
                    fmt(row.get("runtime_per_batch_sec_mean")),
                    str(row.get("graph_mode", "")),
                    fmt(row.get("cutoff_radius_mean")),
                    fmt(row.get("average_neighbors_mean")),
                    fmt(row.get("epochs_mean"), 0),
                    fmt(row.get("max_steps_per_epoch_mean"), 0),
                ]
            )
            + " |"
        )
    return lines


def gate_lines(name: str, rows: list[dict], failed: list[dict], split: str) -> list[str]:
    passed, reasons = claim_gate(rows, failed, split)
    lines = [f"## {name} Claim Gate", "", f"- claim gate passed: {'yes' if passed else 'no'}"]
    if budgets_present(required_rows(rows)):
        lines.append("- budget metadata present/listed: yes")
    if reasons:
        lines.extend(f"- {reason}" for reason in reasons)
    else:
        lines.append("- all required gate checks passed")
    return lines


def conclusion(random_rows: list[dict], chrono_rows: list[dict], failed: list[dict]) -> str:
    random_ok, _ = claim_gate(random_rows, failed, "random") if random_rows else (False, [])
    chrono_ok, _ = claim_gate(chrono_rows, failed, "chronological") if chrono_rows else (False, [])
    if random_ok and chrono_ok:
        return "Global-context advantage is established for this matched rMD17 aspirin run, within the reported budget limits."
    if random_ok:
        return "Global-context advantage is established on random split only; chronological evidence is incomplete or weaker."
    return "Global-context advantage is not established by this run."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-summary")
    parser.add_argument("--random-per-run")
    parser.add_argument("--chrono-summary")
    parser.add_argument("--chrono-per-run")
    parser.add_argument("--ablation-summary")
    parser.add_argument("--ablation-per-run")
    parser.add_argument("--random-failed")
    parser.add_argument("--chrono-failed")
    parser.add_argument("--ablation-failed")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    random_rows = read_csv(args.random_summary)
    chrono_rows = read_csv(args.chrono_summary)
    ablation_rows = read_csv(args.ablation_summary)
    failed_rows = read_csv(args.random_failed) + read_csv(args.chrono_failed) + read_csv(args.ablation_failed)

    lines = [
        "# Phase 13 Global vs Local Report",
        "",
        "This report compares global invariant context against local equivariant baselines on local rMD17 aspirin. It does not claim SOTA or transferable chemistry.",
        "",
        "## Interpretation",
        "",
        f"- {conclusion(random_rows, chrono_rows, failed_rows)}",
        "",
        "## Phase 12 Recap",
        "",
        "- Phase 12 found global_coeff beat zero/mean baselines on 1k random and chronological splits.",
        "- Phase 12 did not include matched local baselines, so architecture superiority was not established.",
        "",
        "## Random Split Comparison",
        "",
        *comparison_table(random_rows),
        "",
        "## Chronological Split Comparison",
        "",
        *comparison_table(chrono_rows),
        "",
        "## Local vs Global Context",
        "",
        *comparison_table([row for row in random_rows + chrono_rows if model_family(row) in {"radial_pair", "global_context_radial_no_global", "global_context_radial", "global_coeff"}]),
        "",
    ]
    if ablation_rows:
        lines.extend(["## Ablation Table", "", *comparison_table(ablation_rows), ""])
    lines.extend(gate_lines("Random Split", random_rows, failed_rows, "random"))
    lines.extend(["", *gate_lines("Chronological Split", chrono_rows, failed_rows, "chronological"), ""])
    if failed_rows:
        lines.extend(["## Failed Runs", "", f"- failed rows: {len(failed_rows)}"])
    lines.extend(
        [
            "## Guardrail",
            "",
            "- Do not claim SOTA, transferable chemistry, or architecture superiority unless the claim gates pass.",
            "- Budget differences are listed in the comparison tables via frames, batch size, epochs, max steps, graph mode, neighbors, parameters, and runtime.",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
