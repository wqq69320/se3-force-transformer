#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


OVERFIT_64_MARGIN = 50.0
OVERFIT_128_MARGIN = 30.0
NORM_RATIO_MARGIN = 0.75
COSINE_MARGIN = 0.7
RANDOM_MARGIN = 5.0
CHRONO_SIGNAL_MARGIN = 1.0
EQUIV_MARGIN = 1e-5


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(row: dict | None, field: str):
    if not row:
        return None
    for key in [field, f"{field}_mean"]:
        value = row.get(key)
        if value in (None, "", "nan"):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def text(row: dict, field: str) -> str:
    return str(row.get(field, row.get(f"{field}_mean", "")))


def fmt(value) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.4g}"


def model_family(row: dict) -> str:
    blob = " ".join(
        str(row.get(field, ""))
        for field in ["run_name", "config_name", "model_name", "model_class", "backbone_class", "architecture_signature"]
    ).lower()
    if "global_context_radial" in blob:
        return "global_context_radial"
    if "global_context_painn" in blob:
        return "global_context_painn"
    if "global_context_se3" in blob:
        return "global_context_se3"
    if "global_coeff" in blob or "global_invariant" in blob:
        return "global_coeff"
    return text(row, "model_name") or "unknown"


def frame_count(row: dict) -> int:
    name = str(row.get("config_name", ""))
    if "1k" in name:
        return 1000
    for field in ["num_frames_used", "num_frames_used_mean", "num_train_frames", "num_train_frames_mean", "num_train_samples", "num_train_samples_mean"]:
        value = number(row, field)
        if value is not None:
            return int(round(value))
    for marker in ["overfit128", "overfit64", "overfit32", "1k"]:
        if marker in name:
            return 1000 if marker == "1k" else int(marker.replace("overfit", ""))
    return 0


def is_overfit(row: dict) -> bool:
    return "overfit" in text(row, "split_type") or "overfit" in str(row.get("config_name", "")).lower()


def split_kind(row: dict) -> str:
    split = text(row, "split_type").lower()
    if "chrono" in split:
        return "chronological"
    if "random" in split:
        return "random"
    if is_overfit(row):
        return "overfit"
    return split or "unknown"


def overfit_success(row: dict) -> bool:
    frames = frame_count(row)
    margin = OVERFIT_128_MARGIN if frames >= 128 else OVERFIT_64_MARGIN
    improvement = number(row, "train_eval_force_vector_l2_mae_improvement_vs_zero_pct")
    ratio = number(row, "train_eval_pred_to_target_force_norm_ratio")
    cosine = number(row, "train_eval_force_cosine_similarity_mean")
    equiv = number(row, "force_equivariance_error")
    if equiv is None:
        equiv = number(row, "equivariance_error")
    return (
        improvement is not None
        and improvement >= margin
        and ratio is not None
        and ratio >= NORM_RATIO_MARGIN
        and cosine is not None
        and cosine >= COSINE_MARGIN
        and equiv is not None
        and equiv <= EQUIV_MARGIN
    )


def generalization_success(row: dict, margin: float) -> bool:
    zero = number(row, "force_vector_l2_mae_improvement_vs_zero_pct")
    mean = number(row, "force_vector_l2_mae_improvement_vs_mean_pct")
    equiv = number(row, "force_equivariance_error")
    if equiv is None:
        equiv = number(row, "equivariance_error")
    return (
        zero is not None
        and mean is not None
        and min(zero, mean) >= margin
        and equiv is not None
        and equiv <= EQUIV_MARGIN
    )


def best_row(rows: list[dict], metric: str, lower: bool = True) -> dict | None:
    scored = [(number(row, metric), row) for row in rows]
    scored = [(score, row) for score, row in scored if score is not None]
    if not scored:
        return rows[0] if rows else None
    return (min if lower else max)(scored, key=lambda item: item[0])[1]


def label(row: dict | None) -> str:
    if not row:
        return "n/a"
    name = str(row.get("config_name", row.get("run_name", "run"))).replace(".yaml", "")
    lr = row.get("learning_rate", row.get("learning_rate_mean", ""))
    loss = row.get("force_loss_type", "")
    family = model_family(row)
    return f"{family}: {name} lr={lr} {loss}".strip()


def table(rows: list[dict], title: str, train: bool = False) -> str:
    lines = [
        f"## {title}",
        "",
        "| frames | split | model | vector-L2 | zero improv % | mean improv % | norm ratio | cosine | equivariance | success |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    if not rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        return "\n".join(lines) + "\n"
    for row in sorted(rows, key=lambda item: (frame_count(item), model_family(item), label(item))):
        if train:
            vec = number(row, "train_eval_force_vector_l2_mae")
            zero = number(row, "train_eval_force_vector_l2_mae_improvement_vs_zero_pct")
            mean = number(row, "train_eval_force_vector_l2_mae_improvement_vs_mean_pct")
            ratio = number(row, "train_eval_pred_to_target_force_norm_ratio")
            cosine = number(row, "train_eval_force_cosine_similarity_mean")
            success = overfit_success(row)
        else:
            vec = number(row, "force_vector_l2_mae")
            zero = number(row, "force_vector_l2_mae_improvement_vs_zero_pct")
            mean = number(row, "force_vector_l2_mae_improvement_vs_mean_pct")
            ratio = number(row, "pred_to_target_force_norm_ratio")
            cosine = number(row, "force_cosine_similarity_mean")
            margin = RANDOM_MARGIN if split_kind(row) == "random" else CHRONO_SIGNAL_MARGIN
            success = generalization_success(row, margin)
        equiv = number(row, "force_equivariance_error")
        if equiv is None:
            equiv = number(row, "equivariance_error")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(frame_count(row)),
                    split_kind(row),
                    model_family(row),
                    fmt(vec),
                    fmt(zero),
                    fmt(mean),
                    fmt(ratio),
                    fmt(cosine),
                    fmt(equiv),
                    "yes" if success else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def claim_gate(rows: list[dict]) -> list[str]:
    gates = {
        "real local rMD17 file used": any(str(row.get("data_source_type", "")) == "local_rmd17_npz" for row in rows),
        "n >= 3": any((number(row, "n") or 0) >= 3 for row in rows),
        "zero/mean baselines included": all(
            number(row, "force_vector_l2_mae_improvement_vs_zero_pct") is not None
            and number(row, "force_vector_l2_mae_improvement_vs_mean_pct") is not None
            for row in rows
        )
        if rows
        else False,
        "vector-L2 metrics included": all(number(row, "force_vector_l2_mae") is not None for row in rows) if rows else False,
        "equivariance <= 1e-5": all((number(row, "force_equivariance_error") or number(row, "equivariance_error") or 1.0) <= EQUIV_MARGIN for row in rows)
        if rows
        else False,
        "comparable local baselines included": any(model_family(row) not in {"global_coeff", "global_context_radial"} for row in rows),
    }
    return [f"- {name}: {'yes' if ok else 'no'}" for name, ok in gates.items()]


def conclusion(overfit_rows: list[dict], random_rows: list[dict], chrono_rows: list[dict]) -> str:
    overfit64_ok = any(model_family(row) == "global_coeff" and frame_count(row) >= 64 and frame_count(row) < 128 and overfit_success(row) for row in overfit_rows)
    overfit128_ok = any(model_family(row) == "global_coeff" and frame_count(row) >= 128 and overfit_success(row) for row in overfit_rows)
    random_ok = any(model_family(row) == "global_coeff" and generalization_success(row, RANDOM_MARGIN) for row in random_rows)
    chrono_ok = any(model_family(row) == "global_coeff" and generalization_success(row, CHRONO_SIGNAL_MARGIN) for row in chrono_rows)
    context_ok = any(model_family(row) == "global_context_radial" and (overfit_success(row) if is_overfit(row) else generalization_success(row, CHRONO_SIGNAL_MARGIN)) for row in overfit_rows + random_rows + chrono_rows)
    global_scales = overfit64_ok or overfit128_ok
    if chrono_ok:
        return "Global invariant context gives a viable rMD17 learning path; next compare against local baselines."
    if random_ok and chrono_rows and not chrono_ok:
        return "The primary bottleneck is trajectory distribution shift."
    if random_ok and not chrono_rows:
        return "Global invariant coefficients beat zero/mean baselines on 1k random split; chronological split is not yet tested."
    if global_scales and not random_ok:
        return "The global equivariant memorizer has capacity, but generalization remains weak."
    if context_ok and not global_scales:
        return "Global context in a local equivariant architecture may be the useful compromise."
    if overfit_rows and not global_scales and not random_ok and not chrono_ok:
        return "The successful 32-frame result may be memorization-specific; more scalable equivariant capacity is needed."
    return "Phase 12 evidence is incomplete or mixed; continue controlled scaling diagnostics before making a broader claim."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--per-run", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    per_rows = read_rows(Path(args.per_run)) if args.per_run else []
    overfit_rows = [row for row in rows if split_kind(row) == "overfit"]
    random_rows = [row for row in rows if split_kind(row) == "random"]
    chrono_rows = [row for row in rows if split_kind(row) == "chronological"]
    best_global = best_row([row for row in rows if model_family(row) == "global_coeff"], "force_vector_l2_mae")
    best_context = best_row([row for row in rows if model_family(row) == "global_context_radial"], "force_vector_l2_mae")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# Phase 12 Global Context Report",
            "",
            "This report extends the Phase 11 global invariant coefficient overfit diagnostic into controlled rMD17 aspirin scaling checks. It does not claim architecture superiority.",
            "",
            "## Phase 11 Recap",
            "",
            "- GlobalInvariantCoefficientForceModel overfit 32 aspirin frames across seeds 0, 1, and 2.",
            "- Phase 11 mean train vector-L2 MAE was 8.9371 with 78.62% improvement vs zero baseline.",
            "- Phase 11 mean equivariance error was 1.1055e-06.",
            "",
            "## Interpretation",
            "",
            f"- {conclusion(overfit_rows, random_rows, chrono_rows)}",
            "",
            "## Best Rows",
            "",
            f"- Best global_coeff row: {label(best_global)}",
            f"- Best global_context_radial row: {label(best_context)}",
            "",
            table(overfit_rows, "Overfit Scaling", train=True),
            table(random_rows, "1k Random Split"),
            table(chrono_rows, "1k Chronological Split"),
            "## Claim Gates",
            "",
            *claim_gate(rows),
            "",
            "## Per-Run Curves",
            "",
            f"- Per-run rows: {len(per_rows)}",
            "- Training curve paths are recorded in `training_curve` in the per-run CSV.",
            "",
            "## Guardrail",
            "",
            "Do not claim SOTA, transferable chemistry, or architecture superiority from this report without comparable local baselines under matched budgets.",
            "",
        ]
    )
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
