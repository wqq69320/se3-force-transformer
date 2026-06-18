#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

OVERFIT_IMPROVEMENT_MARGIN_PCT = 50.0
GENERALIZATION_IMPROVEMENT_MARGIN_PCT = 1.0

OVERFIT_FAIL_STATEMENT = "The current training setup cannot even overfit a tiny rMD17 subset; architecture comparison is invalid."
LOCAL_LEARN_CHRONO_WEAK_STATEMENT = "The model can learn the force field locally, but chronological generalization remains weak."
RANDOM_SHIFT_STATEMENT = "The primary bottleneck is trajectory distribution shift."
BOTH_GENERALIZATION_FAIL_STATEMENT = "The bottleneck is likely optimization, target scaling, or model capacity."
MLP_FAIL_STATEMENT = "The bottleneck is likely data/loss/training-loop scale or label handling."
MLP_ONLY_STATEMENT = "The bottleneck is likely equivariant architecture capacity/optimization."
RADIAL_ONLY_STATEMENT = "The bottleneck is likely SE3 wrapper/training hyperparameters."
TINY_OVERFIT_SOLVED_STATEMENT = "Tiny overfit is solved; proceed to 1k random-vs-chronological and then 1k seed-3 learning."
SE3_WRAPPER_BOTTLENECK_STATEMENT = "SE3 wrapper/optimization remains the main bottleneck."
EQUIVARIANT_CAPACITY_STATEMENT = "Equivariant force parameterization/capacity remains the main bottleneck."
FULL_IRREP_SOLVED_STATEMENT = "Tiny equivariant overfit is solved; proceed to 1k random-vs-chronological diagnostics."
CUTOFF_BOTTLENECK_STATEMENT = "Cutoff graph is a likely bottleneck."
FORCE_SCALE_BOTTLENECK_STATEMENT = "Force magnitude scaling was a key bottleneck."
GRADIENT_PATH_BOTTLENECK_STATEMENT = "Gradient path to force head/backbone is broken or too weak."
GLOBAL_COEFF_SOLVED_STATEMENT = "Equivariant overfit is possible with global invariant coefficients; local message-passing capacity is the likely bottleneck."
INTERNAL_ENERGY_SOLVED_STATEMENT = "Conservative equivariant overfit is possible; direct force-head parameterization may be the bottleneck."
PAINN_LITE_SOLVED_STATEMENT = "Vector message passing can overfit; current SE3/radial wrapper is the bottleneck."
PHASE11_EQUIVARIANT_FAIL_STATEMENT = "Current equivariant parameterizations remain too weak or optimization remains unresolved."


def read_rows(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def number(row: dict, field: str) -> float | None:
    for name in (field, f"{field}_mean"):
        value = _float(row.get(name))
        if value is not None:
            return value
    return None


def fmt(value) -> str:
    value_f = _float(value)
    if value_f is None:
        return "n/a"
    return f"{value_f:.4e}" if abs(value_f) < 1e-3 or abs(value_f) > 1e4 else f"{value_f:.4g}"


def label(row: dict) -> str:
    name = (row.get("run_name") or row.get("config_name", "row")).replace(".yaml", "")
    family = model_family(row)
    if family != "unknown" and family not in name.lower():
        name = f"{family}: {name}"
    lr = row.get("learning_rate", "")
    loss = row.get("force_loss_type", "")
    clip = row.get("gradient_clip_norm", "")
    head = row.get("output_head_init_scale", "")
    parts = [name]
    if lr not in (None, ""):
        parts.append(f"lr={lr}")
    if loss not in (None, ""):
        parts.append(str(loss))
    if clip not in (None, ""):
        parts.append(f"gc={clip}")
    if head not in (None, ""):
        parts.append(f"head={head}")
    return " ".join(parts)


def best_row(rows: list[dict], field: str, lower_is_better: bool = True) -> dict:
    candidates = [(number(row, field), row) for row in rows]
    candidates = [(value, row) for value, row in candidates if value is not None]
    if not candidates:
        return rows[0] if rows else {}
    return min(candidates, key=lambda item: item[0])[1] if lower_is_better else max(candidates, key=lambda item: item[0])[1]


def overfit_succeeded(rows: list[dict], margin_pct: float = OVERFIT_IMPROVEMENT_MARGIN_PCT) -> bool:
    for row in rows:
        improvement = number(row, "train_eval_force_vector_l2_mae_improvement_vs_zero_pct")
        if improvement is None:
            improvement = number(row, "force_vector_l2_mae_improvement_vs_zero_pct")
        if improvement is not None and improvement >= margin_pct:
            return True
    return False


def row_meets_success_criteria(row: dict) -> bool:
    improvement = number(row, "train_eval_force_vector_l2_mae_improvement_vs_zero_pct")
    if improvement is None:
        improvement = number(row, "force_vector_l2_mae_improvement_vs_zero_pct")
    norm_ratio = number(row, "train_eval_pred_to_target_force_norm_ratio")
    if norm_ratio is None:
        norm_ratio = number(row, "pred_to_target_force_norm_ratio")
    cosine = number(row, "train_eval_force_cosine_similarity_mean")
    if cosine is None:
        cosine = number(row, "force_cosine_similarity_mean")
    return (
        improvement is not None
        and improvement >= OVERFIT_IMPROVEMENT_MARGIN_PCT
        and norm_ratio is not None
        and norm_ratio >= 0.8
        and cosine is not None
        and cosine >= 0.7
    )


def any_success(rows: list[dict], families: set[str] | None = None) -> bool:
    return any(row_meets_success_criteria(row) and (families is None or model_family(row) in families) for row in rows)


def model_family(row: dict) -> str:
    text = " ".join(
        str(row.get(field, ""))
        for field in [
            "run_name",
            "config_name",
            "model_name",
            "model_class",
            "backbone_class",
            "architecture_signature",
            "actual_hidden_irreps",
            "uses_non_scalar_hidden",
        ]
    ).lower()
    if "global_context_radial" in text:
        return "global_context_radial"
    if "global_context_painn" in text:
        return "global_context_painn"
    if "global_context_se3" in text:
        return "global_context_se3"
    if "radial_pair" in text or "high_capacity_radial_pair" in text:
        return "radial_pair"
    if "global_coeff" in text or "global_invariant" in text or "globalinvariantcoefficient" in text:
        return "global_coeff"
    if "internal_energy" in text or "internal_coordinate_energy" in text or "internalcoordinateenergy" in text:
        return "internal_energy"
    if "painn_lite" in text or "painn" in text:
        return "painn_lite"
    if "radial" in text:
        return "radial"
    if "memorizer" in text or "non_equivariant_coordinate_mlp" in text:
        return "mlp_memorizer"
    if "egnn" in text:
        return "egnn"
    if "se3_full" in text or "full_irrep" in text or ("molecularfullirrep" in text):
        return "se3_full"
    if "se3_scalar" in text or "scalar_attention_kernel" in text:
        return "se3_scalar"
    if "se3" in text:
        return "se3"
    if "tfn" in text:
        return "tfn"
    return "unknown"


def generalization_succeeded(rows: list[dict], margin_pct: float = GENERALIZATION_IMPROVEMENT_MARGIN_PCT) -> bool:
    for row in rows:
        vector_zero = number(row, "force_vector_l2_mae_improvement_vs_zero_pct")
        vector_mean = number(row, "force_vector_l2_mae_improvement_vs_mean_pct")
        component_zero = number(row, "force_mae_improvement_vs_zero_pct")
        component_mean = number(row, "force_mae_improvement_vs_mean_pct")
        vector_ok = vector_zero is not None and vector_mean is not None and min(vector_zero, vector_mean) >= margin_pct
        component_ok = component_zero is not None and component_mean is not None and min(component_zero, component_mean) >= margin_pct
        if vector_ok or component_ok:
            return True
    return False


def interpretation(
    overfit_rows: list[dict],
    random_rows: list[dict],
    chronological_rows: list[dict],
    high_gradient_rows: list[dict],
) -> list[str]:
    overfit_ok = overfit_succeeded(overfit_rows)
    if not overfit_ok:
        return [OVERFIT_FAIL_STATEMENT]

    families = {model_family(row) for row in overfit_rows}
    statements = []
    mlp_ok = any_success(overfit_rows, {"mlp_memorizer"})
    global_ok = any_success(overfit_rows, {"global_coeff"})
    global_context_ok = any_success(overfit_rows, {"global_context_radial", "global_context_painn", "global_context_se3"})
    internal_ok = any_success(overfit_rows, {"internal_energy"})
    painn_ok = any_success(overfit_rows, {"painn_lite"})
    radial_ok = any_success(overfit_rows, {"radial_pair", "radial"})
    full_se3_ok = any_success(overfit_rows, {"se3_full"})
    scalar_se3_ok = any_success(overfit_rows, {"se3_scalar", "se3"})
    equivariant_ok = any_success(
        overfit_rows,
        {
            "global_coeff",
            "global_context_radial",
            "global_context_painn",
            "global_context_se3",
            "internal_energy",
            "painn_lite",
            "radial_pair",
            "radial",
            "egnn",
            "tfn",
            "se3_scalar",
            "se3",
            "se3_full",
        },
    )
    if "mlp_memorizer" in families and not mlp_ok:
        return [MLP_FAIL_STATEMENT]
    if global_ok:
        statements.append(GLOBAL_COEFF_SOLVED_STATEMENT)
    elif internal_ok:
        statements.append(INTERNAL_ENERGY_SOLVED_STATEMENT)
    elif painn_ok:
        statements.append(PAINN_LITE_SOLVED_STATEMENT)
    elif global_context_ok:
        statements.append("Global-context local equivariant overfit is possible; compare against local-only baselines next.")
    elif full_se3_ok:
        statements.append(FULL_IRREP_SOLVED_STATEMENT)
    elif mlp_ok and not global_ok and not internal_ok and not painn_ok and not radial_ok and not scalar_se3_ok and not full_se3_ok:
        statements.append(PHASE11_EQUIVARIANT_FAIL_STATEMENT)
    elif mlp_ok and radial_ok and not scalar_se3_ok:
        statements.append(SE3_WRAPPER_BOTTLENECK_STATEMENT)
    elif mlp_ok and not radial_ok and not scalar_se3_ok and not full_se3_ok:
        statements.append(EQUIVARIANT_CAPACITY_STATEMENT)
    elif not equivariant_ok and "mlp_memorizer" in families:
        statements.append(MLP_ONLY_STATEMENT)
    elif radial_ok and not scalar_se3_ok and not full_se3_ok:
        statements.append(RADIAL_ONLY_STATEMENT)
    elif scalar_se3_ok:
        statements.append(TINY_OVERFIT_SOLVED_STATEMENT)

    full_graph_ok = any(row_meets_success_criteria(row) and str(row.get("graph_mode", "")).lower() == "full" for row in overfit_rows)
    cutoff_ok = any(row_meets_success_criteria(row) and str(row.get("graph_mode", "")).lower() == "cutoff" for row in overfit_rows)
    has_cutoff_rows = any(str(row.get("graph_mode", "")).lower() == "cutoff" for row in overfit_rows)
    if full_graph_ok and has_cutoff_rows and not cutoff_ok:
        statements.append(CUTOFF_BOTTLENECK_STATEMENT)
    scale_rows = [row for row in overfit_rows if row_meets_success_criteria(row)]
    if scale_rows and any((number(row, "force_output_scale") or 1.0) > 1.0 for row in scale_rows):
        statements.append(FORCE_SCALE_BOTTLENECK_STATEMENT)
    tiny_grad_rows = []
    for row in overfit_rows:
        if model_family(row) not in {
            "global_coeff",
            "global_context_radial",
            "global_context_painn",
            "global_context_se3",
            "internal_energy",
            "painn_lite",
            "se3_full",
            "se3_scalar",
            "se3",
            "egnn",
            "tfn",
        }:
            continue
        grad_values = [
            number(row, "train_force_head_grad_norm_max"),
            number(row, "train_edge_mlp_grad_norm_max"),
            number(row, "train_message_passing_grad_norm_max"),
            number(row, "train_backbone_grad_norm_max"),
        ]
        grad_values = [value for value in grad_values if value is not None]
        if grad_values and max(grad_values) < 1e-10:
            tiny_grad_rows.append(row)
    if tiny_grad_rows:
        statements.append(GRADIENT_PATH_BOTTLENECK_STATEMENT)
    if statements:
        return statements

    chrono_ok = generalization_succeeded(chronological_rows)
    random_ok = generalization_succeeded(random_rows)
    if chronological_rows and not chrono_ok:
        statements.append(LOCAL_LEARN_CHRONO_WEAK_STATEMENT)
    if random_rows and chronological_rows:
        if random_ok and not chrono_ok:
            statements.append(RANDOM_SHIFT_STATEMENT)
        elif not random_ok and not chrono_ok:
            statements.append(BOTH_GENERALIZATION_FAIL_STATEMENT)
    elif random_rows and not random_ok:
        statements.append(BOTH_GENERALIZATION_FAIL_STATEMENT)
    if high_gradient_rows:
        best_high = best_row(high_gradient_rows, "force_vector_l2_mae")
        statements.append(
            "High-gradient subset check: best vector-L2 force MAE is "
            f"{fmt(number(best_high, 'force_vector_l2_mae'))}; use this only as a qualitative stress diagnostic."
        )
    if not statements:
        statements.append("Tiny-overfit passed; add random and chronological diagnostics before making a bottleneck claim.")
    return statements


def table(rows: list[dict], title: str, train: bool = False) -> str:
    if not rows:
        return f"## {title}\n\nNo rows supplied.\n"
    lines = [
        f"## {title}",
        "",
        "| run | model | loss | clip | output scale | graph/split | train vector-L2 | norm ratio | cosine | improvement vs zero | success |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        train_l2 = number(row, "train_eval_force_vector_l2_mae") if train else None
        improvement = number(row, "train_eval_force_vector_l2_mae_improvement_vs_zero_pct") if train else None
        if train_l2 is None:
            train_l2 = number(row, "val_force_vector_l2_mae_final")
        if improvement is None:
            improvement = number(row, "force_vector_l2_mae_improvement_vs_zero_pct")
        lines.append(
            "| "
            + " | ".join(
                [
                    label(row),
                    model_family(row),
                    str(row.get("force_loss_type", "")),
                    str(row.get("gradient_clip_norm", "")),
                    fmt(number(row, "force_output_scale")),
                    str(row.get("graph_mode", row.get("split_type", ""))),
                    fmt(train_l2),
                    fmt(number(row, "pred_to_target_force_norm_ratio")),
                    fmt(number(row, "force_cosine_similarity_mean")),
                    fmt(improvement),
                    "yes" if row_meets_success_criteria(row) else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def best_summary(rows: list[dict], title: str, field: str, lower_is_better: bool) -> str:
    row = best_row(rows, field, lower_is_better=lower_is_better)
    return f"- {title}: {label(row) if row else 'n/a'} ({field}={fmt(number(row, field))})"


def grouped_best(rows: list[dict], group_field: str) -> str:
    values = sorted({str(row.get(group_field, "")) for row in rows if row.get(group_field, "") not in (None, "")})
    if not values:
        return "- n/a"
    lines = []
    for value in values:
        group = [row for row in rows if str(row.get(group_field, "")) == value]
        row = best_row(group, "train_eval_force_vector_l2_mae")
        lines.append(f"- {group_field}={value}: {fmt(number(row, 'train_eval_force_vector_l2_mae'))} ({label(row)})")
    return "\n".join(lines)


def family_best(rows: list[dict], family: str, title: str) -> str:
    group = [row for row in rows if model_family(row) == family]
    if not group:
        return f"- {title}: n/a"
    row = best_row(group, "train_eval_force_vector_l2_mae")
    return (
        f"- {title}: {label(row)}; train vector-L2={fmt(number(row, 'train_eval_force_vector_l2_mae'))}, "
        f"ratio={fmt(number(row, 'train_eval_pred_to_target_force_norm_ratio'))}, "
        f"cosine={fmt(number(row, 'train_eval_force_cosine_similarity_mean'))}"
    )


def gradient_flow_section(rows: list[dict]) -> str:
    if not rows:
        return "- n/a"
    equivariant_rows = [row for row in rows if model_family(row) != "mlp_memorizer"]
    best = best_row(equivariant_rows or rows, "train_eval_force_vector_l2_mae")
    fields = [
        "train_force_head_grad_norm_max",
        "train_backbone_grad_norm_max",
        "train_edge_mlp_grad_norm_max",
        "train_total_grad_norm_before_clip_max",
        "train_total_grad_norm_after_clip_max",
        "train_force_head_output_norm_final",
        "train_last_hidden_norm_final",
    ]
    lines = [f"- row: {label(best)}"]
    lines.extend(f"- {field}: {fmt(number(best, field))}" for field in fields)
    se3_rows = [row for row in rows if model_family(row) in {"se3_full", "se3_scalar", "se3"}]
    if se3_rows:
        se3_best = best_row(se3_rows, "train_eval_force_vector_l2_mae")
        lines.append(f"- best SE3 gradient row: {label(se3_best)}")
        lines.append(f"- best SE3 force-head grad max: {fmt(number(se3_best, 'train_force_head_grad_norm_max'))}")
        lines.append(f"- best SE3 backbone grad max: {fmt(number(se3_best, 'train_backbone_grad_norm_max'))}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit-summary", "--input", dest="overfit_summary", required=True)
    parser.add_argument("--overfit-per-run", "--per-run", dest="overfit_per_run", default=None)
    parser.add_argument("--random-summary", default=None)
    parser.add_argument("--chronological-summary", default=None)
    parser.add_argument("--high-gradient-summary", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    overfit_rows = read_rows(Path(args.overfit_summary))
    overfit_per_rows = read_rows(Path(args.overfit_per_run)) if args.overfit_per_run else []
    random_rows = read_rows(Path(args.random_summary)) if args.random_summary else []
    chronological_rows = read_rows(Path(args.chronological_summary)) if args.chronological_summary else []
    high_gradient_rows = read_rows(Path(args.high_gradient_summary)) if args.high_gradient_summary else []
    best_overfit = best_row(overfit_rows, "train_eval_force_vector_l2_mae")
    best_norm = best_row(overfit_rows, "train_eval_pred_to_target_force_norm_ratio", lower_is_better=False)
    best_cosine = best_row(overfit_rows, "train_eval_force_cosine_similarity_mean", lower_is_better=False)
    statements = interpretation(overfit_rows, random_rows, chronological_rows, high_gradient_rows)
    success_rows = [row for row in overfit_rows if row_meets_success_criteria(row)]
    equivariant_success_rows = [
        row
        for row in success_rows
        if model_family(row)
        in {
            "global_coeff",
            "global_context_radial",
            "global_context_painn",
            "global_context_se3",
            "internal_energy",
            "painn_lite",
            "radial_pair",
            "radial",
            "egnn",
            "tfn",
            "se3_scalar",
            "se3",
            "se3_full",
        }
    ]

    text = "\n".join(
        [
            "# rMD17 Force Diagnostic Report",
            "",
            "This report diagnoses whether real rMD17 force learning is blocked by optimization, target scaling, capacity, or trajectory split shift. It does not make an architecture superiority claim.",
            "",
            "## Interpretation",
            "",
            *[f"- {statement}" for statement in statements],
            "",
            "## Success Criteria",
            "",
            f"- Any run meets all tiny-overfit criteria: {'yes' if success_rows else 'no'}",
            f"- Any equivariant run meets all tiny-overfit criteria: {'yes' if equivariant_success_rows else 'no'}",
            f"- Tiny equivariant overfit solved: {'yes' if equivariant_success_rows else 'no'}",
            "- Required: >=50% train vector-L2 improvement vs zero, >=0.8 pred/target norm ratio, >=0.7 force cosine similarity.",
            "",
            "## Best Tiny-Overfit Row",
            "",
            f"- Config/run: {label(best_overfit) if best_overfit else 'n/a'}",
            f"- Train vector-L2 force MAE: {fmt(number(best_overfit, 'train_eval_force_vector_l2_mae'))}",
            f"- Train improvement vs zero-force vector-L2: {fmt(number(best_overfit, 'train_eval_force_vector_l2_mae_improvement_vs_zero_pct'))}%",
            f"- Prediction/target norm ratio: {fmt(number(best_overfit, 'train_eval_pred_to_target_force_norm_ratio'))}",
            f"- Train cosine similarity: {fmt(number(best_overfit, 'train_eval_force_cosine_similarity_mean'))}",
            best_summary(overfit_rows, "Best norm ratio", "train_eval_pred_to_target_force_norm_ratio", lower_is_better=False),
            best_summary(overfit_rows, "Best cosine similarity", "train_eval_force_cosine_similarity_mean", lower_is_better=False),
            "",
            "## Best By Family",
            "",
            family_best(overfit_rows, "mlp_memorizer", "Best MLP memorizer row"),
            family_best(overfit_rows, "global_coeff", "Best global invariant coefficient row"),
            family_best(overfit_rows, "global_context_radial", "Best global-context radial row"),
            family_best(overfit_rows, "internal_energy", "Best internal coordinate energy row"),
            family_best(overfit_rows, "painn_lite", "Best PaiNN-lite row"),
            family_best(overfit_rows, "radial_pair", "Best radial pair row"),
            family_best(overfit_rows, "egnn", "Best EGNN row"),
            family_best(overfit_rows, "tfn", "Best TFN row"),
            family_best(overfit_rows, "se3_scalar", "Best scalar SE3 row"),
            family_best(overfit_rows, "se3_full", "Best full-irrep SE3 row"),
            "",
            "## Grid Diagnoses",
            "",
            "### Gradient Flow",
            gradient_flow_section(overfit_rows),
            "",
            "### Gradient Clipping",
            grouped_best(overfit_rows, "gradient_clip_norm"),
            "",
            "### Output Scale",
            grouped_best(overfit_rows, "force_output_scale"),
            "",
            "### Graph Mode",
            grouped_best(overfit_rows, "graph_mode"),
            "",
            "### Atom-Pair Conditioning",
            grouped_best(overfit_rows, "uses_atom_pair_embedding"),
            "",
            "### Loss Type",
            grouped_best(overfit_rows, "force_loss_type"),
            "",
            table(overfit_rows, "Tiny-Overfit Diagnostics", train=True),
            table(random_rows, "Random Split Diagnostics"),
            table(chronological_rows, "Chronological Split Diagnostics"),
            table(high_gradient_rows, "High-Gradient Diagnostics"),
            "## Learning Curves",
            "",
            f"- Overfit per-run rows: {len(overfit_per_rows)}",
            "- Training curve paths are recorded in `training_curve` in the per-run CSV.",
            "",
            "## Guardrail",
            "",
            "Architecture superiority is not established by this diagnostic report.",
            "",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
