#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

Path("outputs/.mplconfig").mkdir(parents=True, exist_ok=True)
Path("outputs/.cache").mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(Path("outputs/.mplconfig").resolve())
os.environ["XDG_CACHE_HOME"] = str(Path("outputs/.cache").resolve())
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt


def read_rows(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def value(row: dict, field: str) -> float | None:
    for name in (field, f"{field}_mean"):
        raw = row.get(name)
        if raw in (None, "", "nan"):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def label(row: dict) -> str:
    name = (row.get("run_name") or row.get("config_name", "row")).replace(".yaml", "")
    family = model_family(row)
    if family != "unknown" and family not in name.lower():
        name = f"{family}: {name}"
    lr = row.get("learning_rate", "")
    loss = row.get("force_loss_type", "")
    if lr not in (None, ""):
        return f"{name}\nlr={lr} {loss}"
    return str(name)


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
        return "mlp"
    if "egnn" in text:
        return "egnn"
    if "se3_full" in text or "full_irrep" in text or "molecularfullirrep" in text:
        return "se3_full"
    if "se3_scalar" in text or "scalar_attention_kernel" in text:
        return "se3_scalar"
    if "se3" in text:
        return "se3"
    if "tfn" in text:
        return "tfn"
    return "unknown"


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def add_threshold(ax, metric: str) -> None:
    if "improvement" in metric:
        ax.axhline(50.0, color="#d62728", linestyle="--", linewidth=1.0)
    elif "pred_to_target_force_norm_ratio" in metric:
        ax.axhline(0.8, color="#d62728", linestyle="--", linewidth=1.0)
    elif "force_cosine" in metric:
        ax.axhline(0.7, color="#d62728", linestyle="--", linewidth=1.0)


def curve(per_rows: list[dict], metric: str, ylabel: str, output: Path, logy: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for row in per_rows:
        curve_path = row.get("training_curve")
        if not curve_path:
            continue
        rows = read_rows(Path(curve_path))
        if not rows or metric not in rows[0]:
            continue
        xs = []
        ys = []
        for item in rows:
            raw = item.get(metric)
            if raw in (None, "", "nan"):
                continue
            xs.append(int(float(item["epoch"])))
            ys.append(float(raw))
        if not xs:
            continue
        ax.plot(xs, ys, label=label(row)[:52], linewidth=1.4)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    add_threshold(ax, metric)
    if logy and any(line_y > 0 for line in ax.lines for line_y in line.get_ydata()):
        ax.set_yscale("log")
    ax.legend(fontsize=6)
    fig.tight_layout()
    save(fig, output)


def bar(rows: list[dict], field: str, ylabel: str, output: Path, logy: bool = False) -> None:
    pairs = [(label(row), value(row, field)) for row in rows]
    pairs = [(name, val) for name, val in pairs if val is not None]
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(pairs) * 1.15), 4))
    vals = [max(val, 1e-16) for _, val in pairs] if logy else [val for _, val in pairs]
    ax.bar(range(len(pairs)), vals, color="#4c78a8")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([name for name, _ in pairs], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    add_threshold(ax, field)
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    save(fig, output)


def best_by_group(
    rows: list[dict],
    group_field: str,
    metric: str,
    ylabel: str,
    output: Path,
    logy: bool = False,
    lower_is_better: bool = True,
) -> None:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = model_family(row) if group_field == "model_family" else str(row.get(group_field, ""))
        if key in {"", "None"}:
            continue
        groups.setdefault(key, []).append(row)
    pairs = []
    for key, group in sorted(groups.items()):
        scored = [(value(row, metric), row) for row in group]
        scored = [(score, row) for score, row in scored if score is not None]
        if not scored:
            continue
        best_score, _ = min(scored, key=lambda item: item[0]) if lower_is_better else max(scored, key=lambda item: item[0])
        pairs.append((key, best_score))
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(max(5, len(pairs) * 1.0), 4))
    vals = [max(score, 1e-16) for _, score in pairs] if logy else [score for _, score in pairs]
    ax.bar(range(len(pairs)), vals, color="#72b7b2")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([key for key, _ in pairs], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    save(fig, output)


def random_vs_chrono(random_rows: list[dict], chrono_rows: list[dict], output: Path) -> None:
    rows = [("random", row) for row in random_rows] + [("chronological", row) for row in chrono_rows]
    pairs = []
    for split, row in rows:
        val = value(row, "force_vector_l2_mae")
        if val is not None:
            pairs.append((f"{split}\n{label(row)}", val))
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(pairs) * 1.1), 4))
    ax.bar(range(len(pairs)), [val for _, val in pairs], color=["#54a24b" if name.startswith("random") else "#e45756" for name, _ in pairs])
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([name for name, _ in pairs], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("test vector-L2 force MAE")
    fig.tight_layout()
    save(fig, output)


def heatmap(rows: list[dict], output: Path) -> None:
    points = []
    for row in rows:
        x = row.get("learning_rate")
        y = row.get("force_loss_type") or row.get("config_name", "row")
        z = value(row, "train_eval_force_vector_l2_mae") or value(row, "force_vector_l2_mae")
        if x in (None, "") or y in (None, "") or z is None:
            continue
        points.append((str(x), str(y), z))
    if not points:
        return
    xs = sorted({x for x, _, _ in points}, key=lambda raw: float(raw))
    ys = sorted({y for _, y, _ in points})
    matrix = [[float("nan") for _ in xs] for _ in ys]
    for x, y, z in points:
        matrix[ys.index(y)][xs.index(x)] = z
    fig, ax = plt.subplots(figsize=(max(5, len(xs) * 1.1), max(3.5, len(ys) * 0.55)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=30, ha="right")
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels(ys)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("loss/config")
    ax.set_title("vector-L2 force MAE")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit-summary", "--input", dest="overfit_summary", required=True)
    parser.add_argument("--overfit-per-run", "--per-run", dest="overfit_per_run", required=True)
    parser.add_argument("--random-summary", default=None)
    parser.add_argument("--chronological-summary", default=None)
    parser.add_argument("--high-gradient-summary", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    overfit_rows = read_rows(Path(args.overfit_summary))
    per_rows = read_rows(Path(args.overfit_per_run))
    random_rows = read_rows(Path(args.random_summary)) if args.random_summary else []
    chrono_rows = read_rows(Path(args.chronological_summary)) if args.chronological_summary else []
    high_gradient_rows = read_rows(Path(args.high_gradient_summary)) if args.high_gradient_summary else []
    output = Path(args.output)

    curve(per_rows, "train_force_vector_l2_mae", "train vector-L2 force MAE", output / "train_vector_l2_force_mae_over_epochs.png", logy=True)
    curve(per_rows, "train_force_mae", "train component force MAE", output / "train_component_force_mae_over_epochs.png", logy=True)
    curve(per_rows, "train_pred_to_target_force_norm_ratio", "train prediction/target norm ratio", output / "train_norm_ratio_over_epochs.png")
    curve(per_rows, "train_force_cosine_similarity_mean", "train force cosine similarity", output / "train_force_cosine_over_epochs.png")
    curve(per_rows, "train_force_head_grad_norm", "train force-head gradient norm", output / "train_force_head_grad_norm_over_epochs.png", logy=True)
    curve(per_rows, "train_backbone_grad_norm", "train backbone gradient norm", output / "train_backbone_grad_norm_over_epochs.png", logy=True)
    curve(per_rows, "train_total_grad_norm_before_clip", "total gradient norm before clipping", output / "train_total_grad_norm_before_clip_over_epochs.png", logy=True)
    curve(per_rows, "train_total_grad_norm_after_clip", "total gradient norm after clipping", output / "train_total_grad_norm_after_clip_over_epochs.png", logy=True)
    bar(overfit_rows, "train_eval_force_vector_l2_mae", "final train vector-L2 force MAE", output / "overfit_final_train_vector_l2_force_mae.png", logy=True)
    bar(overfit_rows, "train_eval_force_vector_l2_mae_improvement_vs_zero_pct", "final train improvement vs zero (%)", output / "overfit_final_train_improvement_vs_zero.png")
    best_by_group(overfit_rows, "force_loss_type", "train_eval_force_vector_l2_mae", "best train vector-L2 force MAE", output / "best_train_vector_l2_by_loss_type.png", logy=True)
    best_by_group(overfit_rows, "force_output_scale", "train_eval_force_vector_l2_mae", "best train vector-L2 force MAE", output / "best_train_vector_l2_by_force_output_scale.png", logy=True)
    best_by_group(
        overfit_rows,
        "force_output_scale",
        "train_eval_pred_to_target_force_norm_ratio",
        "best pred/target norm ratio",
        output / "best_norm_ratio_by_force_output_scale.png",
        lower_is_better=False,
    )
    best_by_group(overfit_rows, "gradient_clip_norm", "train_eval_force_vector_l2_mae", "best train vector-L2 force MAE", output / "best_train_vector_l2_by_gradient_clipping.png", logy=True)
    best_by_group(overfit_rows, "graph_mode", "train_eval_force_vector_l2_mae", "best train vector-L2 force MAE", output / "best_train_vector_l2_by_graph_mode.png", logy=True)
    best_by_group(overfit_rows, "model_family", "train_eval_force_vector_l2_mae", "best train vector-L2 force MAE", output / "mlp_radial_egnn_se3_overfit_comparison.png", logy=True)
    best_by_group(
        overfit_rows,
        "model_family",
        "train_eval_pred_to_target_force_norm_ratio",
        "best pred/target norm ratio",
        output / "best_norm_ratio_by_model_family.png",
        lower_is_better=False,
    )
    best_by_group(
        overfit_rows,
        "model_family",
        "train_eval_force_cosine_similarity_mean",
        "best force cosine similarity",
        output / "best_cosine_by_model_family.png",
        lower_is_better=False,
    )
    random_vs_chrono(random_rows, chrono_rows, output / "random_vs_chronological_vector_l2_force_mae.png")
    heatmap(overfit_rows + random_rows + chrono_rows + high_gradient_rows, output / "diagnostic_vector_l2_heatmap.png")
    print(f"wrote force diagnostic plots to {output}")


if __name__ == "__main__":
    main()
