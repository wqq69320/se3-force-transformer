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


def read_csv(path: str | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def text_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def model_family(row: dict) -> str:
    config = str(row.get("config_name", "")).lower()
    model = str(row.get("model_name", "")).lower()
    if "global_context_radial" in config or model == "global_context_radial":
        return "global_context_radial" if text_bool(row.get("uses_global_context", "")) and "_no_global" not in config else "global_context_radial_no_global"
    if "global_coeff" in config or model == "global_coeff":
        return "global_coeff"
    if "radial_pair" in config or model == "radial_pair":
        return "radial_pair"
    if "painn" in config or model == "painn_lite":
        return "painn_lite"
    if "egnn" in config or model == "egnn":
        return "egnn"
    if "tfn" in config or model == "tfn":
        return "tfn"
    if "se3" in config or model in {"se3", "se3_transformer"}:
        return "se3"
    return model or Path(row.get("config_name", "")).stem


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def bar(rows: list[dict], metric: str, ylabel: str, title: str, path: Path) -> None:
    rows = [row for row in rows if number(row, metric) is not None]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: model_family(r))
    labels = [model_family(row) for row in rows]
    values = [number(row, metric) for row in rows]
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.1), 4.5))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    save(fig, path)


def scatter(rows: list[dict], x_key: str, y_key: str, xlabel: str, ylabel: str, title: str, path: Path) -> None:
    points = [(number(row, x_key), number(row, y_key), model_family(row)) for row in rows]
    points = [(x, y, label) for x, y, label in points if x is not None and y is not None]
    if not points:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for x, y, label in points:
        ax.scatter([x], [y], label=label)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save(fig, path)


def equivariance(rows: list[dict], path: Path) -> None:
    rows = [row for row in rows if number(row, "force_equivariance_error_mean") is not None]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: model_family(r))
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.1), 4.5))
    ax.bar([model_family(row) for row in rows], [number(row, "force_equivariance_error_mean") for row in rows])
    ax.set_yscale("log")
    ax.axhline(1e-5, color="red", linestyle="--", linewidth=1)
    ax.set_ylabel("force equivariance error")
    ax.set_title("Equivariance Error By Model")
    ax.tick_params(axis="x", rotation=30)
    save(fig, path)


def generalization_gap(random_rows: list[dict], chrono_rows: list[dict], path: Path) -> None:
    random_by_model = {model_family(row): row for row in random_rows}
    chrono_by_model = {model_family(row): row for row in chrono_rows}
    labels = sorted(set(random_by_model) & set(chrono_by_model))
    gaps = []
    for label in labels:
        r = number(random_by_model[label], "force_vector_l2_mae_mean")
        c = number(chrono_by_model[label], "force_vector_l2_mae_mean")
        if r is not None and c is not None:
            gaps.append((label, c - r))
    if not gaps:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(gaps) * 1.1), 4.5))
    ax.bar([x for x, _ in gaps], [y for _, y in gaps])
    ax.set_ylabel("chrono vector-L2 minus random vector-L2")
    ax.set_title("Random vs Chronological Generalization Gap")
    ax.tick_params(axis="x", rotation=30)
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-input")
    parser.add_argument("--chrono-input")
    parser.add_argument("--ablation-input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    random_rows = read_csv(args.random_input)
    chrono_rows = read_csv(args.chrono_input)
    ablation_rows = read_csv(args.ablation_input)
    all_rows = random_rows + chrono_rows
    output = Path(args.output)

    bar(random_rows, "force_vector_l2_mae_mean", "vector-L2 force MAE", "Random Split Vector-L2 By Model", output / "random_vector_l2_by_model.png")
    bar(chrono_rows, "force_vector_l2_mae_mean", "vector-L2 force MAE", "Chronological Split Vector-L2 By Model", output / "chronological_vector_l2_by_model.png")
    bar(all_rows, "force_vector_l2_mae_improvement_vs_zero_pct_mean", "improvement vs zero (%)", "Improvement vs Zero Baseline", output / "improvement_vs_zero_by_model.png")
    bar(all_rows, "force_vector_l2_mae_improvement_vs_mean_pct_mean", "improvement vs mean (%)", "Improvement vs Mean Baseline", output / "improvement_vs_mean_by_model.png")
    scatter(all_rows, "runtime_per_batch_sec_mean", "force_vector_l2_mae_mean", "runtime per batch (s)", "vector-L2 force MAE", "Runtime vs Vector-L2", output / "runtime_vs_vector_l2.png")
    scatter(all_rows, "parameter_count_mean", "force_vector_l2_mae_mean", "parameter count", "vector-L2 force MAE", "Parameter Count vs Vector-L2", output / "parameter_count_vs_vector_l2.png")
    equivariance(all_rows, output / "equivariance_error_log.png")
    generalization_gap(random_rows, chrono_rows, output / "random_vs_chrono_generalization_gap.png")
    bar(ablation_rows, "force_vector_l2_mae_mean", "vector-L2 force MAE", "Global Feature Ablations", output / "ablation_vector_l2_by_model.png")
    print(f"wrote global-vs-local plots to {output}")


if __name__ == "__main__":
    main()
