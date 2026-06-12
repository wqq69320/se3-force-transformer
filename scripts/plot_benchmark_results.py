#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

Path("outputs/.mplconfig").mkdir(parents=True, exist_ok=True)
Path("outputs/.cache").mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(Path("outputs/.mplconfig").resolve())
os.environ["XDG_CACHE_HOME"] = str(Path("outputs/.cache").resolve())
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt

from se3force.evaluation.display import display_name_for_config


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label(row: dict) -> str:
    return row.get("display_name") or display_name_for_config(row["config_name"])


def labels(rows: list[dict]) -> list[str]:
    return [label(row) for row in rows]


def floats(rows: list[dict], field: str) -> list[float]:
    return [float(row[field]) for row in rows]


def save_figure(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    png_path = output if output.suffix.lower() == ".png" else output.with_suffix(".png")
    svg_path = png_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)


def annotate_n(ax, x: list[int], y: list[float], rows: list[dict], logy: bool = False) -> None:
    for xi, yi, row in zip(x, y, rows):
        n = row.get("n", "")
        if not n:
            continue
        if logy:
            ax.annotate(f"n={n}", (xi, max(yi, 1e-16)), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        else:
            ax.annotate(f"n={n}", (xi, yi), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)


def bar_plot(
    rows: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
    logy: bool = False,
    threshold: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(rows)), 4))
    x = list(range(len(rows)))
    y = floats(rows, f"{metric}_mean")
    err = floats(rows, f"{metric}_std")
    plot_y = [max(value, 1e-16) for value in y] if logy else y
    ax.bar(x, plot_y, yerr=err, capsize=4, color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels(rows), rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    if threshold is not None:
        ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.2)
        ax.text(0.99, threshold, f"threshold {threshold:.0e}", color="#d62728", ha="right", va="bottom", transform=ax.get_yaxis_transform())
    annotate_n(ax, x, plot_y, rows, logy=logy)
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def scatter_plot(rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = floats(rows, "parameter_count_mean")
    y = floats(rows, "rotated_translated_mse_mean")
    ax.scatter(x, y, color="#f58518")
    for row, xi, yi in zip(rows, x, y):
        ax.annotate(f"{label(row)}\nn={row.get('n', '')}", (xi, yi), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("parameter count")
    ax.set_ylabel("rotated-translated MSE")
    ax.set_title("Parameter Count vs Rotated MSE")
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def infer_lmax(config_name: str) -> int | None:
    match = re.search(r"lmax(\d+)", config_name)
    if match:
        return int(match.group(1))
    match = re.search(r"_l(\d+)(?:\.yaml)?$", config_name)
    if match:
        return int(match.group(1))
    return None


def lmax_plot(rows: list[dict], output: Path) -> None:
    lmax_rows = []
    for row in rows:
        if not Path(row["config_name"]).stem.startswith("ablation_lmax"):
            continue
        lmax = infer_lmax(row["config_name"])
        if lmax is not None:
            lmax_rows.append((lmax, row))
    lmax_rows.sort(key=lambda item: item[0])

    fig, ax = plt.subplots(figsize=(6, 4))
    unique_lmax = sorted({item[0] for item in lmax_rows})
    if len(unique_lmax) >= 2:
        xs = [item[0] for item in lmax_rows]
        ys = [float(item[1]["rotated_translated_mse_mean"]) for item in lmax_rows]
        errs = [float(item[1]["rotated_translated_mse_std"]) for item in lmax_rows]
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=4, color="#54a24b")
        for x_value, y_value, (_, row) in zip(xs, ys, lmax_rows):
            ax.annotate(f"n={row.get('n', '')}", (x_value, y_value), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xlabel("lmax")
        ax.set_ylabel("rotated-translated MSE")
    else:
        warning = "Insufficient lmax ablation data\nneed at least two ablation_lmax* configs"
        print(f"warning: {warning.replace(chr(10), '; ')}")
        ax.text(0.5, 0.5, warning, ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("lmax Ablation")
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    if not rows:
        raise SystemExit("no rows found in benchmark summary")
    output_dir = Path(args.output)
    bar_plot(rows, "canonical_mse", "canonical MSE", "Canonical Test MSE", output_dir / "canonical_mse_bar.png")
    bar_plot(
        rows,
        "rotated_translated_mse",
        "rotated-translated MSE",
        "Rotated-Translated Test MSE",
        output_dir / "rotated_mse_bar.png",
    )
    bar_plot(
        rows,
        "equivariance_error",
        "equivariance error",
        "Equivariance Error",
        output_dir / "equivariance_error_bar.png",
        logy=True,
        threshold=1e-5,
    )
    scatter_plot(rows, output_dir / "parameter_count_vs_rotated_mse.png")
    lmax_plot(rows, output_dir / "lmax_ablation.png")
    print(f"wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
