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


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def labels(rows: list[dict]) -> list[str]:
    out = []
    for row in rows:
        label = row.get("config_name", "row").replace(".yaml", "")
        cutoff = row.get("cutoff_radius_mean")
        if cutoff not in (None, ""):
            label = f"{label} r={float(cutoff):g}"
        out.append(label)
    return out


def has_fields(rows: list[dict], *fields: str) -> bool:
    return bool(rows) and all(field in rows[0] for field in fields)


def values(rows: list[dict], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row.get(field) not in (None, "", "nan")]


def save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def bar(rows: list[dict], metric: str, ylabel: str, output: Path, logy: bool = False, threshold: float | None = None) -> None:
    if not has_fields(rows, f"{metric}_mean"):
        return
    y = values(rows, f"{metric}_mean")
    if not y:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(rows)), 4))
    ax.bar(range(len(rows)), [max(v, 1e-16) for v in y] if logy else y, color="#4c78a8")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels(rows), rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if threshold is not None:
        ax.axhline(threshold, color="#d62728", linestyle="--")
    fig.tight_layout()
    save(fig, output)


def scatter(rows: list[dict], x_field: str, y_field: str, xlabel: str, ylabel: str, output: Path) -> None:
    if not has_fields(rows, x_field, y_field):
        return
    rows = [row for row in rows if row.get(x_field) not in (None, "", "nan") and row.get(y_field) not in (None, "", "nan")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = values(rows, x_field)
    ys = values(rows, y_field)
    ax.scatter(xs, ys, color="#f58518")
    for row, x, y in zip(rows, xs, ys):
        ax.annotate(row["config_name"].replace(".yaml", ""), (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    save(fig, output)


def direct_vs_energy(rows: list[dict], output: Path) -> None:
    selected = [row for row in rows if row.get("training_mode") in {"direct_force", "energy_force"}]
    if not selected:
        return
    bar(selected, "force_mae", "force MAE", output)


def cutoff_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("cutoff_radius_mean") not in (None, "", "nan")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_rows(Path(args.input))
    output = Path(args.output)
    bar(rows, "force_mae", "force MAE", output / "force_mae_bar.png")
    bar(rows, "force_rmse", "force RMSE", output / "force_rmse_bar.png")
    bar(rows, "force_vector_l2_mae", "force vector L2 MAE", output / "force_vector_l2_mae_bar.png")
    bar(rows, "rotated_force_mae", "rotated force MAE", output / "rotated_force_mae_bar.png")
    bar(rows, "equivariance_error", "equivariance error", output / "equivariance_error_log.png", logy=True, threshold=1e-5)
    scatter(rows, "runtime_per_batch_sec_mean", "force_mae_mean", "runtime per batch (s)", "force MAE", output / "runtime_vs_force_mae.png")
    scatter(rows, "average_neighbors_mean", "force_mae_mean", "average neighbors", "force MAE", output / "neighbors_vs_force_mae.png")
    crows = cutoff_rows(rows)
    if crows:
        scatter(crows, "cutoff_radius_mean", "force_mae_mean", "cutoff radius", "force MAE", output / "cutoff_vs_force_mae.png")
        scatter(crows, "cutoff_radius_mean", "force_rmse_mean", "cutoff radius", "force RMSE", output / "cutoff_vs_force_rmse.png")
        scatter(crows, "cutoff_radius_mean", "force_vector_l2_mae_mean", "cutoff radius", "force vector L2 MAE", output / "cutoff_vs_force_vector_l2_mae.png")
        scatter(crows, "cutoff_radius_mean", "average_neighbors_mean", "cutoff radius", "average neighbors", output / "cutoff_vs_neighbors.png")
        scatter(crows, "cutoff_radius_mean", "runtime_per_batch_sec_mean", "cutoff radius", "runtime per batch (s)", output / "cutoff_vs_runtime.png")
    direct_vs_energy(rows, output / "direct_vs_energy_force_mae.png")
    print(f"wrote real molecular plots to {output}")


if __name__ == "__main__":
    main()
