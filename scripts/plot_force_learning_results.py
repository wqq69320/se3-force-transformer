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


def label(row: dict) -> str:
    cfg = row.get("config_name", "row").replace(".yaml", "")
    lr = row.get("learning_rate", "")
    loss = row.get("force_loss_type", "")
    return f"{cfg}\nlr={lr} {loss}"


def save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def bar(rows: list[dict], field: str, ylabel: str, output: Path, logy: bool = False) -> None:
    rows = [row for row in rows if row.get(field) not in (None, "", "nan")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.2), 4))
    values = [float(row[field]) for row in rows]
    ax.bar(range(len(rows)), [max(v, 1e-16) for v in values] if logy else values, color="#4c78a8")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([label(row) for row in rows], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    save(fig, output)


def scatter(rows: list[dict], x_field: str, y_field: str, xlabel: str, ylabel: str, output: Path) -> None:
    rows = [row for row in rows if row.get(x_field) not in (None, "", "nan") and row.get(y_field) not in (None, "", "nan")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = [float(row[x_field]) for row in rows]
    ys = [float(row[y_field]) for row in rows]
    ax.scatter(xs, ys, color="#f58518")
    for row, x, y in zip(rows, xs, ys):
        ax.annotate(row.get("config_name", "").replace(".yaml", ""), (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    save(fig, output)


def curve(per_rows: list[dict], metric: str, ylabel: str, output: Path) -> None:
    plotted = False
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in per_rows:
        path = row.get("training_curve")
        if not path or not Path(path).exists():
            continue
        curve_rows = read_rows(Path(path))
        if not curve_rows or metric not in curve_rows[0]:
            continue
        xs = [int(r["epoch"]) for r in curve_rows if r.get(metric) not in (None, "", "nan")]
        ys = [float(r[metric]) for r in curve_rows if r.get(metric) not in (None, "", "nan")]
        if not xs:
            continue
        ax.plot(xs, ys, label=row.get("run_name", row.get("config_name", ""))[:40])
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6)
    fig.tight_layout()
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--per-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_rows(Path(args.input))
    per_rows = read_rows(Path(args.per_run))
    output = Path(args.output)
    curve(per_rows, "val_force_vector_l2_mae", "validation vector-L2 force MAE", output / "val_vector_l2_force_mae_over_epochs.png")
    curve(per_rows, "val_force_mae", "validation component force MAE", output / "val_component_force_mae_over_epochs.png")
    bar(rows, "force_mae_mean", "force MAE", output / "force_mae_by_model_lr.png")
    bar(rows, "force_vector_l2_mae_mean", "vector-L2 force MAE", output / "force_vector_l2_mae_by_model_lr.png")
    bar(rows, "force_mae_improvement_vs_zero_pct_mean", "improvement vs zero (%)", output / "improvement_vs_zero.png")
    bar(rows, "force_mae_improvement_vs_mean_pct_mean", "improvement vs mean (%)", output / "improvement_vs_mean.png")
    scatter(rows, "runtime_per_batch_sec_mean", "force_vector_l2_mae_mean", "runtime per batch (s)", "vector-L2 force MAE", output / "runtime_vs_vector_l2_force_mae.png")
    bar(rows, "equivariance_error_mean", "equivariance error", output / "equivariance_error_log.png", logy=True)
    print(f"wrote force-learning plots to {output}")


if __name__ == "__main__":
    main()
