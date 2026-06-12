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


def plot_metric(rows: list[dict], metric: str, output: Path, ylabel: str) -> None:
    labels = [row["config_name"].replace(".yaml", "") for row in rows]
    values = [float(row[f"{metric}_mean"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.1), 4))
    ax.bar(range(len(rows)), values, color="#4c78a8")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_rows(Path(args.input))
    output = Path(args.output)
    plot_metric(rows, "force_mae", output / "force_mae.png", "force MAE")
    plot_metric(rows, "runtime_per_batch_sec", output / "runtime_per_batch.png", "runtime per batch (s)")
    plot_metric(rows, "edge_count_mean", output / "edge_count_mean.png", "mean edge count")
    print(f"wrote molecular plots to {output}")


if __name__ == "__main__":
    main()
