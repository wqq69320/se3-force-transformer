#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: str) -> str:
    if value in ("", None):
        return "n/a"
    number = float(value)
    return f"{number:.4e}" if abs(number) < 1e-3 or abs(number) > 1e4 else f"{number:.4g}"


def make_report(rows: list[dict]) -> str:
    table = ["| config | molecule | mode | n | force MAE | force RMSE | equivariance | params | avg neighbors |", "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        table.append(
            "| "
            + " | ".join(
                [
                    row["config_name"].replace(".yaml", ""),
                    row.get("molecule_name", ""),
                    row.get("training_mode", ""),
                    row.get("n", ""),
                    fmt(row.get("force_mae_mean", "")),
                    fmt(row.get("force_rmse_mean", "")),
                    fmt(row.get("equivariance_error_mean", "")),
                    fmt(row.get("parameter_count_mean", "")),
                    fmt(row.get("average_neighbors_mean", "")),
                ]
            )
            + " |"
        )
    return f"""# Molecular Benchmark Report

This report summarizes molecular-scale runs. It is not a broad molecular-force-field claim unless real molecular datasets, multiple molecules, and sufficient seeds are present.

## Force Metrics

{chr(10).join(table)}

## Interpretation

Force MAE/RMSE measure prediction performance in the configured force unit. Equivariance error measures coordinate-frame consistency. Runtime, parameter count, and average neighbor count should be treated as part of the accuracy/cost trade-off.

## Limitations

Synthetic molecular scale-up runs validate infrastructure and scaling behavior. rMD17 or MD22 claims require local real-data runs. Short demos are qualitative scientific-AI checks and do not prove physical correctness by themselves.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_rows(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(make_report(rows), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
