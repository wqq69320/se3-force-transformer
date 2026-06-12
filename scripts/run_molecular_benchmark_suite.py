#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import traceback
from pathlib import Path
from statistics import mean, stdev

from se3force.evaluation.metrics_schema import REQUIRED_METRIC_FIELDS
from se3force.evaluation.molecular_metrics import MOLECULAR_REQUIRED_FIELDS
from se3force.training.molecular_trainer import load_molecular_config, train_molecular_from_config

NUMERIC_FIELDS = [
    "seed",
    "num_train_samples",
    "num_val_samples",
    "num_test_samples",
    "force_mae",
    "force_rmse",
    "rotated_force_mae",
    "energy_mae",
    "energy_rmse",
    "equivariance_error",
    "parameter_count",
    "runtime_per_batch_sec",
    "average_neighbors",
    "edge_count_mean",
    "edge_count_max",
]
GROUP_FIELDS = ["config_name", "model_name", "dataset_name", "molecule_name", "training_mode"]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return all(field in data for field in REQUIRED_METRIC_FIELDS + MOLECULAR_REQUIRED_FIELDS)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in GROUP_FIELDS)
        groups.setdefault(key, []).append(row)
    out_rows = []
    for key, group in sorted(groups.items()):
        out = {field: value for field, value in zip(GROUP_FIELDS, key)}
        out["n"] = len(group)
        for field in NUMERIC_FIELDS:
            values = []
            for row in group:
                value = row.get(field)
                if value in (None, "", "nan"):
                    continue
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
            out[f"{field}_mean"] = mean(values) if values else ""
            out[f"{field}_std"] = stdev(values) if len(values) > 1 else (0.0 if values else "")
        out_rows.append(out)
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output)
    rows = []
    failures = []
    for config_arg in args.configs:
        config_path = Path(config_arg)
        for seed in args.seeds:
            run_name = f"{config_path.stem}_seed{seed}"
            run_dir = output_root / "runs" / run_name
            metrics_path = run_dir / "metrics.json"
            try:
                if not args.force and complete(metrics_path):
                    print(f"skipping completed run: {run_name}")
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    config = copy.deepcopy(load_molecular_config(config_path))
                    config["seed"] = seed
                    config["run_name"] = run_name
                    config["output_dir"] = str(run_dir)
                    config.setdefault("dataset", {})["seed"] = seed
                    metrics = train_molecular_from_config(config)
                rows.append(metrics)
                print(f"run={run_name} force_mae={metrics['force_mae']:.6g} equivariance_error={metrics['equivariance_error']:.3e}")
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "run_name": run_name,
                        "config_name": config_path.name,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"FAILED run={run_name} error={type(exc).__name__}: {exc}")

    per_fields = REQUIRED_METRIC_FIELDS + MOLECULAR_REQUIRED_FIELDS + ["last_checkpoint"]
    write_csv(output_root / "per_run_metrics.csv", rows, per_fields)
    summary = summarize(rows)
    summary_fields = GROUP_FIELDS + ["n"]
    for field in NUMERIC_FIELDS:
        summary_fields.extend([f"{field}_mean", f"{field}_std"])
    write_csv(output_root / "summary_mean_std.csv", summary, summary_fields)
    failure_path = output_root / "failed_runs.csv"
    if failures:
        write_csv(failure_path, failures, ["run_name", "config_name", "seed", "error_type", "error_message", "traceback"])
        print(f"wrote {failure_path}")
    elif failure_path.exists():
        failure_path.unlink()
    print(f"wrote {output_root / 'per_run_metrics.csv'}")
    print(f"wrote {output_root / 'summary_mean_std.csv'}")


if __name__ == "__main__":
    main()
