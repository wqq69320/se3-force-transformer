#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import traceback
from pathlib import Path
from statistics import mean, stdev

from se3force.evaluation.display import display_name_for_config
from se3force.evaluation.metrics_schema import REQUIRED_METRIC_FIELDS
from se3force.training.trainer import load_config, train_from_config

NUMERIC_FIELDS = [
    "seed",
    "num_train_samples",
    "num_val_samples",
    "num_test_samples",
    "final_train_loss",
    "best_val_mse",
    "canonical_mse",
    "rotated_translated_mse",
    "equivariance_error",
    "parameter_count",
    "runtime_per_batch_sec",
]

GROUP_FIELDS = ["config_name", "display_name", "model_name", "dataset_name"]


def is_complete_metrics(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return all(field in metrics for field in REQUIRED_METRIC_FIELDS)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def benchmark_config(config_path: Path, seed: int, output_root: Path, force: bool) -> dict:
    config = copy.deepcopy(load_config(config_path))
    config_stem = config_path.stem
    run_name = f"{config_stem}_seed{seed}"
    run_dir = output_root / "runs" / run_name
    metrics_path = run_dir / "metrics.json"

    if not force and is_complete_metrics(metrics_path):
        print(f"skipping completed run: {run_name}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.setdefault("display_name", display_name_for_config(config_path.name))
        return metrics

    config["seed"] = seed
    config["run_name"] = run_name
    config["config_name"] = config_path.name
    config["_config_path"] = str(config_path)
    config["output_dir"] = str(run_dir)
    config.setdefault("dataset", {})["seed"] = seed
    metrics = train_from_config(config)
    metrics.setdefault("display_name", display_name_for_config(config_path.name))
    return metrics


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in GROUP_FIELDS)
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for key, group_rows in sorted(groups.items()):
        out = {field: value for field, value in zip(GROUP_FIELDS, key)}
        out["n"] = len(group_rows)
        for field in NUMERIC_FIELDS:
            values = [float(row[field]) for row in group_rows if row.get(field) not in (None, "")]
            if values:
                out[f"{field}_mean"] = mean(values)
                out[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
            else:
                out[f"{field}_mean"] = ""
                out[f"{field}_std"] = ""
        summary_rows.append(out)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", default="outputs/benchmark_suite")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output)
    rows = []
    failures = []
    for config_arg in args.configs:
        config_path = Path(config_arg)
        for seed in args.seeds:
            try:
                metrics = benchmark_config(config_path, seed, output_root, args.force)
            except Exception as exc:  # noqa: BLE001 - benchmark suites should log every failed run.
                run_name = f"{config_path.stem}_seed{seed}"
                failure = {
                    "run_name": run_name,
                    "config_name": config_path.name,
                    "display_name": display_name_for_config(config_path.name),
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                print(f"FAILED run={run_name} error={type(exc).__name__}: {exc}")
                continue
            rows.append(metrics)
            print(
                f"run={metrics['run_name']} canonical_mse={metrics['canonical_mse']:.6g} "
                f"equivariance_error={metrics['equivariance_error']:.3e}"
            )

    per_run_fields = REQUIRED_METRIC_FIELDS + ["display_name", "last_checkpoint"]
    write_csv(output_root / "per_run_metrics.csv", rows, per_run_fields)

    summary_rows = summarize(rows)
    summary_fields = GROUP_FIELDS + ["n"]
    for field in NUMERIC_FIELDS:
        summary_fields.extend([f"{field}_mean", f"{field}_std"])
    write_csv(output_root / "summary_mean_std.csv", summary_rows, summary_fields)
    failure_path = output_root / "failed_runs.csv"
    if failures:
        write_csv(
            failure_path,
            failures,
            ["run_name", "config_name", "display_name", "seed", "error_type", "error_message", "traceback"],
        )
        print(f"wrote {failure_path} with {len(failures)} failed run(s)")
    elif failure_path.exists():
        failure_path.unlink()
        print(f"removed stale {failure_path}")
    print(f"wrote {output_root / 'per_run_metrics.csv'}")
    print(f"wrote {output_root / 'summary_mean_std.csv'}")


if __name__ == "__main__":
    main()
