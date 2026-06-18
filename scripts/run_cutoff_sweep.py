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
from se3force.training.molecular_overrides import add_molecular_override_args, apply_molecular_overrides
from se3force.training.molecular_trainer import load_molecular_config, train_molecular_from_config

GROUP_FIELDS = [
    "config_name",
    "model_name",
    "model_class",
    "backbone_class",
    "architecture_signature",
    "dataset_name",
    "molecule_name",
    "data_source_type",
    "is_fake_or_synthetic",
    "is_real_rmd17",
    "training_mode",
    "split_type",
]

FIELDS = [
    "cutoff_radius",
    "force_mae",
    "force_rmse",
    "rotated_force_mae",
    "force_vector_l2_mae",
    "force_vector_l2_rmse",
    "equivariance_error",
    "force_equivariance_error",
    "edge_count_mean",
    "edge_count_max",
    "average_neighbors",
    "graph_build_time_sec",
    "runtime_per_batch_sec",
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fieldnames})


def csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def complete(path: Path, run_name: str, seed: int) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if not all(field in data for field in REQUIRED_METRIC_FIELDS + MOLECULAR_REQUIRED_FIELDS):
        return False
    return data.get("run_name") == run_name and int(data.get("seed", -1)) == seed


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in GROUP_FIELDS) + (float(row["cutoff_radius"]),)
        grouped.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(grouped.items()):
        out_row = {field: value for field, value in zip(GROUP_FIELDS, key[:-1])}
        out_row["n"] = len(group)
        for field in FIELDS:
            values = [float(item[field]) for item in group if item.get(field) not in (None, "", "nan")]
            out_row[f"{field}_mean"] = mean(values) if values else ""
            out_row[f"{field}_std"] = stdev(values) if len(values) > 1 else (0.0 if values else "")
        out.append(out_row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cutoffs", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    add_molecular_override_args(parser)
    args = parser.parse_args()

    output_root = Path(args.output)
    rows = []
    failures = []
    for cutoff in args.cutoffs:
        for seed in args.seeds:
            run_name = f"{Path(args.config).stem}_cutoff{cutoff:g}_seed{seed}"
            run_dir = output_root / "runs" / run_name
            metrics_path = run_dir / "metrics.json"
            try:
                if not args.force and complete(metrics_path, run_name, seed):
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    print(f"skipping completed run: {run_name}")
                else:
                    config = copy.deepcopy(load_molecular_config(args.config))
                    config.setdefault("dataset", {})["cutoff_radius"] = cutoff
                    config.setdefault("model", {})["cutoff_radius"] = cutoff
                    config["seed"] = seed
                    config["run_name"] = run_name
                    config["output_dir"] = str(run_dir)
                    config.setdefault("dataset", {})["seed"] = seed
                    config = apply_molecular_overrides(config, args)
                    config.setdefault("dataset", {})["cutoff_radius"] = cutoff
                    config.setdefault("model", {})["cutoff_radius"] = cutoff
                    config["seed"] = seed
                    config["run_name"] = run_name
                    config["output_dir"] = str(run_dir)
                    config.setdefault("dataset", {})["seed"] = seed
                    metrics = train_molecular_from_config(config)
                rows.append(metrics)
                print(f"run={run_name} cutoff={metrics['cutoff_radius']} force_mae={metrics['force_mae']:.6g}")
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "run_name": run_name,
                        "cutoff_radius": cutoff,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"FAILED run={run_name} error={type(exc).__name__}: {exc}")
    per_fields = ["run_name", "seed"] + GROUP_FIELDS + ["parameter_count", "parameter_count_by_module", "dataset_path_basename", "num_frames_total", "num_frames_used"] + FIELDS
    write_csv(output_root / "per_run_metrics.csv", rows, per_fields)
    summary = summarize(rows)
    summary_fields = GROUP_FIELDS + ["n"] + [f"{field}_{suffix}" for field in FIELDS for suffix in ("mean", "std")]
    write_csv(output_root / "summary_mean_std.csv", summary, summary_fields)
    if failures:
        write_csv(output_root / "failed_runs.csv", failures, ["run_name", "cutoff_radius", "seed", "error_type", "error_message", "traceback"])
    print(f"wrote {output_root / 'summary_mean_std.csv'}")


if __name__ == "__main__":
    main()
