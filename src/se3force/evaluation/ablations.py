from __future__ import annotations

from pathlib import Path

from se3force.training.trainer import load_config, train_from_config


def run_ablation_configs(paths: list[str | Path]) -> list[dict]:
    results = []
    for path in paths:
        config = load_config(path)
        metrics = train_from_config(config)
        results.append({"config": str(path), "metrics": metrics})
    return results
