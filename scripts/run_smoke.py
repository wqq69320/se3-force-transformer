#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from se3force.evaluation import evaluate_checkpoint
from se3force.training.trainer import load_config, train_from_config


def main() -> None:
    config = load_config("configs/small_cpu.yaml")
    metrics = train_from_config(config)
    eval_metrics = evaluate_checkpoint(config, metrics["best_checkpoint"], Path(config["output_dir"]) / "smoke_eval_metrics.json")
    print({"train": metrics["history"][-1], "eval": eval_metrics})


if __name__ == "__main__":
    main()
