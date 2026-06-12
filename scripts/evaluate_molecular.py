#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from se3force.data.dataset_registry import build_molecular_dataloaders
from se3force.evaluation.molecular_evaluate import evaluate_molecular_checkpoint
from se3force.training.molecular_trainer import load_molecular_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_molecular_config(args.config)
    loaders, metadata = build_molecular_dataloaders(config)
    output = args.output or str(Path(config.get("output_dir", "outputs/molecular_run")) / "eval_metrics.json")
    metrics = evaluate_molecular_checkpoint(config, args.checkpoint, loaders, metadata, output)
    print(metrics)


if __name__ == "__main__":
    main()
