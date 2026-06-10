#!/usr/bin/env python3
from __future__ import annotations

import argparse

from se3force.evaluation import evaluate_checkpoint
from se3force.training.trainer import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    metrics = evaluate_checkpoint(load_config(args.config), args.checkpoint, args.output)
    print(metrics)


if __name__ == "__main__":
    main()
