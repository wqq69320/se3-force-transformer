#!/usr/bin/env python3
from __future__ import annotations

import argparse

from se3force.training.trainer import load_config, train_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    metrics = train_from_config(load_config(args.config))
    print(f"best_checkpoint={metrics['best_checkpoint']}")


if __name__ == "__main__":
    main()
