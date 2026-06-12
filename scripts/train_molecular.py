#!/usr/bin/env python3
from __future__ import annotations

import argparse

from se3force.training.molecular_trainer import load_molecular_config, train_molecular_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    metrics = train_molecular_from_config(load_molecular_config(args.config))
    print(f"best_checkpoint={metrics['best_checkpoint']}")
    print(f"force_mae={metrics['force_mae']:.6g}")


if __name__ == "__main__":
    main()
