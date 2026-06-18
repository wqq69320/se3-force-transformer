#!/usr/bin/env python3
from __future__ import annotations

import argparse

from se3force.training.molecular_trainer import load_molecular_config, train_molecular_from_config
from se3force.training.molecular_overrides import add_molecular_override_args, apply_molecular_overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    add_molecular_override_args(parser)
    args = parser.parse_args()
    config = apply_molecular_overrides(load_molecular_config(args.config), args)
    metrics = train_molecular_from_config(config)
    print(f"best_checkpoint={metrics['best_checkpoint']}")
    print(f"force_mae={metrics['force_mae']:.6g}")


if __name__ == "__main__":
    main()
