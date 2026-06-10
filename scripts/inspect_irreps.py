#!/usr/bin/env python3
from __future__ import annotations

import argparse

from se3force.geometry.irreps import build_hidden_irreps, spherical_harmonics_irreps
from se3force.training.trainer import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/small_cpu.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    model_cfg = config["model"]
    hidden = build_hidden_irreps(lmax=model_cfg["lmax"], channels_by_l=model_cfg["channels_by_l"])
    print(f"hidden={hidden} dim={hidden.dim}")
    print(f"spherical_harmonics={spherical_harmonics_irreps(model_cfg['lmax'])}")


if __name__ == "__main__":
    main()
