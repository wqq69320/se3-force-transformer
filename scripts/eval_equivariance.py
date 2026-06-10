#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from se3force.data import build_dataloaders
from se3force.evaluation.equivariance import model_equivariance_error
from se3force.models.common import build_model, to_device
from se3force.training.seed import set_seed
from se3force.training.trainer import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="se3_transformer")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config.setdefault("model", {})["name"] = args.model
    set_seed(int(config.get("seed", 0)))
    device = torch.device(config.get("device", "cpu"))
    model = build_model(config).to(device)
    loader = build_dataloaders(config)["test"]
    batch = to_device(next(iter(loader)), device)
    err = model_equivariance_error(model, batch["x"], batch["z"])
    print(f"model={args.model} equivariance_error={float(err):.8e}")


if __name__ == "__main__":
    main()
