#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

Path("outputs/.mplconfig").mkdir(parents=True, exist_ok=True)
Path("outputs/.cache").mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(Path("outputs/.mplconfig").resolve())
os.environ["XDG_CACHE_HOME"] = str(Path("outputs/.cache").resolve())
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import torch

from se3force.data.dataset_registry import build_molecular_dataloaders
from se3force.evaluation.molecular_evaluate import batch_to_device
from se3force.models.molecular import build_molecular_model
from se3force.training.checkpointing import load_checkpoint
from se3force.training.molecular_trainer import load_molecular_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_molecular_config(args.config)
    loaders, _ = build_molecular_dataloaders(config)
    batch = next(iter(loaders["test"]))
    device = torch.device(config.get("device", "cpu"))
    model = build_molecular_model(config).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    batch = batch_to_device(batch, device)
    out = model(batch["pos"], batch["z"], batch["mask"])
    idx = min(args.frame, batch["pos"].shape[0] - 1)
    mask = batch["mask"][idx].detach().cpu()
    pos = batch["pos"][idx][mask].detach().cpu()
    force = out["forces"][idx][mask].detach().cpu()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=40)
    ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], force[:, 0], force[:, 1], force[:, 2], length=0.3)
    ax.set_title("Predicted molecular forces")
    fig.tight_layout()
    fig.savefig(output / "force_field.png", dpi=160)
    plt.close(fig)
    print(f"wrote {output / 'force_field.png'}")


if __name__ == "__main__":
    main()
