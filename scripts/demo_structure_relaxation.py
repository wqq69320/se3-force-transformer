#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_molecular_config(args.config)
    loaders, _ = build_molecular_dataloaders(config)
    batch = batch_to_device(next(iter(loaders["test"])), config.get("device", "cpu"))
    model = build_molecular_model(config).to(config.get("device", "cpu"))
    load_checkpoint(args.checkpoint, model=model, map_location=config.get("device", "cpu"))
    pos = batch["pos"].detach().clone()
    records = []
    for step in range(args.steps):
        pos = pos.detach().requires_grad_(config.get("training", {}).get("mode") == "energy_force")
        out = model(pos, batch["z"], batch["mask"])
        forces = out["forces"].detach()
        pos = pos + args.step_size * forces
        records.append({"step": step, "force_norm": float(forces[batch["mask"]].norm()), "energy": None if out["energy"] is None else float(out["energy"].mean().detach())})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "relaxation.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([r["step"] for r in records], [r["force_norm"] for r in records])
    ax.set_xlabel("step")
    ax.set_ylabel("predicted force norm")
    ax.set_title("Short relaxation diagnostic")
    fig.tight_layout()
    fig.savefig(output / "relaxation_curve.png", dpi=160)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
