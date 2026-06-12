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
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dt-fs", "--dt_fs", dest="dt_fs", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_molecular_config(args.config)
    loaders, _ = build_molecular_dataloaders(config)
    batch = batch_to_device(next(iter(loaders["test"])), config.get("device", "cpu"))
    model = build_molecular_model(config).to(config.get("device", "cpu"))
    load_checkpoint(args.checkpoint, model=model, map_location=config.get("device", "cpu"))
    pos = batch["pos"].detach().clone()
    vel = torch.zeros_like(pos)
    dt = args.dt_fs * 0.01
    records = []
    for step in range(args.steps):
        out = model(pos, batch["z"], batch["mask"])
        force = out["forces"].detach()
        vel = vel + dt * force
        pos = pos + dt * vel
        records.append({"step": step, "force_norm": float(force[batch["mask"]].norm()), "energy": float(out["energy"].mean().detach())})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "md_rollout.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([r["step"] for r in records], [r["energy"] for r in records])
    ax.set_xlabel("step")
    ax.set_ylabel("predicted energy")
    ax.set_title("Short rollout energy diagnostic")
    fig.tight_layout()
    fig.savefig(output / "energy_drift.png", dpi=160)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
