from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from se3force.data.dataset_registry import build_molecular_dataloaders
from se3force.evaluation.molecular_evaluate import batch_to_device, evaluate_molecular_model, molecular_standard_metrics
from se3force.models.molecular import build_molecular_model
from se3force.training.checkpointing import load_checkpoint, save_checkpoint
from se3force.training.logging import write_json
from se3force.training.seed import set_seed


def load_molecular_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(path)
    config.setdefault("config_name", Path(path).name)
    return config


def molecular_losses(out: dict, batch: dict, config: dict) -> tuple[torch.Tensor, dict]:
    mask3 = batch["mask"].unsqueeze(-1).expand_as(batch["forces"])
    diff = out["forces"] - batch["forces"]
    force_mse = (diff[mask3] ** 2).mean()
    force_mae = diff[mask3].abs().mean()
    loss = float(config.get("training", {}).get("lambda_force", 1.0)) * force_mse
    energy_mse = None
    e_mask = torch.isfinite(batch["energy"].squeeze(-1))
    if e_mask.any() and out.get("energy") is not None:
        e_diff = out["energy"][e_mask] - batch["energy"][e_mask]
        energy_mse = (e_diff * e_diff).mean()
        loss = loss + float(config.get("training", {}).get("lambda_energy", 0.0)) * energy_mse
    return loss, {
        "loss": float(loss.detach()),
        "force_mse": float(force_mse.detach()),
        "force_mae": float(force_mae.detach()),
        "energy_mse": None if energy_mse is None else float(energy_mse.detach()),
    }


def train_one_epoch_molecular(model, loader, optimizer, config, device) -> dict:
    model.train()
    total = {"loss": 0.0, "force_mse": 0.0, "force_mae": 0.0}
    count = 0
    max_steps = config.get("training", {}).get("max_steps_per_epoch")
    max_steps = int(max_steps) if max_steps is not None else None
    iterator = tqdm(loader, desc="molecular-train", leave=False)
    for step, raw_batch in enumerate(iterator, start=1):
        batch = batch_to_device(raw_batch, device)
        if config.get("training", {}).get("mode", "direct_force") == "energy_force":
            batch["pos"] = batch["pos"].detach().clone().requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["pos"], batch["z"], batch["mask"])
        loss, row = molecular_losses(out, batch, config)
        loss.backward()
        grad_clip = config.get("training", {}).get("gradient_clip")
        if grad_clip is not None:
            clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        for key in total:
            total[key] += row[key]
        count += 1
        iterator.set_postfix(loss=total["loss"] / count, force_mae=total["force_mae"] / count)
        if max_steps is not None and step >= max_steps:
            break
    return {key: value / max(1, count) for key, value in total.items()}


def train_molecular_from_config(config: dict) -> dict:
    set_seed(int(config.get("seed", 0)))
    device = torch.device(config.get("device", "cpu"))
    output_dir = Path(config.get("output_dir", "outputs/molecular_run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders, metadata = build_molecular_dataloaders(config)
    model = build_molecular_model(config).to(device)
    train_cfg = config.get("training", {})
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 1))
    history = []
    best_val = float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch_molecular(model, loaders["train"], optimizer, config, device)
        val_metrics = evaluate_molecular_model(model, loaders["val"], config, metadata, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        save_checkpoint(last_path, model, optimizer, epoch, row, config)
        if val_metrics["force_mae"] <= best_val:
            best_val = val_metrics["force_mae"]
            save_checkpoint(best_path, model, optimizer, epoch, row, config)
        print(f"epoch={epoch} loss={train_metrics['loss']:.6g} val_force_mae={val_metrics['force_mae']:.6g}")

    load_checkpoint(best_path, model=model, map_location=device)
    eval_metrics = evaluate_molecular_model(model, loaders["test"], config, metadata, device)
    metrics = molecular_standard_metrics(
        config,
        loaders,
        metadata,
        best_path,
        history[-1]["train"]["loss"],
        history[-1]["val"]["force_rmse"] ** 2,
        eval_metrics,
        extra={"history": history, "last_checkpoint": str(last_path), "metadata": metadata},
    )
    write_json(output_dir / "metrics.json", metrics)
    return metrics
