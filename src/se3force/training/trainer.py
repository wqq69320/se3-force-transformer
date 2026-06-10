from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm

from se3force.data import build_dataloaders
from se3force.models.common import build_model, to_device
from se3force.training.checkpointing import save_checkpoint
from se3force.training.logging import write_json
from se3force.training.losses import force_mse_loss
from se3force.training.seed import set_seed


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, device, max_steps: int | None = None) -> dict:
    model.train()
    total = 0.0
    count = 0
    iterator = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(iterator, start=1):
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch["x"], batch["z"])
        loss = force_mse_loss(pred, batch["force"])
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        count += 1
        iterator.set_postfix(loss=total / count)
        if max_steps is not None and step >= max_steps:
            break
    return {"loss": total / max(1, count)}


@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = to_device(batch, device)
        pred = model(batch["x"], batch["z"])
        loss = force_mse_loss(pred, batch["force"])
        total += float(loss) * batch["x"].shape[0]
        count += batch["x"].shape[0]
    return {"mse": total / max(1, count)}


def train_from_config(config: dict) -> dict:
    set_seed(int(config.get("seed", 0)))
    device = torch.device(config.get("device", "cpu"))
    output_dir = Path(config.get("output_dir", "outputs/run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(config)
    model = build_model(config).to(device)
    train_cfg = config.get("training", {})
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 1))
    max_steps = train_cfg.get("max_steps_per_epoch")
    max_steps = int(max_steps) if max_steps is not None else None

    history = []
    best_val = float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, device, max_steps=max_steps)
        val_metrics = evaluate_loader(model, loaders["val"], device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        save_checkpoint(last_path, model, optimizer, epoch, row, config)
        if val_metrics["mse"] <= best_val:
            best_val = val_metrics["mse"]
            save_checkpoint(best_path, model, optimizer, epoch, row, config)
        print(f"epoch={epoch} train_loss={train_metrics['loss']:.6g} val_mse={val_metrics['mse']:.6g}")

    test_metrics = evaluate_loader(model, loaders["test"], device)
    metrics = {"history": history, "test": test_metrics, "best_checkpoint": str(best_path), "last_checkpoint": str(last_path)}
    write_json(output_dir / "metrics.json", metrics)
    return metrics
