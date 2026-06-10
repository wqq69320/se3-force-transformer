from __future__ import annotations

import time
from pathlib import Path

import torch

from se3force.data import build_dataloaders
from se3force.evaluation.equivariance import model_equivariance_error
from se3force.geometry.metrics import parameter_count
from se3force.geometry.rotations import apply_rotation, random_rotation_matrix, random_translation
from se3force.geometry.transforms import apply_transform
from se3force.models.common import build_model, to_device
from se3force.training.checkpointing import load_checkpoint
from se3force.training.logging import write_json
from se3force.training.losses import force_mse_loss


@torch.no_grad()
def evaluate_model(model, loader, device="cpu") -> dict:
    model.eval()
    canonical_loss = 0.0
    rotated_loss = 0.0
    equiv_errors = []
    runtime = 0.0
    count = 0
    for batch in loader:
        batch = to_device(batch, device)
        start = time.perf_counter()
        pred = model(batch["x"], batch["z"])
        runtime += time.perf_counter() - start
        canonical_loss += float(force_mse_loss(pred, batch["force"])) * batch["x"].shape[0]

        R = random_rotation_matrix(batch["x"].shape[0], device=batch["x"].device, dtype=batch["x"].dtype)
        t = random_translation(batch["x"].shape[0], device=batch["x"].device, dtype=batch["x"].dtype)
        x_rt = apply_transform(batch["x"], R, t)
        force_rt = apply_rotation(batch["force"], R)
        pred_rt = model(x_rt, batch["z"])
        rotated_loss += float(force_mse_loss(pred_rt, force_rt)) * batch["x"].shape[0]
        equiv_errors.append(float(model_equivariance_error(model, batch["x"], batch["z"])))
        count += batch["x"].shape[0]
    return {
        "canonical_mse": canonical_loss / max(1, count),
        "rotated_translated_mse": rotated_loss / max(1, count),
        "equivariance_error": sum(equiv_errors) / max(1, len(equiv_errors)),
        "parameter_count": parameter_count(model),
        "runtime_per_batch_sec": runtime / max(1, len(loader)),
    }


def evaluate_checkpoint(config: dict, checkpoint_path: str | Path, output_path: str | Path | None = None) -> dict:
    device = torch.device(config.get("device", "cpu"))
    model = build_model(config).to(device)
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    loaders = build_dataloaders(config)
    metrics = evaluate_model(model, loaders["test"], device=device)
    if output_path is None:
        output_path = Path(config.get("output_dir", "outputs/run")) / "eval_metrics.json"
    write_json(output_path, metrics)
    return metrics
