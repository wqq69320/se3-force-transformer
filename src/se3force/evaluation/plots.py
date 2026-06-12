from __future__ import annotations

from pathlib import Path

import os

Path("outputs/.mplconfig").mkdir(parents=True, exist_ok=True)
Path("outputs/.cache").mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(Path("outputs/.mplconfig").resolve())
os.environ["XDG_CACHE_HOME"] = str(Path("outputs/.cache").resolve())
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt


def plot_loss(history: list[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    epochs = [row["epoch"] for row in history]
    train = [row["train"]["loss"] for row in history]
    val = [row["val"]["mse"] for row in history]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(epochs, train, label="train")
    ax.plot(epochs, val, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
