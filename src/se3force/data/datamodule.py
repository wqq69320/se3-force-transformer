from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split

from .angular_potential import AngularPotentialDataset
from .central_force import CentralForceDataset


def build_dataset(config: dict):
    data_cfg = dict(config.get("dataset", {}))
    name = data_cfg.pop("name", "angular")
    if name == "central":
        return CentralForceDataset(**data_cfg)
    if name == "angular":
        return AngularPotentialDataset(**data_cfg)
    raise ValueError(f"unknown dataset: {name}")


def build_dataloaders(config: dict):
    dataset = build_dataset(config)
    train_cfg = config.get("training", {})
    n = len(dataset)
    test_n = max(1, int(n * float(train_cfg.get("test_fraction", 0.2))))
    val_n = max(1, int(n * float(train_cfg.get("val_fraction", 0.2))))
    train_n = n - val_n - test_n
    if train_n <= 0:
        raise ValueError("dataset too small for requested splits")
    gen = torch.Generator().manual_seed(int(config.get("seed", 0)))
    train_set, val_set, test_set = random_split(dataset, [train_n, val_n, test_n], generator=gen)
    batch_size = int(train_cfg.get("batch_size", 8))
    return {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False),
    }
