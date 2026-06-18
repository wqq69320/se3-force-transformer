from __future__ import annotations

import argparse
import copy
import os


def add_molecular_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--molecule", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--split-type",
        choices=["random", "chronological", "first_middle_last", "train_only_overfit", "overfit"],
        default=None,
    )
    parser.add_argument("--device", default=None)


def apply_molecular_overrides(config: dict, args) -> dict:
    config = copy.deepcopy(config)
    dataset = config.setdefault("dataset", {})
    data_path = getattr(args, "data_path", None) or os.environ.get("SE3FORCE_MOLECULAR_DATA_PATH")
    if data_path:
        dataset["path"] = data_path
    molecule = getattr(args, "molecule", None)
    if molecule:
        dataset["molecule"] = molecule
    seed = getattr(args, "seed", None)
    if seed is not None:
        config["seed"] = int(seed)
        dataset["seed"] = int(seed)
    max_frames = getattr(args, "max_frames", None)
    if max_frames is not None:
        dataset["max_frames"] = int(max_frames)
        _resize_splits_for_max_frames(dataset, int(max_frames))
    split_type = getattr(args, "split_type", None)
    if split_type:
        dataset["split_type"] = split_type
    device = getattr(args, "device", None)
    if device:
        config["device"] = device
    output = getattr(args, "output", None)
    if output:
        config["output_dir"] = output
    return config


def _resize_splits_for_max_frames(dataset: dict, max_frames: int) -> None:
    train = int(dataset.get("train_size", max(1, int(0.7 * max_frames))))
    val = int(dataset.get("val_size", max(1, int(0.15 * max_frames))))
    test = int(dataset.get("test_size", max(1, max_frames - train - val)))
    total = max(1, train + val + test)
    if total != max_frames:
        train_ratio = train / total
        val_ratio = val / total
        train = max(1, int(round(max_frames * train_ratio)))
        val = max(1, int(round(max_frames * val_ratio)))
        test = max(1, max_frames - train - val)
    while train + val + test > max_frames:
        if train >= val and train >= test and train > 1:
            train -= 1
        elif val >= test and val > 1:
            val -= 1
        elif test > 1:
            test -= 1
        else:
            break
    while train + val + test < max_frames:
        train += 1
    dataset["train_size"] = train
    dataset["val_size"] = val
    dataset["test_size"] = test
