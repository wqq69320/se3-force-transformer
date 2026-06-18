#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

Path("outputs/.mplconfig").mkdir(parents=True, exist_ok=True)
Path("outputs/.cache").mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(Path("outputs/.mplconfig").resolve())
os.environ["XDG_CACHE_HOME"] = str(Path("outputs/.cache").resolve())
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(row: dict, field: str):
    for key in [field, f"{field}_mean"]:
        value = row.get(key)
        if value in (None, "", "nan"):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def model_family(row: dict) -> str:
    blob = " ".join(
        str(row.get(field, ""))
        for field in ["run_name", "config_name", "model_name", "model_class", "backbone_class", "architecture_signature"]
    ).lower()
    if "global_context_radial" in blob:
        return "global_context_radial"
    if "global_coeff" in blob or "global_invariant" in blob:
        return "global_coeff"
    return str(row.get("model_name", "unknown"))


def split_kind(row: dict) -> str:
    split = str(row.get("split_type", "")).lower()
    if "chrono" in split:
        return "chronological"
    if "random" in split:
        return "random"
    if "overfit" in split or "overfit" in str(row.get("config_name", "")).lower():
        return "overfit"
    return split or "unknown"


def frame_count(row: dict) -> int:
    name = str(row.get("config_name", ""))
    if "1k" in name:
        return 1000
    for field in ["num_frames_used", "num_train_frames"]:
        value = number(row, field)
        if value is not None:
            return int(round(value))
    if "1k" in name:
        return 1000
    for n in [128, 64, 32]:
        if f"overfit{n}" in name:
            return n
    return 0


def save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def line_by_frames(rows: list[dict], metric: str, ylabel: str, output: Path, train: bool = False) -> None:
    groups: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        value = number(row, metric)
        if value is None:
            continue
        groups.setdefault(model_family(row), []).append((frame_count(row), value))
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for family, points in sorted(groups.items()):
        points = sorted(points)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=family)
    ax.set_xlabel("train frames")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    if "equivariance" in metric:
        ax.set_yscale("log")
        ax.axhline(1e-5, color="#d62728", linestyle="--", linewidth=1.0)
    if "ratio" in metric:
        ax.axhline(0.75 if train else 1.0, color="#d62728", linestyle="--", linewidth=1.0)
    if "cosine" in metric:
        ax.axhline(0.7, color="#d62728", linestyle="--", linewidth=1.0)
    save(fig, output)


def bar_split(rows: list[dict], metric: str, ylabel: str, output: Path) -> None:
    pairs = []
    for row in rows:
        value = number(row, metric)
        if value is None:
            continue
        pairs.append((f"{split_kind(row)}\n{model_family(row)}", value))
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(pairs)), 4))
    ax.bar(range(len(pairs)), [v for _, v in pairs], color="#4c78a8")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([name for name, _ in pairs], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    save(fig, output)


def runtime_vs_error(rows: list[dict], output: Path) -> None:
    points = []
    for row in rows:
        x = number(row, "runtime_per_batch_sec")
        y = number(row, "force_vector_l2_mae")
        if x is not None and y is not None:
            points.append((x, y, model_family(row)))
    if not points:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for x, y, family in points:
        ax.scatter([x], [y], label=family)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8)
    ax.set_xlabel("runtime per batch (sec)")
    ax.set_ylabel("vector-L2 force MAE")
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--per-run", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    output = Path(args.output)
    overfit = [row for row in rows if split_kind(row) == "overfit"]
    gen = [row for row in rows if split_kind(row) in {"random", "chronological"}]
    line_by_frames(overfit, "train_eval_force_vector_l2_mae", "train vector-L2 force MAE", output / "overfit_train_vector_l2_vs_frames.png", train=True)
    line_by_frames(overfit, "train_eval_force_vector_l2_mae_improvement_vs_zero_pct", "improvement vs zero (%)", output / "overfit_improvement_vs_frames.png", train=True)
    line_by_frames(overfit, "train_eval_pred_to_target_force_norm_ratio", "pred/target norm ratio", output / "overfit_norm_ratio_vs_frames.png", train=True)
    line_by_frames(overfit, "train_eval_force_cosine_similarity_mean", "force cosine similarity", output / "overfit_cosine_vs_frames.png", train=True)
    bar_split(gen, "force_vector_l2_mae", "1k vector-L2 force MAE", output / "one_k_random_vs_chrono_vector_l2.png")
    bar_split(rows, "force_vector_l2_mae", "vector-L2 force MAE", output / "global_coeff_vs_global_context_radial.png")
    runtime_vs_error(rows, output / "runtime_vs_vector_l2.png")
    line_by_frames(rows, "force_equivariance_error", "force equivariance error", output / "equivariance_error_by_frames.png")
    print(f"wrote global context plots to {output}")


if __name__ == "__main__":
    main()
