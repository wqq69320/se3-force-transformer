from __future__ import annotations

from pathlib import Path

DISPLAY_NAME_BY_CONFIG = {
    "baseline_mlp": "MLP",
    "baseline_vanilla_gt": "Vanilla GT",
    "baseline_egnn": "EGNN",
    "baseline_egnn_matched_params": "EGNN matched",
    "baseline_tfn": "TFNConv",
    "angular_force_se3_l2": "SE3 l=2",
    "angular_force_se3_l2_small": "SE3 l=2 small",
    "angular_force_se3_l3": "SE3 l=3",
    "ablation_lmax0": "SE3 l=0",
    "ablation_lmax1": "SE3 l=1",
    "ablation_lmax2": "SE3 l=2",
    "ablation_lmax3": "SE3 l=3",
    "ablation_no_attention": "SE3 no attention",
    "ablation_no_gate": "SE3 no gate",
}


def config_stem(config_name: str) -> str:
    return Path(config_name).stem


def display_name_for_config(config_name: str) -> str:
    stem = config_stem(config_name)
    return DISPLAY_NAME_BY_CONFIG.get(stem, stem)
