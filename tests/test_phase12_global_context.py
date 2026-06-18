import csv
import importlib.util
import sys
from pathlib import Path

import torch

from se3force.evaluation.metrics_schema import REQUIRED_METRIC_FIELDS
from se3force.evaluation.molecular_metrics import MOLECULAR_REQUIRED_FIELDS
from se3force.geometry.rotations import apply_rotation, random_rotation_matrix
from se3force.models.molecular import build_molecular_model
from se3force.training.molecular_trainer import molecular_losses


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(Path(path).parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def config_for(name: str = "global_context_radial") -> dict:
    return {
        "model": {
            "name": name,
            "max_atoms": 8,
            "hidden_dim": 24,
            "radial_num_basis": 6,
            "radial_hidden_dim": 32,
            "edge_mlp_hidden_dim": 32,
            "edge_mlp_layers": 1,
            "global_context_dim": 12,
            "global_hidden_dim": 24,
            "global_layers": 1,
            "atom_embedding_dim": 8,
            "pair_embedding_dim": 8,
            "graph_mode": "full",
        },
        "training": {"mode": "direct_force", "force_scale_normalization": "none", "force_loss_type": "mse"},
    }


def sample():
    torch.manual_seed(120)
    pos = torch.randn(2, 5, 3)
    z = torch.tensor([[1, 6, 7, 8, 1], [6, 6, 1, 8, 7]], dtype=torch.long)
    mask = torch.ones(2, 5, dtype=torch.bool)
    return pos, z, mask


def test_global_context_descriptor_invariant_to_rotation_translation():
    model = build_molecular_model(config_for())
    pos, z, mask = sample()
    rotation = random_rotation_matrix(2, dtype=pos.dtype)
    translation = torch.randn(2, 1, 3)
    descriptor = model.global_descriptor(pos, z, mask)
    transformed = model.global_descriptor(apply_rotation(pos, rotation) + translation, z, mask)
    assert torch.allclose(descriptor, transformed, atol=1e-5, rtol=1e-5)


def test_global_context_radial_equivariance_atom_sensitivity_and_training_step():
    model = build_molecular_model(config_for())
    pos, z, mask = sample()
    rotation = random_rotation_matrix(2, dtype=pos.dtype)
    translation = torch.randn(2, 1, 3)
    out = model(pos, z, mask)["forces"]
    out_rot = model(apply_rotation(pos, rotation) + translation, z, mask)["forces"]
    expected = apply_rotation(out, rotation)
    error = (out_rot - expected).norm() / expected.norm().clamp_min(1e-8)
    assert float(error) < 1e-5
    z_alt = torch.flip(z, dims=[1])
    assert float((model(pos, z, mask)["forces"] - model(pos, z_alt, mask)["forces"]).abs().max()) > 1e-8
    batch = {
        "pos": pos,
        "z": z,
        "mask": mask,
        "forces": torch.randn_like(pos),
        "energy": torch.full((pos.shape[0], 1), float("nan")),
    }
    loss, row = molecular_losses(out=model(pos, z, mask), batch=batch, config=config_for(), metadata={"force_rms": 1.0, "force_std": 1.0}, model=model)
    loss.backward()
    assert row["force_vector_l2_mae"] > 0
    assert any(param.grad is not None for param in model.parameters())


def test_phase12_runner_accepts_model_and_writes_schema(tmp_path):
    runner = load_script("run_rmd17_overfit_diagnostic_phase12", "scripts/run_rmd17_overfit_diagnostic.py")
    cfg = {"model": {}}
    runner.apply_model_choice(cfg, "global_context_radial")
    assert cfg["model"]["name"] == "global_context_radial"
    assert runner.diagnostic_type_for(cfg) == "equivariant_global_context_radial"
    row = {field: "" for field in REQUIRED_METRIC_FIELDS + MOLECULAR_REQUIRED_FIELDS}
    row.update({"run_name": "r", "seed": 0, "uses_global_context": True, "global_context_dim": 12, "global_context_type": "test"})
    path = tmp_path / "schema.csv"
    runner.write_csv(path, [row], list(row))
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "uses_global_context" in header
    assert "global_context_dim" in header
    assert "global_context_type" in header


def test_phase12_summary_groups_seed_dependent_force_scale():
    runner = load_script("run_rmd17_overfit_diagnostic_phase12_summary", "scripts/run_rmd17_overfit_diagnostic.py")
    base = {
        field: ""
        for field in runner.GROUP_FIELDS + runner.NUMERIC_FIELDS
    }
    base.update(
        {
            "config_name": "rmd17_aspirin_1k_random_global_coeff.yaml",
            "model_name": "global_coeff",
            "model_class": "GlobalInvariantCoefficientForceModel",
            "backbone_class": "global_invariant_coefficients",
            "dataset_name": "rmd17",
            "molecule_name": "aspirin",
            "data_source_type": "local_rmd17_npz",
            "is_fake_or_synthetic": False,
            "is_real_rmd17": True,
            "diagnostic_type": "random_split_global_memorizer",
            "training_mode": "direct_force",
            "split_type": "random",
            "force_scale_normalization": "train_force_rms",
            "force_loss_type": "mse",
            "learning_rate": 0.001,
            "gradient_clip_norm": 100.0,
            "output_head_init_scale": 1.0,
            "force_output_scale": 1.0,
            "learnable_force_output_scale": False,
            "initial_force_output_scale": 1.0,
            "force_output_scale_regularization": 0.0,
            "graph_mode": "full",
            "actual_hidden_irreps": "global_invariant_scalars_to_vectors",
            "uses_non_scalar_hidden": False,
            "force_head_type": "global_invariant_edge_coefficients",
            "uses_atom_pair_embedding": True,
            "uses_global_context": False,
            "global_context_type": "",
            "weight_decay": 0.0,
        }
    )
    rows = []
    for seed, scale in [(0, 29.4), (1, 29.5), (2, 29.6)]:
        row = dict(base)
        row.update({"seed": seed, "fixed_force_scale_value": scale, "force_vector_l2_mae": 20.0 + seed})
        rows.append(row)
    summary = runner.summarize(rows)
    assert len(summary) == 1
    assert summary[0]["n"] == 3
    assert summary[0]["fixed_force_scale_value_mean"] == 29.5


def fake_row(
    config_name: str,
    split_type: str,
    frames: int,
    model_name: str = "global_coeff",
    zero: float = 10.0,
    mean: float = 10.0,
    train_zero: float = 60.0,
    ratio: float = 0.9,
    cosine: float = 0.9,
) -> dict:
    return {
        "config_name": config_name,
        "model_name": model_name,
        "model_class": "GlobalInvariantCoefficientForceModel" if model_name == "global_coeff" else "GlobalContextRadialForceModel",
        "backbone_class": "global_invariant_coefficients" if model_name == "global_coeff" else "global_context_radial_pair",
        "split_type": split_type,
        "num_train_frames_mean": str(frames),
        "n": "3",
        "data_source_type": "local_rmd17_npz",
        "force_vector_l2_mae_mean": "10",
        "force_vector_l2_mae_improvement_vs_zero_pct_mean": str(zero),
        "force_vector_l2_mae_improvement_vs_mean_pct_mean": str(mean),
        "train_eval_force_vector_l2_mae_mean": "10",
        "train_eval_force_vector_l2_mae_improvement_vs_zero_pct_mean": str(train_zero),
        "train_eval_force_vector_l2_mae_improvement_vs_mean_pct_mean": str(train_zero),
        "train_eval_pred_to_target_force_norm_ratio_mean": str(ratio),
        "train_eval_force_cosine_similarity_mean_mean": str(cosine),
        "pred_to_target_force_norm_ratio_mean": str(ratio),
        "force_cosine_similarity_mean_mean": str(cosine),
        "force_equivariance_error_mean": "1e-6",
        "equivariance_error_mean": "1e-6",
        "runtime_per_batch_sec_mean": "0.1",
    }


def test_phase12_report_conclusion_logic():
    report = load_script("make_global_context_report", "scripts/make_global_context_report.py")
    overfit = [fake_row("overfit64", "train_only_overfit", 64), fake_row("overfit128", "train_only_overfit", 128, train_zero=35)]
    random_fail = [fake_row("1k_random", "random", 700, zero=1, mean=1)]
    assert "generalization remains weak" in report.conclusion(overfit, random_fail, [])
    random_ok = [fake_row("1k_random", "random", 700, zero=7, mean=6)]
    chrono_fail = [fake_row("1k_chrono", "chronological", 700, zero=0, mean=0)]
    assert "trajectory distribution shift" in report.conclusion(overfit, random_ok, chrono_fail)
    chrono_ok = [fake_row("1k_chrono", "chronological", 700, zero=2, mean=2)]
    assert "viable rMD17 learning path" in report.conclusion(overfit, random_ok, chrono_ok)


def test_phase12_plotting_with_minimal_csv(tmp_path):
    plotter = load_script("plot_global_context_results", "scripts/plot_global_context_results.py")
    rows = [
        fake_row("rmd17_aspirin_overfit64_global_coeff.yaml", "train_only_overfit", 64),
        fake_row("rmd17_aspirin_1k_random_global_coeff.yaml", "random", 700, zero=7, mean=6),
    ]
    csv_path = tmp_path / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "plots"
    old_argv = sys.argv
    sys.argv = ["plot_global_context_results.py", "--input", str(csv_path), "--output", str(output)]
    try:
        plotter.main()
    finally:
        sys.argv = old_argv
    assert (output / "overfit_train_vector_l2_vs_frames.png").exists()
    assert (output / "one_k_random_vs_chrono_vector_l2.png").exists()
