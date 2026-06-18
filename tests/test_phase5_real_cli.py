import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from se3force.data.dataset_registry import build_molecular_dataloaders
from se3force.data.molecular import MolecularListDataset, MolecularMetadata, molecular_collate, split_indices
from se3force.evaluation.molecular_evaluate import evaluate_molecular_checkpoint, evaluate_molecular_model
from se3force.geometry.rotations import apply_rotation, random_rotation_matrix
from se3force.models.equivariant.se3_force_transformer import SE3ForceTransformer
from se3force.models.molecular import build_molecular_model, molecular_model_identity
from se3force.training.molecular_overrides import add_molecular_override_args, apply_molecular_overrides
from se3force.training.molecular_trainer import load_molecular_config, molecular_losses, train_molecular_from_config


def load_real_config(name: str) -> dict:
    return load_molecular_config(Path("configs/molecular/real") / name)


def test_molecular_cli_overrides_update_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    add_molecular_override_args(parser)
    args = parser.parse_args(
        [
            "--data-path",
            "data/rmd17/aspirin.npz",
            "--molecule",
            "aspirin",
            "--seed",
            "3",
            "--max-frames",
            "20",
            "--split-type",
            "chronological",
            "--device",
            "cpu",
            "--output",
            "outputs/test",
        ]
    )
    config = {"dataset": {"name": "rmd17", "train_size": 70, "val_size": 15, "test_size": 15}}
    out = apply_molecular_overrides(config, args)
    assert out["dataset"]["path"] == "data/rmd17/aspirin.npz"
    assert out["dataset"]["molecule"] == "aspirin"
    assert out["seed"] == 3
    assert out["dataset"]["max_frames"] == 20
    assert out["dataset"]["split_type"] == "chronological"
    assert out["device"] == "cpu"
    assert out["output_dir"] == "outputs/test"
    assert out["dataset"]["train_size"] + out["dataset"]["val_size"] + out["dataset"]["test_size"] <= 20


def test_chronological_split_policy_preserves_order():
    splits = split_indices(10, 5, 2, 3, seed=123, split_type="chronological")
    assert splits["train"] == [0, 1, 2, 3, 4]
    assert splits["val"] == [5, 6]
    assert splits["test"] == [7, 8, 9]


def test_train_only_overfit_split_reuses_same_frames():
    splits = split_indices(4, 4, 4, 4, seed=123, split_type="train_only_overfit")
    assert splits["train"] == [0, 1, 2, 3]
    assert splits["val"] == [0, 1, 2, 3]
    assert splits["test"] == [0, 1, 2, 3]


def test_cutoff_sweep_summary_aggregates_rows():
    spec = importlib.util.spec_from_file_location("run_cutoff_sweep", Path("scripts/run_cutoff_sweep.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    base = {
        "config_name": "rmd17_tiny_direct_se3.yaml",
        "model_name": "se3_transformer",
        "model_class": "MolecularSE3ForceTransformer",
        "backbone_class": "se3_scalar_attention_kernel",
        "architecture_signature": "sig",
        "dataset_name": "rmd17",
        "molecule_name": "aspirin",
        "data_source_type": "local_rmd17_npz",
        "is_fake_or_synthetic": False,
        "is_real_rmd17": True,
        "training_mode": "direct_force",
        "split_type": "chronological",
        "cutoff_radius": 3.0,
        "rotated_force_mae": 1.1,
        "force_vector_l2_mae": 1.5,
        "force_vector_l2_rmse": 2.5,
        "equivariance_error": 1e-7,
        "force_equivariance_error": 1e-7,
        "edge_count_max": 10,
        "graph_build_time_sec": 0.1,
        "runtime_per_batch_sec": 0.2,
    }
    rows = [
        {**base, "force_mae": 1.0, "force_rmse": 2.0, "edge_count_mean": 4, "average_neighbors": 2},
        {**base, "force_mae": 3.0, "force_rmse": 4.0, "edge_count_mean": 8, "average_neighbors": 4, "graph_build_time_sec": 0.3, "runtime_per_batch_sec": 0.4},
    ]
    summary = module.summarize(rows)
    assert summary[0]["cutoff_radius_mean"] == 3.0
    assert summary[0]["n"] == 2
    assert summary[0]["force_mae_mean"] == 2.0
    assert summary[0]["edge_count_mean_mean"] == 6.0
    for field in [
        "config_name",
        "model_name",
        "model_class",
        "backbone_class",
        "architecture_signature",
        "dataset_name",
        "molecule_name",
        "data_source_type",
        "is_fake_or_synthetic",
        "is_real_rmd17",
        "training_mode",
        "split_type",
        "cutoff_radius_std",
        "force_vector_l2_mae_mean",
        "equivariance_error_mean",
        "force_equivariance_error_mean",
        "edge_count_max_mean",
        "runtime_per_batch_sec_std",
    ]:
        assert field in summary[0]


def test_molecular_benchmark_summary_keeps_real_schema_fields():
    spec = importlib.util.spec_from_file_location("run_molecular_benchmark_suite", Path("scripts/run_molecular_benchmark_suite.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rows = [
        {
            "config_name": "rmd17_tiny_direct_se3.yaml",
            "model_name": "se3_transformer",
            "dataset_name": "rmd17",
            "molecule_name": "aspirin",
            "training_mode": "direct_force",
            "split_type": "chronological",
            "force_unit": "kcal/mol/A",
            "energy_unit": "kcal/mol",
            "seed": 0,
            "force_mae": 1.0,
            "force_vector_l2_mae": 1.4,
            "force_vector_l2_rmse": 1.8,
            "zero_force_mae": 2.0,
            "zero_force_rmse": 2.5,
            "zero_force_vector_l2_mae": 3.0,
            "zero_force_vector_l2_rmse": 3.5,
            "mean_force_mae": 1.5,
            "mean_force_rmse": 2.0,
            "mean_force_vector_l2_mae": 2.5,
            "mean_force_vector_l2_rmse": 3.0,
            "force_mae_improvement_vs_zero_pct": 50.0,
            "force_mae_improvement_vs_mean_pct": 33.3,
            "energy_invariance_error": 1e-8,
            "energy_mae_raw": 1.0,
            "energy_rmse_raw": 1.0,
            "energy_mae_centered": 1.0,
            "energy_rmse_centered": 1.0,
            "energy_train_mean": -10.0,
            "energy_train_std": 2.0,
            "energy_centering": True,
            "energy_standardization": False,
            "energy_loss_on_centered": True,
            "force_equivariance_error": 1e-7,
            "model_class": "MolecularSE3ForceTransformer",
            "backbone_class": "se3_scalar_attention_kernel",
            "architecture_signature": "se3_l2|attention=True",
            "lmax": 2,
            "hidden_irreps": "64x0",
            "use_attention": True,
            "use_gate": True,
            "data_source_type": "fake_rmd17_npz_smoke",
            "is_fake_or_synthetic": True,
            "is_real_rmd17": False,
            "dataset_path_basename": "fake.npz",
            "num_atoms_mean": 6.0,
            "num_atoms_max": 6,
            "num_frames_total": 12,
            "num_frames_used": 10,
            "num_train_frames": 7,
            "num_val_frames": 2,
            "num_test_frames": 1,
            "num_train_batches": 4,
            "num_val_batches": 1,
            "num_test_batches": 1,
            "batch_size": 2,
            "force_loss_weight": 1.0,
            "energy_loss_weight": 0.0,
            "val_force_mae_epoch1": 2.0,
            "val_force_mae_final": 1.0,
            "val_force_rmse_final": 1.2,
            "val_force_vector_l2_mae_epoch1": 3.0,
            "val_force_vector_l2_mae_final": 2.0,
            "val_force_mae_decreased": True,
            "learning_established": True,
        }
    ]
    summary = module.summarize(rows)
    assert summary[0]["num_frames_total_mean"] == 12.0
    assert summary[0]["num_atoms_mean_mean"] == 6.0
    assert summary[0]["energy_invariance_error_mean"] == 1e-8
    assert summary[0]["force_equivariance_error_mean"] == 1e-7
    assert summary[0]["force_loss_weight_mean"] == 1.0
    assert summary[0]["num_frames_used_mean"] == 10.0
    assert summary[0]["data_source_type"] == "fake_rmd17_npz_smoke"
    assert summary[0]["force_vector_l2_mae_mean"] == 1.4
    assert summary[0]["zero_force_mae_mean"] == 2.0
    assert summary[0]["num_train_frames_mean"] == 7.0
    assert summary[0]["energy_centering"] is True


def test_cutoff_plot_and_report_generation_are_schema_robust(tmp_path):
    row = {
        "config_name": "rmd17_tiny_direct_se3.yaml",
        "model_name": "se3_transformer",
        "model_class": "MolecularSE3ForceTransformer",
        "backbone_class": "se3_scalar_attention_kernel",
        "architecture_signature": "sig",
        "dataset_name": "rmd17",
        "molecule_name": "aspirin",
        "data_source_type": "local_rmd17_npz",
        "is_fake_or_synthetic": "False",
        "is_real_rmd17": "True",
        "training_mode": "direct_force",
        "split_type": "chronological",
        "n": "3",
        "cutoff_radius_mean": "3.0",
        "cutoff_radius_std": "0.0",
        "force_mae_mean": "1.0",
        "force_mae_std": "0.1",
        "force_rmse_mean": "2.0",
        "force_rmse_std": "0.1",
        "rotated_force_mae_mean": "1.1",
        "rotated_force_mae_std": "0.1",
        "force_vector_l2_mae_mean": "1.5",
        "force_vector_l2_mae_std": "0.1",
        "force_vector_l2_rmse_mean": "2.5",
        "force_vector_l2_rmse_std": "0.1",
        "equivariance_error_mean": "1e-7",
        "equivariance_error_std": "0.0",
        "force_equivariance_error_mean": "1e-7",
        "force_equivariance_error_std": "0.0",
        "average_neighbors_mean": "4.0",
        "average_neighbors_std": "0.2",
        "edge_count_mean_mean": "80",
        "edge_count_mean_std": "2",
        "edge_count_max_mean": "100",
        "edge_count_max_std": "1",
        "graph_build_time_sec_mean": "0.01",
        "graph_build_time_sec_std": "0.001",
        "runtime_per_batch_sec_mean": "0.02",
        "runtime_per_batch_sec_std": "0.001",
        "zero_force_mae_mean": "2.0",
        "mean_force_mae_mean": "1.5",
        "num_train_frames_mean": "700",
        "num_val_frames_mean": "150",
        "num_test_frames_mean": "150",
        "batch_size_mean": "4",
        "energy_centering": "True",
        "energy_standardization": "False",
        "val_force_mae_epoch1_mean": "2.0",
        "val_force_mae_final_mean": "1.0",
        "val_force_vector_l2_mae_epoch1_mean": "3.0",
        "val_force_vector_l2_mae_final_mean": "1.5",
        "learning_established_mean": "1.0",
    }
    summary = tmp_path / "summary_mean_std.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    plot_dir = tmp_path / "plots"
    report = tmp_path / "report.md"
    subprocess.run([sys.executable, "scripts/plot_molecular_real_results.py", "--input", str(summary), "--output", str(plot_dir)], check=True)
    subprocess.run([sys.executable, "scripts/make_molecular_real_report.py", "--input", str(summary), "--output", str(report), "--data-path", "data/rmd17/rmd17_aspirin.npz", "--molecule", "aspirin"], check=True)
    assert (plot_dir / "cutoff_vs_force_mae.png").exists()
    assert (plot_dir / "cutoff_vs_force_vector_l2_mae.png").exists()
    text = report.read_text(encoding="utf-8")
    assert "Cutoff rows should be interpreted" in text
    assert "Architecture superiority is not established by this run." in text


def test_model_factory_records_distinct_identity_metadata():
    configs = {
        "egnn": load_real_config("rmd17_tiny_direct_egnn.yaml"),
        "tfn": load_real_config("rmd17_tiny_direct_tfn.yaml"),
        "se3": load_real_config("rmd17_tiny_direct_se3.yaml"),
        "se3_energy": load_real_config("rmd17_tiny_energy_force_se3.yaml"),
    }
    identities = {name: molecular_model_identity(build_molecular_model(config)) for name, config in configs.items()}
    assert identities["egnn"]["model_class"] == "MolecularEGNN"
    assert identities["tfn"]["model_class"] == "MolecularTFNConv"
    assert identities["se3"]["model_class"] == "MolecularSE3ForceTransformer"
    assert identities["se3_energy"]["training_mode"] == "energy_force"
    assert identities["egnn"]["architecture_signature"] != identities["se3"]["architecture_signature"]
    assert identities["tfn"]["architecture_signature"] != identities["se3"]["architecture_signature"]
    assert identities["se3"]["use_attention"] is True
    assert identities["tfn"]["use_attention"] is False
    counts = {name: identity["parameter_count_by_module"]["_total"] for name, identity in identities.items()}
    assert len({counts["egnn"], counts["tfn"], counts["se3"]}) == 3


def test_output_head_init_scale_is_recorded_in_model_identity():
    config = load_real_config("rmd17_tiny_direct_se3.yaml")
    config["model"]["output_head_init_scale"] = 0.25
    model = build_molecular_model(config)
    identity = molecular_model_identity(model)
    assert model.output_head_init_scale == 0.25
    assert "force_head_scale=0.25" in identity["architecture_signature"]


def test_scalar_force_output_scaling_preserves_equivariance():
    torch.manual_seed(11)
    config = load_real_config("rmd17_tiny_direct_se3.yaml")
    config["model"]["force_output_scale"] = 10.0
    model = build_molecular_model(config)
    pos = torch.randn(2, 5, 3)
    z = torch.tensor([[1, 6, 7, 8, 1], [6, 6, 1, 8, 7]], dtype=torch.long)
    mask = torch.ones(2, 5, dtype=torch.bool)
    rotation = random_rotation_matrix(2, dtype=pos.dtype)
    out = model(pos, z, mask)["forces"]
    out_rot = model(apply_rotation(pos, rotation), z, mask)["forces"]
    expected = apply_rotation(out, rotation)
    error = (out_rot - expected).norm() / expected.norm().clamp_min(1e-8)
    assert float(error) < 1e-5


def test_learnable_force_output_scale_receives_gradients():
    torch.manual_seed(12)
    config = load_real_config("rmd17_tiny_direct_se3.yaml")
    config["model"]["learnable_force_output_scale"] = True
    config["model"]["initial_force_output_scale"] = 3.0
    model = build_molecular_model(config)
    pos = torch.randn(1, 5, 3)
    z = torch.tensor([[1, 6, 7, 8, 1]], dtype=torch.long)
    mask = torch.ones(1, 5, dtype=torch.bool)
    loss = model(pos, z, mask)["forces"].pow(2).mean()
    loss.backward()
    assert model.force_output_log_scale is not None
    assert model.force_output_log_scale.grad is not None
    assert float(model.force_output_log_scale.grad.abs()) > 0


def test_radial_pair_baseline_is_equivariant():
    torch.manual_seed(13)
    config = load_real_config("rmd17_tiny_direct_se3.yaml")
    config["model"]["name"] = "radial"
    config["model"]["force_output_scale"] = 5.0
    model = build_molecular_model(config)
    pos = torch.randn(2, 5, 3)
    z = torch.tensor([[1, 6, 7, 8, 1], [6, 6, 1, 8, 7]], dtype=torch.long)
    mask = torch.ones(2, 5, dtype=torch.bool)
    rotation = random_rotation_matrix(2, dtype=pos.dtype)
    out = model(pos, z, mask)["forces"]
    out_rot = model(apply_rotation(pos, rotation), z, mask)["forces"]
    expected = apply_rotation(out, rotation)
    error = (out_rot - expected).norm() / expected.norm().clamp_min(1e-8)
    assert float(error) < 1e-5


def test_mlp_memorizer_diagnostic_output_shape():
    config = {"model": {"name": "mlp_memorizer", "hidden_dim": 32, "num_layers": 1, "max_atoms": 8}}
    model = build_molecular_model(config)
    pos = torch.randn(3, 5, 3)
    z = torch.randint(1, 9, (3, 5), dtype=torch.long)
    mask = torch.ones(3, 5, dtype=torch.bool)
    out = model(pos, z, mask)
    assert out["forces"].shape == pos.shape
    assert out["graph_stats"]["edge_count_mean"] == 0.0


def test_random_weight_molecular_backbones_do_not_predict_identically():
    pos = torch.randn(2, 5, 3)
    z = torch.tensor([[1, 6, 7, 8, 1], [6, 6, 1, 8, 7]], dtype=torch.long)
    mask = torch.ones(2, 5, dtype=torch.bool)
    predictions = {}
    for name, config_name in {
        "egnn": "rmd17_tiny_direct_egnn.yaml",
        "tfn": "rmd17_tiny_direct_tfn.yaml",
        "se3": "rmd17_tiny_direct_se3.yaml",
    }.items():
        torch.manual_seed(0)
        model = build_molecular_model(load_real_config(config_name))
        predictions[name] = model(pos, z, mask)["forces"].detach()
    assert (predictions["egnn"] - predictions["tfn"]).abs().max().item() > 1e-8
    assert (predictions["egnn"] - predictions["se3"]).abs().max().item() > 1e-8
    assert (predictions["tfn"] - predictions["se3"]).abs().max().item() > 1e-8


def test_atom_identity_changes_molecular_predictions():
    torch.manual_seed(5)
    model = build_molecular_model(load_real_config("rmd17_tiny_direct_se3.yaml"))
    pos = torch.randn(1, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)
    z_a = torch.tensor([[1, 6, 7, 8]], dtype=torch.long)
    z_b = torch.tensor([[8, 7, 6, 1]], dtype=torch.long)
    out_a = model(pos, z_a, mask)
    out_b = model(pos, z_b, mask)
    force_diff = (out_a["forces"] - out_b["forces"]).abs().max().item()
    energy_diff = (out_a["energy"] - out_b["energy"]).abs().max().item()
    assert max(force_diff, energy_diff) > 1e-8


def test_benchmark_complete_rejects_mismatched_skip_metrics(tmp_path):
    spec = importlib.util.spec_from_file_location("run_molecular_benchmark_suite", Path("scripts/run_molecular_benchmark_suite.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    metrics = {field: "" for field in module.REQUIRED_METRIC_FIELDS + module.MOLECULAR_REQUIRED_FIELDS}
    metrics.update({"config_name": "a.yaml", "run_name": "a_seed0", "seed": 0})
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    assert module.complete(path, config_name="a.yaml", run_name="a_seed0", seed=0)
    assert not module.complete(path, config_name="b.yaml", run_name="a_seed0", seed=0)
    assert not module.complete(path, config_name="a.yaml", run_name="wrong", seed=0)
    assert not module.complete(path, config_name="a.yaml", run_name="a_seed0", seed=1)


def test_real_report_labels_fake_smoke_data():
    spec = importlib.util.spec_from_file_location("make_molecular_real_report", Path("scripts/make_molecular_real_report.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    level, note = module.validation_level([{"data_source_type": "fake_rmd17_npz_smoke"}])
    assert level == "Fake rMD17-style local file"
    assert "This is a fake rMD17-style smoke file, not a real rMD17 benchmark." in note
    assert not module.ranking_claim_status([{"data_source_type": "fake_rmd17_npz_smoke", "is_fake_or_synthetic": "True"}], Path("summary.csv"))[0]


def test_force_scale_normalization_scales_loss_not_physical_metrics():
    out = {"forces": torch.full((1, 2, 3), 2.0)}
    batch = {
        "forces": torch.zeros(1, 2, 3),
        "mask": torch.ones(1, 2, dtype=torch.bool),
        "energy": torch.full((1, 1), float("nan")),
    }
    config = {"training": {"force_scale_normalization": "fixed", "force_scale_value": 2.0, "force_loss_type": "mse"}}
    loss, row = molecular_losses(out, batch, config, {"force_rms": 2.0, "force_std": 2.0})
    assert row["force_mse"] == 4.0
    assert row["force_mae"] == 2.0
    assert row["force_loss"] == 1.0
    assert float(loss) == 1.0


def test_vector_l2_loss_uses_per_atom_force_norm():
    out = {"forces": torch.zeros(1, 2, 3)}
    batch = {
        "forces": torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]]]),
        "mask": torch.ones(1, 2, dtype=torch.bool),
        "energy": torch.full((1, 1), float("nan")),
    }
    config = {"training": {"force_scale_normalization": "none", "force_loss_type": "vector_l2"}}
    loss, row = molecular_losses(out, batch, config, {"force_rms": 1.0, "force_std": 1.0})
    assert row["force_vector_l2_mae"] == 8.5
    assert float(loss) == 8.5


def test_fixed_force_scale_value_and_huber_delta_are_parsed():
    out = {"forces": torch.full((1, 1, 3), 2.0)}
    batch = {
        "forces": torch.zeros(1, 1, 3),
        "mask": torch.ones(1, 1, dtype=torch.bool),
        "energy": torch.full((1, 1), float("nan")),
    }
    config = {
        "training": {
            "force_scale_normalization": "fixed",
            "fixed_force_scale_value": 10.0,
            "force_loss_type": "huber",
            "huber_delta": 5.0,
        }
    }
    loss, row = molecular_losses(out, batch, config, {"force_rms": 1.0, "force_std": 1.0})
    assert row["force_loss"] > 0
    assert float(loss) == row["force_loss"]


def test_diagnostic_training_curve_records_gradient_fields(tmp_path):
    npz = tmp_path / "fake.npz"
    np.savez(
        npz,
        R=np.random.randn(4, 3, 3).astype("float32"),
        F=np.random.randn(4, 3, 3).astype("float32"),
        E=np.random.randn(4).astype("float32"),
        z=np.array([1, 6, 8], dtype="int64"),
    )
    config = {
        "seed": 0,
        "device": "cpu",
        "diagnostic_type": "overfit",
        "output_dir": str(tmp_path / "run"),
        "dataset": {
            "name": "rmd17",
            "molecule": "fake",
            "path": str(npz),
            "train_size": 2,
            "val_size": 1,
            "test_size": 1,
            "cutoff_radius": 4.0,
            "split_type": "chronological",
        },
        "model": {"name": "radial", "hidden_dim": 24, "radial_num_basis": 4, "radial_hidden_dim": 24},
        "training": {
            "mode": "direct_force",
            "diagnostic_logging": True,
            "batch_size": 2,
            "epochs": 1,
            "lr": 0.001,
            "force_loss_weight": 1.0,
            "energy_loss_weight": 0.0,
            "force_scale_normalization": "none",
            "force_loss_type": "vector_l2",
            "gradient_clip_norm": 1.0,
        },
    }
    metrics = train_molecular_from_config(config)
    rows = list(csv.DictReader(Path(metrics["training_curve"]).open()))
    assert "train_total_grad_norm_before_clip" in rows[0]
    assert "train_total_grad_norm_after_clip" in rows[0]
    assert "train_force_head_grad_norm" in rows[0]
    assert metrics["train_force_head_grad_norm_max"] is not None
    assert Path(metrics["train_force_norm_distribution"]).exists()


def test_force_diagnostic_metrics_have_expected_cosine_for_matching_forces():
    class CenteredForceModel(nn.Module):
        def forward(self, pos: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> dict:
            centered = pos - (pos * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
            return {
                "forces": centered * mask.unsqueeze(-1),
                "energy": pos.new_zeros(pos.shape[0], 1),
                "energy_raw": pos.new_zeros(pos.shape[0], 1),
                "graph_stats": {
                    "average_neighbors": 0.0,
                    "edge_count_mean": 0.0,
                    "edge_count_max": 0.0,
                    "graph_build_time_sec": 0.0,
                },
            }

    pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    centered = pos - pos.mean(dim=0, keepdim=True)
    metadata = MolecularMetadata(
        dataset_name="unit",
        molecule_name="toy",
        num_atoms=2,
        num_frames=1,
        unit_energy="arb",
        unit_force="arb",
        split_type="chronological",
        cutoff_radius=3.0,
    ).to_dict()
    dataset = MolecularListDataset(
        [{"pos": pos, "z": torch.tensor([1, 6]), "forces": centered, "energy": 0.0}],
        MolecularMetadata(
            dataset_name="unit",
            molecule_name="toy",
            num_atoms=2,
            num_frames=1,
            unit_energy="arb",
            unit_force="arb",
            split_type="chronological",
            cutoff_radius=3.0,
        ),
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=molecular_collate)
    metrics = evaluate_molecular_model(
        CenteredForceModel(),
        loader,
        {"training": {"mode": "direct_force", "gradient_clip_norm": 0.5}, "model": {"output_head_init_scale": 0.2}},
        metadata,
        device="cpu",
    )
    assert metrics["force_vector_l2_mae"] < 1e-7
    assert metrics["residual_force_norm_mean"] < 1e-7
    assert abs(metrics["pred_to_target_force_norm_ratio"] - 1.0) < 1e-6
    assert abs(metrics["force_cosine_similarity_mean"] - 1.0) < 1e-6
    assert metrics["gradient_clip_norm"] == 0.5
    assert metrics["output_head_init_scale"] == 0.2


def test_force_learning_sweep_summary_and_report_gate(tmp_path):
    spec = importlib.util.spec_from_file_location("run_rmd17_force_learning_sweep", Path("scripts/run_rmd17_force_learning_sweep.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = {
        "config_name": "rmd17_aspirin_1k_direct_se3_norm_epoch20.yaml",
        "model_name": "se3_transformer",
        "model_class": "MolecularSE3ForceTransformer",
        "backbone_class": "se3_scalar_attention_kernel",
        "dataset_name": "rmd17",
        "molecule_name": "aspirin",
        "data_source_type": "local_rmd17_npz",
        "is_fake_or_synthetic": False,
        "is_real_rmd17": True,
        "training_mode": "direct_force",
        "split_type": "chronological",
        "force_scale_normalization": "train_force_rms",
        "force_loss_type": "mse",
        "learning_rate": 0.0003,
        "seed": 0,
        "force_mae": 1.0,
        "force_vector_l2_mae": 1.5,
        "zero_force_mae": 2.0,
        "zero_force_vector_l2_mae": 3.0,
        "mean_force_mae": 1.8,
        "mean_force_vector_l2_mae": 2.8,
        "force_mae_improvement_vs_zero_pct": 50.0,
        "force_mae_improvement_vs_mean_pct": 44.4,
        "equivariance_error": 1e-7,
        "force_equivariance_error": 1e-7,
        "num_train_frames": 700,
        "num_val_frames": 150,
        "num_test_frames": 150,
        "batch_size": 8,
        "val_force_vector_l2_mae_final": 1.4,
    }
    summary = module.summarize([row, {**row, "seed": 1, "force_mae": 1.2, "force_vector_l2_mae": 1.7}])
    assert summary[0]["n"] == 2
    assert summary[0]["force_mae_mean"] == 1.1
    assert summary[0]["force_scale_normalization"] == "train_force_rms"
    report_spec = importlib.util.spec_from_file_location("make_force_learning_report", Path("scripts/make_force_learning_report.py"))
    report = importlib.util.module_from_spec(report_spec)
    assert report_spec.loader is not None
    report_spec.loader.exec_module(report)
    assert report.any_model_beats_baselines(summary)
    ok, blockers = report.claim_gate(summary, tmp_path)
    assert not ok
    assert "fewer than 3 seeds" in blockers


def test_overfit_runner_summary_includes_diagnostic_schema():
    spec = importlib.util.spec_from_file_location("run_rmd17_overfit_diagnostic", Path("scripts/run_rmd17_overfit_diagnostic.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = {
        "config_name": "rmd17_aspirin_overfit32_se3.yaml",
        "model_name": "se3_transformer",
        "model_class": "MolecularSE3ForceTransformer",
        "backbone_class": "se3_scalar_attention_kernel",
        "dataset_name": "rmd17",
        "molecule_name": "aspirin",
        "data_source_type": "local_rmd17_npz",
        "is_fake_or_synthetic": False,
        "is_real_rmd17": True,
        "diagnostic_type": "overfit",
        "training_mode": "direct_force",
        "split_type": "train_only_overfit",
        "force_scale_normalization": "train_force_rms",
        "force_loss_type": "mse",
        "learning_rate": 0.003,
        "gradient_clip_norm": 10.0,
        "output_head_init_scale": 0.5,
        "weight_decay": 0.0,
        "seed": 0,
        "target_force_norm_mean": 30.0,
        "pred_force_norm_mean": 29.0,
        "pred_to_target_force_norm_ratio": 0.97,
        "residual_force_norm_mean": 0.5,
        "force_cosine_similarity_mean": 0.99,
        "force_component_mean_pred": 0.0,
        "force_component_std_pred": 10.0,
        "force_component_mean_target": 0.0,
        "force_component_std_target": 10.0,
        "train_eval_force_vector_l2_mae": 0.5,
        "train_eval_force_vector_l2_mae_improvement_vs_zero_pct": 98.0,
    }
    summary = module.summarize([row])
    assert summary[0]["diagnostic_type"] == "overfit"
    assert summary[0]["gradient_clip_norm"] == 10.0
    assert summary[0]["output_head_init_scale"] == 0.5
    assert summary[0]["train_eval_force_vector_l2_mae_mean"] == 0.5
    assert summary[0]["force_cosine_similarity_mean_mean"] == 0.99


def test_force_diagnostic_report_interpretation_claims_are_conservative():
    spec = importlib.util.spec_from_file_location("make_force_diagnostic_report", Path("scripts/make_force_diagnostic_report.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    weak_overfit = [{"train_eval_force_vector_l2_mae_improvement_vs_zero_pct_mean": "10"}]
    strong_overfit = [{"train_eval_force_vector_l2_mae_improvement_vs_zero_pct_mean": "90"}]
    random_ok = [{"force_vector_l2_mae_improvement_vs_zero_pct_mean": "10", "force_vector_l2_mae_improvement_vs_mean_pct_mean": "5"}]
    chrono_bad = [{"force_vector_l2_mae_improvement_vs_zero_pct_mean": "0.2", "force_vector_l2_mae_improvement_vs_mean_pct_mean": "-1"}]
    random_bad = [{"force_vector_l2_mae_improvement_vs_zero_pct_mean": "0.2", "force_vector_l2_mae_improvement_vs_mean_pct_mean": "-1"}]
    assert module.interpretation(weak_overfit, [], [], []) == [module.OVERFIT_FAIL_STATEMENT]
    assert module.RANDOM_SHIFT_STATEMENT in module.interpretation(strong_overfit, random_ok, chrono_bad, [])
    assert module.BOTH_GENERALIZATION_FAIL_STATEMENT in module.interpretation(strong_overfit, random_bad, chrono_bad, [])


def test_energy_force_training_on_fake_npz_records_real_schema(tmp_path):
    npz = tmp_path / "fake.npz"
    np.savez(
        npz,
        R=np.random.randn(8, 4, 3).astype("float32"),
        F=np.random.randn(8, 4, 3).astype("float32"),
        E=np.random.randn(8).astype("float32"),
        z=np.array([1, 6, 7, 8], dtype="int64"),
    )
    config = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "run"),
        "dataset": {
            "name": "rmd17",
            "molecule": "fake",
            "path": str(npz),
            "train_size": 4,
            "val_size": 2,
            "test_size": 2,
            "cutoff_radius": 4.0,
            "split_type": "chronological",
        },
        "model": {"name": "se3_transformer", "channels_by_l": {0: 10}, "num_layers": 1, "radial_num_basis": 5, "radial_hidden_dim": 12},
        "training": {
            "mode": "energy_force",
            "batch_size": 2,
            "epochs": 1,
            "lr": 0.001,
            "lambda_force": 1.0,
            "lambda_energy": 0.1,
            "force_scale_normalization": "train_force_rms",
            "force_loss_type": "huber",
            "energy_centering": True,
            "energy_loss_on_centered": True,
            "max_steps_per_epoch": 1,
        },
    }
    metrics = train_molecular_from_config(config)
    assert metrics["dataset_name"] == "rmd17"
    assert metrics["split_type"] == "chronological"
    assert metrics["force_equivariance_error"] < 1e-5
    assert metrics["loss_mode"] == "combined"
    assert metrics["data_source_type"] == "fake_rmd17_npz_smoke"
    assert metrics["is_fake_or_synthetic"] is True
    assert metrics["is_real_rmd17"] is False
    assert metrics["dataset_path_basename"] == "fake.npz"
    assert metrics["num_frames_total"] == 8
    assert metrics["num_frames_used"] == 8
    assert metrics["num_train_frames"] == 4
    assert metrics["num_val_frames"] == 2
    assert metrics["num_test_frames"] == 2
    assert metrics["num_train_batches"] == 2
    assert metrics["batch_size"] == 2
    assert metrics["force_vector_l2_mae"] >= 0
    assert metrics["force_vector_l2_rmse"] >= 0
    assert metrics["zero_force_mae"] >= 0
    assert metrics["mean_force_mae"] >= 0
    assert metrics["energy_centering"] is True
    assert metrics["energy_loss_on_centered"] is True
    assert metrics["energy_mae_raw"] is not None
    assert metrics["energy_mae_centered"] is not None
    assert metrics["energy_train_mean"] != 0
    assert metrics["force_scale_normalization"] == "train_force_rms"
    assert metrics["force_scale_value"] > 0
    assert metrics["force_train_rms"] > 0
    assert metrics["force_train_std"] > 0
    assert metrics["force_loss_type"] == "huber"
    assert metrics["training_curve"].endswith("training_curve.csv")
    assert metrics["model_class"] == "MolecularSE3ForceTransformer"
    assert metrics["backbone_class"] == "se3_scalar_attention_kernel"
    run_dir = Path(config["output_dir"])
    assert Path(metrics["best_checkpoint"]).parent == run_dir
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "model_repr.txt").exists()
    assert (run_dir / "parameter_count_by_module.json").exists()
    assert (run_dir / "data_metadata.json").exists()
    loaders, metadata = build_molecular_dataloaders(config)
    eval_metrics = evaluate_molecular_checkpoint(config, metrics["best_checkpoint"], loaders, metadata, run_dir / "eval_metrics.json")
    assert eval_metrics["best_checkpoint"] == metrics["best_checkpoint"]
    assert Path(eval_metrics["best_checkpoint"]).parent == run_dir
    assert (run_dir / "eval_metrics.json").exists()


def test_se3_transformer_scalar_embed_receives_only_node_scalars():
    class Recorder(nn.Module):
        def __init__(self, out_dim: int) -> None:
            super().__init__()
            self.out_dim = out_dim
            self.seen_shape = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.seen_shape = tuple(x.shape)
            return x.new_zeros(*x.shape[:-1], self.out_dim)

    model = SE3ForceTransformer(scalar_input_dim=1, lmax=1, channels_by_l={0: 4, 1: 1}, num_layers=0)
    recorder = Recorder(model.scalar_dim)
    model.scalar_embed = recorder
    x = torch.randn(2, 3, 3)
    z = torch.randn(2, 3, 1)
    _ = model(x, z)
    assert recorder.seen_shape == (2, 3, 1)
