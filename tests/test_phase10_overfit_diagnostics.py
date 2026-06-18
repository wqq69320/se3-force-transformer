import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from se3force.geometry.rotations import apply_rotation, random_rotation_matrix
from se3force.models.molecular import build_molecular_model, build_molecular_graph, molecular_model_identity
from se3force.training.molecular_trainer import load_molecular_config, train_molecular_from_config


def test_full_irrep_se3_config_records_non_scalar_irreps():
    config = load_molecular_config("configs/molecular/real/diagnostics/rmd17_aspirin_overfit32_se3_full_irrep_l2.yaml")
    identity = molecular_model_identity(build_molecular_model(config))
    assert identity["model_class"] == "MolecularFullIrrepSE3ForceTransformer"
    assert identity["uses_non_scalar_hidden"] is True
    assert "1o" in identity["actual_hidden_irreps"]
    assert "2e" in identity["actual_hidden_irreps"]
    assert identity["force_head_irreps"] == "1x1o"


def test_scalar_se3_config_is_labeled_scalar_kernel():
    config = load_molecular_config("configs/molecular/real/diagnostics/rmd17_aspirin_overfit32_se3.yaml")
    identity = molecular_model_identity(build_molecular_model(config))
    assert identity["model_class"] == "MolecularSE3ForceTransformer"
    assert identity["uses_non_scalar_hidden"] is False
    assert identity["backbone_class"] == "se3_scalar_attention_kernel"
    assert "se3_scalar" in identity["architecture_signature"]


def test_full_graph_excludes_self_edges():
    pos = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    graph = build_molecular_graph(pos, mask, cutoff_radius=0.1, graph_mode="full")
    assert graph.edge_count == 2 * 4 * 3
    assert not torch.any(graph.src == graph.dst)


def test_radial_pair_model_is_equivariant():
    torch.manual_seed(21)
    config = load_molecular_config("configs/molecular/real/diagnostics/rmd17_aspirin_overfit32_radial_pair_wide.yaml")
    config["model"]["force_output_scale"] = 3.0
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


def test_atom_pair_embedding_changes_predictions_when_atom_types_change():
    torch.manual_seed(22)
    config = load_molecular_config("configs/molecular/real/diagnostics/rmd17_aspirin_overfit32_radial_pair_wide.yaml")
    model = build_molecular_model(config)
    with torch.no_grad():
        model.atom_embedding.weight.zero_()
    pos = torch.randn(1, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)
    z_a = torch.tensor([[1, 6, 7, 8]], dtype=torch.long)
    z_b = torch.tensor([[8, 7, 6, 1]], dtype=torch.long)
    diff = (model(pos, z_a, mask)["forces"] - model(pos, z_b, mask)["forces"]).abs().max()
    assert float(diff) > 1e-8


def test_full_irrep_scalar_output_scaling_preserves_equivariance():
    torch.manual_seed(23)
    config = load_molecular_config("configs/molecular/real/diagnostics/rmd17_aspirin_overfit32_se3_full_irrep_l1.yaml")
    config["model"]["force_output_scale"] = 10.0
    model = build_molecular_model(config)
    pos = torch.randn(1, 5, 3)
    z = torch.tensor([[1, 6, 7, 8, 1]], dtype=torch.long)
    mask = torch.ones(1, 5, dtype=torch.bool)
    rotation = random_rotation_matrix(1, dtype=pos.dtype)
    out = model(pos, z, mask)["forces"]
    out_rot = model(apply_rotation(pos, rotation), z, mask)["forces"]
    expected = apply_rotation(out, rotation)
    error = (out_rot - expected).norm() / expected.norm().clamp_min(1e-8)
    assert float(error) < 1e-5


def test_phase10_diagnostic_gradient_fields_are_present(tmp_path):
    npz = tmp_path / "fake.npz"
    np.savez(
        npz,
        R=np.random.randn(4, 4, 3).astype("float32"),
        F=np.random.randn(4, 4, 3).astype("float32"),
        E=np.random.randn(4).astype("float32"),
        z=np.array([1, 6, 7, 8], dtype="int64"),
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
            "cutoff_radius": 5.0,
            "split_type": "chronological",
        },
        "model": {
            "name": "radial_pair",
            "hidden_dim": 24,
            "radial_num_basis": 4,
            "radial_hidden_dim": 24,
            "edge_mlp_hidden_dim": 24,
            "edge_mlp_layers": 2,
            "use_atom_pair_embedding": True,
            "pair_embedding_dim": 6,
            "learnable_force_output_scale": True,
            "initial_force_output_scale": 2.0,
        },
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
    for field in [
        "train_last_hidden_norm",
        "train_message_norm_mean",
        "train_edge_message_norm_mean",
        "train_force_head_output_norm",
        "train_backbone_grad_norm",
        "train_edge_mlp_grad_norm",
        "train_learnable_force_output_scale_value",
        "train_learnable_force_output_scale_grad",
    ]:
        assert field in rows[0]
    assert metrics["train_edge_mlp_grad_norm_max"] is not None


def test_phase10_report_conclusion_logic():
    spec = importlib.util.spec_from_file_location("make_force_diagnostic_report", Path("scripts/make_force_diagnostic_report.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    def row(model_text: str, ok: bool) -> dict:
        return {
            "model_class": model_text,
            "architecture_signature": model_text,
            "train_eval_force_vector_l2_mae_improvement_vs_zero_pct_mean": "90" if ok else "10",
            "train_eval_pred_to_target_force_norm_ratio_mean": "0.95" if ok else "0.2",
            "train_eval_force_cosine_similarity_mean_mean": "0.9" if ok else "0.1",
        }

    mlp_ok = row("MolecularCoordinateMLPMemorizer", True)
    radial_ok = row("HighCapacityRadialForceModel radial_pair", True)
    se3_bad = row("MolecularFullIrrepSE3ForceTransformer se3_full", False)
    assert module.SE3_WRAPPER_BOTTLENECK_STATEMENT in module.interpretation([mlp_ok, radial_ok, se3_bad], [], [], [])

    se3_ok = row("MolecularFullIrrepSE3ForceTransformer se3_full", True)
    assert module.FULL_IRREP_SOLVED_STATEMENT in module.interpretation([se3_ok], [], [], [])


def test_overfit_runner_accepts_phase10_grid_args(tmp_path):
    out = tmp_path / "grid"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_rmd17_overfit_diagnostic.py",
            "--configs",
            "configs/molecular/real/diagnostics/rmd17_aspirin_overfit32_radial_pair_wide.yaml",
            "--models",
            "radial_pair",
            "--lrs",
            "1e-3",
            "--losses",
            "vector_l2",
            "--gradient-clips",
            "none",
            "--force-output-scales",
            "1",
            "--learnable-force-output-scales",
            "false",
            "--force-scale-normalizations",
            "none",
            "--graph-modes",
            "full",
            "--epochs",
            "1",
            "--max-runs",
            "1",
            "--seeds",
            "0",
            "--data-path",
            str(tmp_path / "missing.npz"),
            "--output",
            str(out),
            "--force",
        ],
        check=False,
    )
    assert (out / "per_run_metrics.csv").exists()
    assert (out / "summary_mean_std.csv").exists()
    assert (out / "best_runs.csv").exists()
