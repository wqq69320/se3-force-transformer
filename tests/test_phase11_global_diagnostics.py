import argparse
import importlib.util

import torch

from se3force.evaluation.molecular_evaluate import force_scale_settings
from se3force.geometry.rotations import apply_rotation, random_rotation_matrix
from se3force.models.molecular import build_molecular_model
from se3force.training.molecular_trainer import molecular_losses


def config_for(name: str) -> dict:
    base = {
        "model": {
            "name": name,
            "max_atoms": 8,
            "hidden_dim": 32,
            "num_layers": 2,
            "global_context_dim": 24,
            "atom_embedding_dim": 8,
            "pair_embedding_dim": 8,
            "radial_num_basis": 6,
            "edge_mlp_hidden_dim": 32,
            "edge_mlp_layers": 1,
            "vector_channels": 4,
            "radial_hidden_dim": 32,
            "graph_mode": "full",
        },
        "training": {"mode": "direct_force", "force_scale_normalization": "none", "force_loss_type": "mse"},
    }
    if name == "internal_energy":
        base["model"]["use_angles"] = True
    return base


def sample():
    torch.manual_seed(31)
    pos = torch.randn(2, 5, 3)
    z = torch.tensor([[1, 6, 7, 8, 1], [6, 6, 1, 8, 7]], dtype=torch.long)
    mask = torch.ones(2, 5, dtype=torch.bool)
    return pos, z, mask


def assert_force_equivariant(model):
    pos, z, mask = sample()
    rotation = random_rotation_matrix(2, dtype=pos.dtype)
    translation = torch.randn(2, 1, 3)
    out = model(pos, z, mask)["forces"]
    out_rot = model(apply_rotation(pos, rotation) + translation, z, mask)["forces"]
    expected = apply_rotation(out, rotation)
    error = (out_rot - expected).norm() / expected.norm().clamp_min(1e-8)
    assert float(error) < 1e-5


def test_global_invariant_coefficient_model_equivariance_and_atom_sensitivity():
    model = build_molecular_model(config_for("global_coeff"))
    assert_force_equivariant(model)
    pos, _, mask = sample()
    z_a = torch.tensor([[1, 6, 7, 8, 1], [1, 6, 7, 8, 1]], dtype=torch.long)
    z_b = torch.tensor([[8, 7, 6, 1, 1], [8, 7, 6, 1, 1]], dtype=torch.long)
    diff = (model(pos, z_a, mask)["forces"] - model(pos, z_b, mask)["forces"]).abs().max()
    assert float(diff) > 1e-8


def test_global_invariant_prototype_memory_equivariance_and_training_step():
    cfg = config_for("global_coeff")
    cfg["model"].update({"use_prototype_memory": True, "prototype_count": 2, "prototype_assignment": "nearest"})
    model = build_molecular_model(cfg)
    assert_force_equivariant(model)
    pos, z, mask = sample()
    batch = {
        "pos": pos,
        "z": z,
        "mask": mask,
        "forces": torch.randn_like(pos),
        "energy": torch.full((pos.shape[0], 1), float("nan")),
    }
    out = model(pos, z, mask)
    loss, row = molecular_losses(out, batch, cfg, {"force_rms": 1.0, "force_std": 1.0}, model=model)
    loss.backward()
    assert row["force_vector_l2_mae"] > 0
    assert model.prototype_coefficients is not None
    assert model.prototype_coefficients.grad is not None


def test_internal_coordinate_energy_invariant_and_force_equivariant():
    model = build_molecular_model(config_for("internal_energy"))
    pos, z, mask = sample()
    rotation = random_rotation_matrix(2, dtype=pos.dtype)
    translation = torch.randn(2, 1, 3)
    out = model(pos, z, mask)
    out_rot = model(apply_rotation(pos, rotation) + translation, z, mask)
    energy_error = (out["energy"] - out_rot["energy"]).abs().max()
    expected_force = apply_rotation(out["forces"], rotation)
    force_error = (out_rot["forces"] - expected_force).norm() / expected_force.norm().clamp_min(1e-8)
    assert float(energy_error) < 1e-5
    assert float(force_error) < 1e-5


def test_painn_lite_equivariance_and_training_step():
    model = build_molecular_model(config_for("painn_lite"))
    assert_force_equivariant(model)
    pos, z, mask = sample()
    batch = {
        "pos": pos,
        "z": z,
        "mask": mask,
        "forces": torch.randn_like(pos),
        "energy": torch.full((pos.shape[0], 1), float("nan")),
    }
    out = model(pos, z, mask)
    loss, row = molecular_losses(out, batch, config_for("painn_lite"), {"force_rms": 1.0, "force_std": 1.0}, model=model)
    loss.backward()
    assert row["force_vector_l2_mae"] > 0
    assert any(param.grad is not None for param in model.parameters())


def test_internal_energy_training_step():
    model = build_molecular_model(config_for("internal_energy"))
    pos, z, mask = sample()
    batch = {
        "pos": pos,
        "z": z,
        "mask": mask,
        "forces": torch.randn_like(pos),
        "energy": torch.full((pos.shape[0], 1), float("nan")),
    }
    out = model(pos, z, mask)
    loss, row = molecular_losses(out, batch, config_for("internal_energy"), {"force_rms": 1.0, "force_std": 1.0}, model=model)
    loss.backward()
    assert row["force_vector_l2_mae"] > 0
    assert any(param.grad is not None for param in model.parameters())


def test_runner_accepts_phase11_model_names_and_options():
    spec = importlib.util.spec_from_file_location("run_rmd17_overfit_diagnostic", "scripts/run_rmd17_overfit_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    for name in ["global_coeff", "internal_energy", "painn_lite"]:
        config = {"model": {}}
        module.apply_model_choice(config, name)
        assert config["model"]["name"] in {"global_coeff", "internal_energy", "painn_lite"}
        assert module.diagnostic_type_for(config).startswith("equivariant_")
    assert module.parse_bool("true") is True
    assert module.parse_bool("false") is False


def test_phase11_report_conclusion_logic():
    spec = importlib.util.spec_from_file_location("make_force_diagnostic_report", "scripts/make_force_diagnostic_report.py")
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

    assert module.GLOBAL_COEFF_SOLVED_STATEMENT in module.interpretation([row("GlobalInvariantCoefficientForceModel global_coeff", True)], [], [], [])
    assert module.INTERNAL_ENERGY_SOLVED_STATEMENT in module.interpretation([row("InternalCoordinateEnergyMemorizer internal_energy", True)], [], [], [])
    assert module.PAINN_LITE_SOLVED_STATEMENT in module.interpretation([row("PaiNNLiteForceModel painn_lite", True)], [], [], [])
    assert module.PHASE11_EQUIVARIANT_FAIL_STATEMENT in module.interpretation(
        [row("MolecularCoordinateMLPMemorizer", True), row("GlobalInvariantCoefficientForceModel global_coeff", False)],
        [],
        [],
        [],
    )
