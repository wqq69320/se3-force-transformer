import csv
import importlib.util
import sys
from pathlib import Path

from se3force.models.molecular import build_molecular_model, molecular_model_identity
from se3force.training.molecular_trainer import load_molecular_config


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


def row(model: str, mae: float, split: str = "random", n: int = 3, budgets: bool = True) -> dict:
    config = f"rmd17_aspirin_1k_{'chrono' if split == 'chronological' else 'random'}_{model}.yaml"
    data = {
        "config_name": config,
        "model_name": "se3_transformer" if model == "se3" else model,
        "backbone_class": model,
        "split_type": split,
        "n": str(n),
        "is_real_rmd17": "True",
        "is_fake_or_synthetic": "False",
        "force_vector_l2_mae_mean": str(mae),
        "zero_force_vector_l2_mae_mean": "40",
        "mean_force_vector_l2_mae_mean": "40",
        "force_vector_l2_mae_improvement_vs_zero_pct_mean": str((40 - mae) / 40 * 100),
        "force_vector_l2_mae_improvement_vs_mean_pct_mean": str((40 - mae) / 40 * 100),
        "force_component_std_pred_mean": "1.0",
        "force_equivariance_error_mean": "1e-6",
        "parameter_count_mean": "1000",
        "runtime_per_batch_sec_mean": "0.1",
        "graph_mode": "cutoff",
        "cutoff_radius_mean": "5.0",
        "average_neighbors_mean": "8.0",
    }
    if model == "global_coeff":
        data["model_name"] = "global_coeff"
        data["backbone_class"] = "global_invariant_coefficients"
        data["graph_mode"] = "full"
    if model == "global_context_radial":
        data["model_name"] = "global_context_radial"
        data["backbone_class"] = "global_context_radial_pair"
        data["uses_global_context"] = "True"
        data["global_context_type"] = "full_distance_atom_embedding"
        data["graph_mode"] = "full"
    if budgets:
        data.update(
            {
                "learning_rate": "0.001",
                "force_loss_type": "mse",
                "force_scale_normalization": "train_force_rms",
                "num_frames_used_mean": "1000",
                "num_train_frames_mean": "700",
                "num_val_frames_mean": "150",
                "num_test_frames_mean": "150",
                "batch_size_mean": "32",
                "epochs_mean": "50",
                "max_steps_per_epoch_mean": "8",
            }
        )
    return data


def test_phase13_required_configs_and_ablation_fields_exist():
    root = Path("configs/molecular/real/phase13")
    required = [
        "rmd17_aspirin_1k_random_egnn_matched.yaml",
        "rmd17_aspirin_1k_random_tfn_matched.yaml",
        "rmd17_aspirin_1k_random_se3_local_matched.yaml",
        "rmd17_aspirin_1k_random_radial_pair_matched.yaml",
        "rmd17_aspirin_1k_random_painn_lite_matched.yaml",
        "rmd17_aspirin_1k_random_global_coeff.yaml",
        "rmd17_aspirin_1k_random_global_context_radial.yaml",
        "rmd17_aspirin_1k_chrono_egnn_matched.yaml",
        "rmd17_aspirin_1k_chrono_tfn_matched.yaml",
        "rmd17_aspirin_1k_chrono_se3_local_matched.yaml",
        "rmd17_aspirin_1k_chrono_radial_pair_matched.yaml",
        "rmd17_aspirin_1k_chrono_painn_lite_matched.yaml",
        "rmd17_aspirin_1k_chrono_global_coeff.yaml",
        "rmd17_aspirin_1k_chrono_global_context_radial.yaml",
        "rmd17_aspirin_1k_random_global_coeff_no_atom_pair.yaml",
        "rmd17_aspirin_1k_random_global_coeff_no_prototype_memory.yaml",
        "rmd17_aspirin_1k_random_global_coeff_cutoff.yaml",
        "rmd17_aspirin_1k_random_global_coeff_fullgraph.yaml",
        "rmd17_aspirin_1k_random_global_context_radial_no_global.yaml",
        "rmd17_aspirin_1k_random_global_context_radial_with_global.yaml",
    ]
    for filename in required:
        assert (root / filename).exists()
    no_pair = load_molecular_config(root / "rmd17_aspirin_1k_random_global_coeff_no_atom_pair.yaml")
    assert no_pair["model"]["use_atom_pair_embedding"] is False
    no_proto = load_molecular_config(root / "rmd17_aspirin_1k_random_global_coeff_no_prototype_memory.yaml")
    assert no_proto["model"]["use_prototype_memory"] is False
    no_global = load_molecular_config(root / "rmd17_aspirin_1k_random_global_context_radial_no_global.yaml")
    assert no_global["model"]["use_global_context"] is False


def test_phase13_model_ablation_metadata():
    no_pair = build_molecular_model({"model": {"name": "global_coeff", "use_atom_pair_embedding": False}, "training": {"mode": "direct_force"}})
    ident = molecular_model_identity(no_pair)
    assert ident["uses_atom_pair_embedding"] is False
    assert ident["pair_embedding_dim"] == 0
    no_global = build_molecular_model(
        {
            "model": {"name": "global_context_radial", "max_atoms": 8, "use_global_context": False, "hidden_dim": 16, "global_context_dim": 8},
            "training": {"mode": "direct_force"},
        }
    )
    ident = molecular_model_identity(no_global)
    assert ident["uses_global_context"] is False
    assert ident["global_context_type"] == "none"


def test_phase13_runner_groups_seeds_and_keeps_ablations_separate():
    runner = load_script("run_rmd17_overfit_diagnostic_phase13", "scripts/run_rmd17_overfit_diagnostic.py")
    base = {field: "" for field in runner.GROUP_FIELDS + runner.NUMERIC_FIELDS}
    base.update(
        {
            "config_name": "cfg.yaml",
            "model_name": "global_coeff",
            "model_class": "GlobalInvariantCoefficientForceModel",
            "backbone_class": "global_invariant_coefficients",
            "split_type": "random",
            "learning_rate": 0.001,
            "force_loss_type": "mse",
            "uses_atom_pair_embedding": True,
            "use_prototype_memory": True,
            "prototype_assignment": "nearest",
            "global_context_type": "",
        }
    )
    rows = []
    for seed, scale in [(0, 29.4), (1, 29.5), (2, 29.6)]:
        item = dict(base)
        item.update({"seed": seed, "fixed_force_scale_value": scale, "prototype_count": 128})
        rows.append(item)
    no_proto = dict(base)
    no_proto.update({"seed": 0, "use_prototype_memory": False, "prototype_assignment": "", "prototype_count": 0})
    summary = runner.summarize(rows + [no_proto])
    counts = sorted(int(row["n"]) for row in summary)
    assert counts == [1, 3]


def test_phase13_report_claim_gate_and_refusals():
    report = load_script("make_global_vs_local_report", "scripts/make_global_vs_local_report.py")
    passing = [
        row("egnn", 12),
        row("tfn", 11),
        row("se3", 10),
        row("radial_pair", 13),
        row("painn_lite", 12),
        row("global_coeff", 8),
    ]
    ok, reasons = report.claim_gate(passing, [], "random")
    assert ok, reasons
    chrono_global_context = row("global_context_radial", 8, split="chronological")
    assert report.model_family(chrono_global_context) == "global_context_radial"
    no_global_context = row("global_context_radial", 8)
    no_global_context["config_name"] = "rmd17_aspirin_1k_random_global_context_radial_no_global.yaml"
    no_global_context["uses_global_context"] = "False"
    assert report.model_family(no_global_context) == "global_context_radial_no_global"
    missing = [row("egnn", 12), row("global_coeff", 8)]
    ok, reasons = report.claim_gate(missing, [], "random")
    assert not ok
    assert any("TFN" in reason or "local SE3" in reason for reason in reasons)
    bad_budget = [row("egnn", 12), row("tfn", 11), row("se3", 10), row("global_coeff", 8, budgets=False)]
    ok, reasons = report.claim_gate(bad_budget, [], "random")
    assert not ok
    assert any("budget" in reason for reason in reasons)


def test_phase13_plot_script_with_minimal_csv(tmp_path):
    plotter = load_script("plot_global_vs_local_results", "scripts/plot_global_vs_local_results.py")
    rows = [row("egnn", 12), row("tfn", 11), row("se3", 10), row("global_coeff", 8)]
    csv_path = tmp_path / "summary.csv"
    fields = sorted({key for item in rows for key in item})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "plots"
    old = sys.argv
    sys.argv = ["plot_global_vs_local_results.py", "--random-input", str(csv_path), "--output", str(output)]
    try:
        plotter.main()
    finally:
        sys.argv = old
    assert (output / "random_vector_l2_by_model.png").exists()
    assert (output / "runtime_vs_vector_l2.png").exists()
