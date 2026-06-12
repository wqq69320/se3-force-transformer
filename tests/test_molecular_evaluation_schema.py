from se3force.evaluation.metrics_schema import REQUIRED_METRIC_FIELDS
from se3force.evaluation.molecular_metrics import MOLECULAR_REQUIRED_FIELDS
from se3force.training.molecular_trainer import train_molecular_from_config


def test_molecular_metrics_schema_is_complete(tmp_path):
    config = {
        "seed": 1,
        "device": "cpu",
        "output_dir": str(tmp_path / "mol_schema"),
        "dataset": {
            "name": "scale_synthetic",
            "molecule": "schema",
            "num_frames": 8,
            "num_atoms": 4,
            "train_size": 4,
            "val_size": 2,
            "test_size": 2,
            "cutoff_radius": 4.0,
            "seed": 1,
        },
        "model": {"name": "se3_transformer", "channels_by_l": {0: 10}, "num_layers": 1, "radial_num_basis": 5, "radial_hidden_dim": 12},
        "training": {"mode": "energy_force", "batch_size": 2, "epochs": 1, "lr": 0.001, "lambda_energy": 0.1, "max_steps_per_epoch": 1},
    }
    metrics = train_molecular_from_config(config)
    missing = [field for field in REQUIRED_METRIC_FIELDS + MOLECULAR_REQUIRED_FIELDS if field not in metrics]
    assert not missing
    assert metrics["energy_invariance_error"] is not None
