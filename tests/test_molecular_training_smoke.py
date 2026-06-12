from se3force.training.molecular_trainer import train_molecular_from_config


def test_molecular_training_smoke_runs(tmp_path):
    config = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "mol_smoke"),
        "dataset": {
            "name": "scale_synthetic",
            "molecule": "tiny",
            "num_frames": 8,
            "num_atoms": 5,
            "train_size": 4,
            "val_size": 2,
            "test_size": 2,
            "cutoff_radius": 4.0,
            "seed": 0,
        },
        "model": {"name": "se3_transformer", "channels_by_l": {0: 12}, "num_layers": 1, "radial_num_basis": 5, "radial_hidden_dim": 12},
        "training": {"mode": "direct_force", "batch_size": 2, "epochs": 1, "lr": 0.001, "max_steps_per_epoch": 1},
    }
    metrics = train_molecular_from_config(config)
    assert metrics["force_mae"] >= 0
    assert metrics["average_neighbors"] >= 0
