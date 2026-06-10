from se3force.training.trainer import train_from_config


def test_tiny_training_loop_runs():
    config = {
        "seed": 3,
        "device": "cpu",
        "output_dir": "outputs/test_training_smoke",
        "dataset": {
            "name": "central",
            "num_samples": 8,
            "num_nodes": 4,
            "position_scale": 1.0,
            "mass_range": [0.8, 1.2],
            "seed": 3,
        },
        "model": {
            "name": "se3_transformer",
            "scalar_input_dim": 1,
            "lmax": 1,
            "channels_by_l": {0: 6, 1: 3},
            "num_layers": 1,
            "num_heads": 1,
            "num_query_channels": 3,
            "radial_num_basis": 5,
            "radial_hidden_dim": 12,
            "dropout": 0.0,
            "use_attention": True,
            "use_gate": True,
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "val_fraction": 0.25,
            "test_fraction": 0.25,
            "max_steps_per_epoch": 1,
        },
    }
    metrics = train_from_config(config)
    assert metrics["history"]
