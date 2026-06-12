# SE3-ForceTransformer

SE3-ForceTransformer is a compact PyTorch research repository for 3D force field prediction with a full irrep-based SE(3)-Transformer-class architecture. The task is to map point positions `x: [B, N, 3]` and scalar node attributes `z: [B, N, C]` to force vectors `F: [B, N, 3]`.

Ordinary neural networks can depend on an arbitrary coordinate frame. The research question here is whether a direct SE(3)-equivariant transformer, built from irreducible representations, spherical harmonics, tensor products, invariant attention scores, and equivariant value aggregation, can predict forces while respecting the physics symmetry:

```text
f(RX + t, z) = R f(X, z)
```

## Architecture

The SE(3) model never concatenates raw coordinates into scalar node features. Scalar attributes are embedded into `0e` channels; all geometry enters through directed relative edge vectors `r_ij = x_j - x_i`.

Irreducible representations are handled by e3nn:

- `0e`: invariant scalars.
- `1o`: polar vectors, used for force output.
- `2e`: rank-2 even tensor features.

For each edge, real spherical harmonics `Y_l(r_ij)` provide angular basis features. Radial MLPs produce per-edge weights for `o3.FullyConnectedTensorProduct`, giving TFN-style equivariant kernels. The SE(3)-attention layer uses scalar invariant queries and keys:

```text
q_i = Linear(h_i) in C x 0e
k_ij = TP_key(h_j, Y(r_ij), phi_k(||r_ij||)) in C x 0e
v_ij = TP_value(h_j, Y(r_ij), phi_v(||r_ij||)) in hidden irreps
a_ij = softmax_j((q_i dot k_ij) / sqrt(C) + b(||r_ij||))
h'_i = sum_j a_ij v_ij
```

The attention score is invariant because it is built only from scalar irreps and radial functions. The value is equivariant, and the scalar attention weight preserves equivariance under weighted summation. Feed-forward blocks use an equivariant scalar-gated nonlinearity; ordinary ReLU/GELU is never applied to vector or tensor components.

The final head is `o3.Linear(hidden_irreps, "1x1o")`, so the output transforms as a force vector.

## Datasets

The central-force dataset uses

```text
F_i = sum_{j != i} m_i m_j (x_j - x_i) / (||x_j - x_i||^2 + eps)^(3/2)
```

The angular-potential dataset defines an invariant energy:

```text
E = sum_{i<j} m_i m_j exp(-||r_ij||^2 / sigma2^2)
  + lambda sum_i sum_{j<k, j,k != i} m_i m_j m_k
    exp(-(||r_ij||^2 + ||r_ik||^2) / sigma3^2) P2(cos theta)
```

where `P2(c) = 0.5 (3c^2 - 1)`. Forces are generated as `F = -grad_X E`, so the force field is equivariant because the energy is invariant to rotations and translations.

## Baselines And Ablations

Included baselines:

- `CoordMLP`: absolute-coordinate MLP, intentionally non-equivariant.
- `VanillaGraphTransformer`: standard transformer over `[x_i, z_i]`, intentionally non-equivariant.
- `EGNN`: scalar-message relative-vector equivariant baseline.
- `TFNConv` mode: `use_attention: false`.

Ablation configs cover `lmax = 0, 1, 2, 3`, no attention, no gate, central versus angular datasets, and rotated-translated evaluation.

## Install

```bash
pip install -e .
```

If your system only exposes Python as `python3` and pip as `pip3`, use either fallback:

```bash
python3 -m pip install -e .
pip3 install -e .
```

## Commands

Smoke test:

```bash
python scripts/run_smoke.py
python3 scripts/run_smoke.py
```

Equivariance evaluation:

```bash
python scripts/eval_equivariance.py --model se3_transformer --config configs/small_cpu.yaml
python3 scripts/eval_equivariance.py --model se3_transformer --config configs/small_cpu.yaml
```

Train:

```bash
python scripts/train.py --config configs/angular_force_se3_l2.yaml
python3 scripts/train.py --config configs/angular_force_se3_l2.yaml
```

Evaluate a checkpoint and write metrics JSON:

```bash
python scripts/evaluate.py --config configs/angular_force_se3_l2.yaml --checkpoint outputs/angular_force_se3_l2/best.pt
python3 scripts/evaluate.py --config configs/angular_force_se3_l2.yaml --checkpoint outputs/angular_force_se3_l2/best.pt
```

Ablation grid:

```bash
python scripts/run_ablation_grid.py --configs configs/ablation_lmax0.yaml configs/ablation_lmax1.yaml configs/ablation_lmax2.yaml configs/ablation_lmax3.yaml
```

Tests:

```bash
pytest
python3 -m pytest
```

Benchmark suite:

```bash
python3 scripts/run_benchmark_suite.py --configs configs/baseline_egnn.yaml configs/angular_force_se3_l2.yaml --seeds 0 1 2 --output outputs/benchmark_suite
python3 scripts/plot_benchmark_results.py --input outputs/benchmark_suite/summary_mean_std.csv --output outputs/benchmark_suite/plots
python3 scripts/make_research_report.py --input outputs/benchmark_suite/summary_mean_std.csv --output outputs/benchmark_suite/research_report.md
```

Full Phase 3 smoke matrix:

```bash
python3 scripts/run_benchmark_suite.py \
  --configs \
    configs/baseline_mlp.yaml \
    configs/baseline_vanilla_gt.yaml \
    configs/baseline_egnn.yaml \
    configs/baseline_egnn_matched_params.yaml \
    configs/baseline_tfn.yaml \
    configs/ablation_lmax0.yaml \
    configs/ablation_lmax1.yaml \
    configs/ablation_lmax2.yaml \
    configs/ablation_lmax3.yaml \
    configs/ablation_no_attention.yaml \
    configs/ablation_no_gate.yaml \
    configs/angular_force_se3_l2.yaml \
    configs/angular_force_se3_l2_small.yaml \
    configs/angular_force_se3_l3.yaml \
  --seeds 0 1 2 \
  --output outputs/benchmark_full_seed3
```

The matched-parameter configs are approximate fairness checks: `baseline_egnn_matched_params.yaml` increases EGNN capacity toward the default SE3 l=2 parameter count, while `angular_force_se3_l2_small.yaml` reduces SE3 capacity toward the default EGNN size.

Expected SE3ForceTransformer equivariance error is typically around `1e-5` to `1e-4` in float32 CPU runs. The EGNN baseline should also have low final force equivariance error, while the coordinate MLP and vanilla graph transformer should be much larger.

## Metrics

All training and evaluation JSON outputs use a flat schema with:

```text
run_name, model_name, config_name, dataset_name, seed, device,
num_train_samples, num_val_samples, num_test_samples, best_checkpoint,
final_train_loss, best_val_mse, canonical_mse, rotated_translated_mse,
equivariance_error, parameter_count, runtime_per_batch_sec
```

`canonical_mse` measures test error in the original coordinate frame. `rotated_translated_mse` evaluates the same checkpoint after random rigid transforms of inputs and targets. `equivariance_error` measures whether `model(Rx+t, z)` matches `R model(x, z)`, independent of prediction accuracy. `parameter_count` and `runtime_per_batch_sec` help compare model size and CPU cost.

## Verification Status

Current smoke verification covers editable install, pytest, smoke training, standalone SE3 equivariance evaluation, angular-force SE3 lmax=2 training, and checkpoint evaluation. On the local Python 3 environment used during MVP validation, the SE3 equivariance check was around `1e-7`.

This is implementation and symmetry evidence, not yet a broad research claim. Baseline and ablation evidence across comparable seeds, data sizes, and train budgets is required before claiming superior predictive performance or architectural advantage.

## Molecular Scale-Up

Phase 4 adds a local-data molecular path for rMD17/MD22-style force prediction, variable atom counts, cutoff graphs, atom embeddings, and direct-force or conservative energy-force training modes. This does not by itself establish a broad molecular-force-field claim; real molecular datasets, multiple molecules, and multiple seeds are required.

Inspect a local rMD17 file:

```bash
python3 scripts/inspect_molecular_dataset.py --dataset rmd17 --path data/rmd17/aspirin.npz --molecule aspirin
```

Train and evaluate a molecular config:

```bash
python3 scripts/train_molecular.py --config configs/molecular/scale_synthetic_n8.yaml
python3 scripts/evaluate_molecular.py --config configs/molecular/scale_synthetic_n8.yaml --checkpoint outputs/scale_synthetic_n8/best.pt
```

Run a synthetic scale sweep:

```bash
python3 scripts/run_scale_sweep.py --configs configs/molecular/scale_synthetic_n8.yaml configs/molecular/scale_synthetic_n12.yaml --seeds 0 --output outputs/scale_synthetic_smoke
```

Run a demo:

```bash
python3 scripts/demo_structure_relaxation.py --config configs/molecular/scale_synthetic_n8.yaml --checkpoint outputs/scale_synthetic_n8/best.pt --output outputs/demos/relaxation
```

See [SCALEUP.md](SCALEUP.md), [MOLECULAR_DATA.md](MOLECULAR_DATA.md), and [SCIENTIFIC_DEMO.md](SCIENTIFIC_DEMO.md).

This repository uses e3nn for group-theoretic primitives, while implementing the model architecture, synthetic tasks, tests, training pipeline, evaluation scripts, and ablation entry points directly.
