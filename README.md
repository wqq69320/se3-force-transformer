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

If your system only exposes Python as `python3`, use:

```bash
python3 -m pip install -e .
```

## Commands

Smoke test:

```bash
python scripts/run_smoke.py
```

Equivariance evaluation:

```bash
python scripts/eval_equivariance.py --model se3_transformer --config configs/small_cpu.yaml
```

Train:

```bash
python scripts/train.py --config configs/angular_force_se3_l2.yaml
```

Evaluate a checkpoint and write metrics JSON:

```bash
python scripts/evaluate.py --config configs/angular_force_se3_l2.yaml --checkpoint outputs/angular_force_se3_l2/best.pt
```

Ablation grid:

```bash
python scripts/run_ablation_grid.py --configs configs/ablation_lmax0.yaml configs/ablation_lmax1.yaml configs/ablation_lmax2.yaml configs/ablation_lmax3.yaml
```

Tests:

```bash
pytest
```

Expected SE3ForceTransformer equivariance error is typically around `1e-5` to `1e-4` in float32 CPU runs. The EGNN baseline should also have low final force equivariance error, while the coordinate MLP and vanilla graph transformer should be much larger.

This repository uses e3nn for group-theoretic primitives, while implementing the model architecture, synthetic tasks, tests, training pipeline, evaluation scripts, and ablation entry points directly.
