Build a complete, runnable, paper-grade PyTorch research repository named `se3-force-transformer`.

Project title:
"SE3-ForceTransformer: Full Irrep-Based SE(3)-Transformer for 3D Force Field Prediction"

This is NOT a minimal relative-vector EGNN-style model. Implement a full irrep-based SE(3)-Transformer-class architecture for prediction, using irreducible representations, spherical harmonics, radial networks, tensor product kernels, invariant attention scores, equivariant value aggregation, gated equivariant nonlinearities, and a force-vector prediction head.

Primary task:
3D force field prediction. Given 3D point positions `x: [B, N, 3]` and scalar node attributes `z: [B, N, C]`, predict force vectors `F: [B, N, 3]`.

Target equivariance:
For any rotation matrix `R in SO(3)` and translation `t in R^3`,

`model(Rx + t, z) ≈ R model(x, z)`

The project is complete only when:
1. `pip install -e .` succeeds.
2. `pytest` passes.
3. `python scripts/run_smoke.py` runs on CPU.
4. `python scripts/eval_equivariance.py --model se3_transformer --config configs/small_cpu.yaml` runs and reports low equivariance error.
5. `python scripts/train.py --config configs/angular_force_se3_l2.yaml` starts training successfully.
6. `python scripts/evaluate.py --config <config> --checkpoint <path>` evaluates a checkpoint and writes metrics JSON.
7. The README explains the mathematics, model architecture, datasets, equivariance proof sketch, baselines, ablation plan, and command-line usage.
8. `AGENTS.md` exists and documents repo conventions, tests, commands, scientific invariants, and future-agent rules.
9. There are no placeholder-only files, fake tests, or unimplemented TODO stubs.

Use Python 3.10+.

Dependencies:
- Required: torch, numpy, pyyaml, pytest, matplotlib, tqdm, e3nn.
- Avoid heavy graph frameworks in the MVP: do not use PyTorch Geometric, DGL, Lightning, Hydra, wandb, or torch-scatter unless absolutely necessary.
- Implement dense all-pairs graphs for small-N synthetic datasets first. This is acceptable for the paper-grade MVP because correctness and equivariance are more important than large-scale speed at this stage.

Repository structure:

```text
se3-force-transformer/
  README.md
  AGENTS.md
  pyproject.toml

  configs/
    central_force_se3_l2.yaml
    angular_force_se3_l2.yaml
    angular_force_se3_l3.yaml
    baseline_mlp.yaml
    baseline_vanilla_gt.yaml
    baseline_egnn.yaml
    baseline_tfn.yaml
    ablation_lmax0.yaml
    ablation_lmax1.yaml
    ablation_lmax2.yaml
    ablation_lmax3.yaml
    ablation_no_attention.yaml
    ablation_no_gate.yaml
    small_cpu.yaml

  src/
    se3force/
      __init__.py

      data/
        __init__.py
        central_force.py
        angular_potential.py
        datamodule.py

      geometry/
        __init__.py
        rotations.py
        transforms.py
        pairwise.py
        irreps.py
        metrics.py

      models/
        __init__.py
        common.py
        radial.py
        edge_graph.py

        baselines/
          __init__.py
          coord_mlp.py
          vanilla_graph_transformer.py
          egnn.py

        equivariant/
          __init__.py
          irrep_norm.py
          gate.py
          tfn_conv.py
          se3_attention.py
          se3_transformer_block.py
          se3_force_transformer.py

      training/
        __init__.py
        losses.py
        trainer.py
        checkpointing.py
        logging.py
        seed.py

      evaluation/
        __init__.py
        evaluate.py
        equivariance.py
        ablations.py
        plots.py

  scripts/
    train.py
    evaluate.py
    eval_equivariance.py
    inspect_irreps.py
    run_smoke.py
    run_ablation_grid.py

  tests/
    test_rotations.py
    test_pairwise.py
    test_spherical_harmonics.py
    test_central_force_equivariance.py
    test_angular_force_equivariance.py
    test_model_shapes.py
    test_tfn_layer_equivariance.py
    test_se3_attention_equivariance.py
    test_se3_force_transformer_equivariance.py
    test_training_smoke.py

  outputs/
    .gitkeep
Implementation details:

1. Package setup

* Use pyproject.toml.
* Package name: se3force.
* Scripts must be runnable from repo root.
* Use simple YAML config loading, not Hydra.
* Use deterministic seeds.

2. Geometry utilities
    Implement in src/se3force/geometry/:

* random_rotation_matrix(batch, device=None, dtype=None) -> Tensor returning [B, 3, 3].
* random_translation(batch, scale=1.0, device=None, dtype=None) -> Tensor returning [B, 1, 3].
* apply_rotation(x, R) for [B, N, 3].
* apply_transform(x, R, t).
* pairwise_relative(x) returning r_ij = x_j - x_i with shape [B, N, N, 3].
* pairwise_dist(x) and pairwise_dist2(x).
* unit_vectors(r, eps).
* relative_error(a, b, eps=1e-8).
* parameter_count(model).

Tests:

* Rotation matrices must be orthogonal.
* Determinants must be close to +1.
* Pairwise relative vectors must be translation invariant.
* Pairwise distances must be rotation/translation invariant.

3. Irrep utilities
    Use e3nn.
    Implement in geometry/irreps.py:

* helper to build hidden irreps from config:
    * num_scalar_channels
    * num_vector_channels
    * lmax
    * channels_by_l
* helper to build spherical harmonics irreps:
    * o3.Irreps.spherical_harmonics(lmax)
* helper to get Wigner-D matrices:
    * irreps.D_from_matrix(R)
* helper to transform irrep features:
    * transform_features(features, irreps, R)

Tests:

* For random vectors r, verify spherical harmonics transform correctly:
    Y(Rr) ≈ D(R) Y(r) within reasonable float tolerance.
* For hidden features and irreps, verify transform_features has correct shape.

4. Datasets

Implement two synthetic datasets.

4.1 Central-force dataset
File:
src/se3force/data/central_force.py

Input:

* x: [N, 3]
* z: [N, C], at least mass as one scalar channel
* F: [N, 3]

Force:
F_i = sum_{j != i} m_i * m_j * (x_j - x_i) / (||x_j - x_i||^2 + eps)^(3/2)

Support:

* num_samples
* num_nodes
* position_scale
* anisotropic=True
* mass_range
* seed
* random_rotate_translate=False/True

4.2 Angular-potential force dataset
File:
src/se3force/data/angular_potential.py

Define an invariant energy:
E(X) = E2(X) + lambda_angle * E3(X)

Pairwise:
E2 = sum_{i<j} m_i m_j exp(-||r_ij||^2 / sigma2^2)

Angular:
For every center node i and unordered pair (j, k) with j < k, j != i, k != i:

* r_ij = x_j - x_i
* r_ik = x_k - x_i
* cos_theta = dot(r_ij, r_ik) / (||r_ij|| ||r_ik|| + eps)
* P2 = 0.5 * (3 * cos_theta^2 - 1)
* E3 += m_i * m_j * m_k * exp(-(||r_ij||^2 + ||r_ik||^2) / sigma3^2) * P2

Force:
F = -grad_x E, computed with PyTorch autograd during sample generation.

Important:

* The energy must be invariant under rotation and translation.
* The resulting force must satisfy:
    F(Rx + t) ≈ R F(x).
* Add tests for this property.

5. Dense edge graph
    Implement models/edge_graph.py:

* Build dense directed edges excluding self-edges.
* Return edge source indices, destination indices, batch indices, edge vectors, distances.
* Support flattening [B, N, N] edges into [E].
* Provide dense aggregation utilities using index_add_.
* Provide per-destination softmax for attention logits without torch-scatter.
* The edge softmax should work for logits [E, H], grouped by (batch, dst_node).

6. Baselines

6.1 CoordMLP
File:
models/baselines/coord_mlp.py

* Input absolute coordinates and scalar node attributes.
* Flatten fixed-N input.
* Predict [B, N, 3].
* Intentionally non-equivariant.

6.2 VanillaGraphTransformer
File:
models/baselines/vanilla_graph_transformer.py

* Node embedding from [x_i, z_i].
* Standard multi-head attention over nodes.
* Linear vector head.
* Intentionally non-equivariant.

6.3 EGNN
File:
models/baselines/egnn.py

* EGNN-style scalar messages from h_i, h_j, d2_ij, and scalar attributes.
* Force output as scalar-gated sum of relative vectors.
* Equivariant baseline.
* No spherical harmonics or higher-order irreps.

7. Equivariant TFN convolution

File:
models/equivariant/tfn_conv.py

Implement a Tensor Field Network-style equivariant convolution layer using e3nn:

* Input node features with irreps irreps_in.
* Edge spherical harmonics Y(r_ij) with irreps_sh.
* Radial MLP produces weights for o3.FullyConnectedTensorProduct.
* Tensor product:
    message_ij = TP(h_j, Y(r_ij), radial_weights(d_ij))
* Aggregate messages to destination nodes.
* Output irreps irreps_out.

The layer must not use absolute coordinates as scalar features.
Only edge vectors and scalar node attributes are allowed.

Test:

* For random scalar input and random positions, verify:
    layer(Rx + t, transformed_features) ≈ transform_features(layer(x, features), R).

8. SE(3) attention head

File:
models/equivariant/se3_attention.py

Implement SE3AttentionHead.

Inputs:

* x: [B, N, 3]
* features: [B, N, irreps_in.dim]
* optional mask

Parameters:

* irreps_in
* irreps_value
* lmax
* num_query_channels
* radial_hidden_dim
* radial_num_basis
* attention_dropout

Query:

* Use o3.Linear(irreps_in, f"{num_query_channels}x0e")
* This produces scalar invariant query features per node.

Key:

* Use an edge tensor product:
    key_ij = TP_key(h_j, Y(r_ij), radial_key_weights(d_ij))
* Key irreps must be f"{num_query_channels}x0e".

Value:

* Use an edge tensor product:
    value_ij = TP_value(h_j, Y(r_ij), radial_value_weights(d_ij))
* Value irreps should be irreps_value.

Attention:

* score_ij = dot(q_i, key_ij) / sqrt(num_query_channels) + radial_bias(d_ij)
* Softmax over incoming neighbors j for each destination node i.
* Exclude self-edges.
* Attention weights are scalar invariant.
* Output:
    out_i = sum_j alpha_ij * value_ij

Test:

* Attention output must transform according to irreps_value under random rotations.

9. Multi-head SE(3) attention

Implement SE3MultiHeadAttention:

* Use ModuleList of SE3AttentionHead.
* Concatenate head outputs.
* Use o3.Linear(concat_irreps, irreps_out) to mix heads.
* Support num_heads.

This avoids complex head-major irrep slicing.

10. SE(3)-Transformer block

File:
models/equivariant/se3_transformer_block.py

Implement:

* pre-norm or post-norm over irreps
* multi-head SE(3) attention
* residual connection where irreps match
* equivariant feed-forward network:
    * o3.Linear
    * e3nn-compatible gate or norm activation
    * o3.Linear
* residual connection

If layer norm is tricky, implement a safe irrep-aware norm:

* scalar channels can use standard LayerNorm.
* non-scalar irreps can use norm-based scaling or e3nn BatchNorm/NormActivation.
* Do not apply ordinary elementwise nonlinearities to non-scalar irrep components.

Scientific invariant:
Never apply ReLU/GELU directly to vector/tensor irrep components.

11. Full model

File:
models/equivariant/se3_force_transformer.py

Implement SE3ForceTransformer.

Inputs:

* x: [B, N, 3]
* z: [B, N, C]

Architecture:

* Initial node embedding from scalar attributes only:
    z -> irreps_hidden
    where initial non-scalar channels may start as zeros.
* Stack num_layers SE(3)-Transformer blocks.
* Output head:
    o3.Linear(irreps_hidden, "1x1o")
* Return force vectors [B, N, 3].

Configurable:

* lmax
* channels_by_l
* num_layers
* num_heads
* num_query_channels
* radial_num_basis
* radial_hidden_dim
* dropout
* use_attention=True/False
* use_gate=True/False
* dataset

Important:

* Do not concatenate raw coordinates to scalar node features in SE3ForceTransformer.
* Coordinates may only enter through edge vectors and spherical harmonics.
* The final output must be a polar vector irrep "1x1o".

12. Training

Implement:

* MSE force loss:
    loss = mean((F_pred - F_target)^2)
* Optional energy loss later, but not required in MVP.
* AdamW optimizer.
* train/val/test split.
* checkpoint saving.
* metrics JSON/CSV.
* config-driven runs.

Scripts:

* scripts/train.py --config configs/angular_force_se3_l2.yaml
* scripts/evaluate.py --config <config> --checkpoint <path>
* scripts/run_smoke.py

13. Evaluation

Implement:

* canonical test MSE
* rotated-translated test MSE
* equivariance error:
    ||model(Rx+t,z) - R model(x,z)|| / (||R model(x,z)|| + eps)
* parameter count
* runtime per batch
* save metrics to JSON

Script:
python scripts/eval_equivariance.py --model se3_transformer --config configs/small_cpu.yaml

Expected:

* SE3ForceTransformer: low equivariance error, roughly 1e-5 to 1e-4 depending on float precision and e3nn tolerance.
* EGNN: low final force equivariance error.
* CoordMLP and VanillaGraphTransformer: much larger equivariance error.

14. Ablations

Implement configs and CLI support for:

* lmax = 0
* lmax = 1
* lmax = 2
* lmax = 3
* no attention: TFNConvNet
* no gate
* different number of layers
* different number of heads
* central-force vs angular-force dataset
* canonical vs rotated-OOD test

Script:
python scripts/run_ablation_grid.py --configs configs/ablation_lmax0.yaml configs/ablation_lmax1.yaml configs/ablation_lmax2.yaml configs/ablation_lmax3.yaml

15. Tests

Implement pytest tests:

test_rotations.py

* orthogonality
* determinant +1

test_pairwise.py

* translation invariance of relative vectors
* rotation/translation invariance of distances

test_spherical_harmonics.py

* check spherical harmonics transformation using e3nn Wigner-D matrices

test_central_force_equivariance.py

* central force target transforms equivariantly

test_angular_force_equivariance.py

* angular potential energy is invariant
* angular force transforms equivariantly

test_model_shapes.py

* all baselines and SE3ForceTransformer return [B, N, 3]

test_tfn_layer_equivariance.py

* TFNConv layer output transforms by the correct irreps

test_se3_attention_equivariance.py

* SE3 attention output transforms by the correct irreps

test_se3_force_transformer_equivariance.py

* final force prediction satisfies:
    relative_error(model(Rx+t,z), R model(x,z)) < tolerance

test_training_smoke.py

* one tiny training loop runs without error

16. README

README must include:

* Motivation:
    ordinary neural networks depend on the arbitrary choice of coordinate system.
* Research question.
* Prediction task formulation.
* SE(3) equivariance definition:
    f(RX+t,z)=R f(X,z).
* Explanation of irreps:
    * scalar 0e
    * vector 1o
    * rank-2 tensor 2e
* Spherical harmonics role.
* Tensor product role.
* SE(3)-attention equations:
    * query scalar
    * key scalar
    * value equivariant
    * invariant attention score
    * equivariant weighted sum
* Dataset equations:
    * central force
    * angular potential with P2(cos theta)
* Proof sketch of equivariance.
* Install command.
* Smoke test command.
* Training commands.
* Evaluation commands.
* Ablation commands.
* Expected output examples.
* A note that the repo uses e3nn for group-theoretic primitives but implements the model architecture, tasks, tests, and training pipeline directly.

17. AGENTS.md

Create AGENTS.md with:

* repo overview
* how to run tests
* how to run smoke training
* style rules
* no heavy dependencies without approval
* scientific invariants:
    * no raw coordinates as scalar features in SE3 model
    * no ordinary elementwise nonlinearities on non-scalar irreps
    * attention logits must be invariant scalars
    * values must be equivariant
    * output force must be "1x1o"
    * equivariance tests must remain active
* done criteria

After implementing, run:

1. pip install -e .
2. pytest
3. python scripts/run_smoke.py
4. python scripts/eval_equivariance.py --model se3_transformer --config configs/small_cpu.yaml

Fix all errors until these pass.
