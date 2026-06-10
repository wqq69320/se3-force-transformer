# Agent Guide

## Repo Overview

This repository implements `se3force`, a PyTorch/e3nn package for SE(3)-equivariant force prediction. Source code lives under `src/se3force`, configs under `configs`, scripts under `scripts`, and tests under `tests`.

## Core Commands

Install:

```bash
pip install -e .
```

Run tests:

```bash
pytest
```

Smoke training:

```bash
python scripts/run_smoke.py
```

Equivariance check:

```bash
python scripts/eval_equivariance.py --model se3_transformer --config configs/small_cpu.yaml
```

Main training entry:

```bash
python scripts/train.py --config configs/angular_force_se3_l2.yaml
```

## Style Rules

Keep dependencies lightweight. Do not add PyTorch Geometric, DGL, Lightning, Hydra, wandb, torch-scatter, or other heavy frameworks without explicit approval. Prefer dense all-pairs graph utilities for the MVP.

Use simple YAML configs and deterministic seeds. Keep scripts runnable from the repository root.

## Scientific Invariants

- Do not feed raw coordinates as scalar features to `SE3ForceTransformer`.
- Coordinates may enter the SE(3) model only through relative edge vectors, distances, and spherical harmonics.
- Do not apply ordinary elementwise nonlinearities to vector or tensor irrep components.
- Attention logits must be invariant scalars.
- Attention values must be equivariant irreps.
- The force output head must be `"1x1o"`.
- Equivariance tests must remain active and meaningful.

## Done Criteria

A change is done only when editable install, pytest, smoke training, equivariance evaluation, and the relevant train/evaluate commands work for the touched path. Do not leave placeholder-only files, fake tests, or unimplemented TODO stubs.
