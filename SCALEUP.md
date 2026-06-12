# Molecular Scale-Up Plan

Phase 4 moves `se3-force-transformer` from small synthetic force fields toward molecular force-field evidence.

## Claim Ladder

1. Synthetic correctness: equivariance, variable atom counts, cutoff graphs, and scale sweeps.
2. Local rMD17: force MAE/RMSE and energy metrics when labels exist.
3. Multi-molecule rMD17: compare EGNN, TFN-style, and SE3-style models across at least three seeds.
4. Larger molecules: MD22 local conversions or future larger datasets.
5. Scientific demos: force visualization, relaxation, short rollout, and energy-drift diagnostics.

## Commands

Synthetic scale smoke:

```bash
python3 scripts/run_scale_sweep.py --configs configs/molecular/scale_synthetic_n8.yaml configs/molecular/scale_synthetic_n12.yaml --seeds 0 --output outputs/scale_synthetic_smoke
```

Plot and report:

```bash
python3 scripts/plot_molecular_results.py --input outputs/scale_synthetic_smoke/summary_mean_std.csv --output outputs/scale_synthetic_smoke/plots
python3 scripts/make_molecular_report.py --input outputs/scale_synthetic_smoke/summary_mean_std.csv --output outputs/scale_synthetic_smoke/research_report.md
python3 scripts/make_scaleup_report.py --molecular-summary outputs/scale_synthetic_smoke/summary_mean_std.csv --output outputs/scaleup_report.md
```

## Compute Expectations

The default scale configs are CPU smoke runs. Real rMD17 and MD22 runs should increase epochs, frames, and model sizes only after the smoke path is passing.

## Reporting Rule

Do not claim broad molecular-force-field performance from synthetic scale-up alone. Real-data claims require local molecular datasets, multiple seeds, and baseline comparisons.
