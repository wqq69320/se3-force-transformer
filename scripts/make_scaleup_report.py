#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecular-summary", default=None)
    parser.add_argument("--output", default="outputs/scaleup_report.md")
    args = parser.parse_args()
    molecular_status = "not provided"
    if args.molecular_summary and Path(args.molecular_summary).exists():
        molecular_status = f"summary available at `{args.molecular_summary}`"
    text = f"""# Scale-Up Report

## Research Goal

Move from synthetic angular force fields toward real molecular force-field evidence while preserving SE(3) equivariance and honest reporting.

## Current Synthetic Benchmark Status

Synthetic scale-up infrastructure is available through `scripts/run_scale_sweep.py`.

## Real Molecular Dataset Status

Molecular summary status: {molecular_status}. Local rMD17/MD22 files are required for real molecular claims.

## Model Families Tested

The code supports molecular EGNN-style, TFN-labeled, and SE3-labeled cutoff force-field models with direct-force and energy-force modes.

## Claim Readiness Checklist

- rMD17 included: depends on local benchmark input
- at least 3 seeds: inspect summary `n`
- at least 2 molecules: inspect summary molecule rows
- EGNN/TFN/SE3 compared: inspect summary config rows
- parameter/runtime reported: supported
- energy/force metrics reported: supported
- equivariance error <= 1e-5 for equivariant models: must be verified per run
- at least one scientific demo generated: run demo scripts
- no overclaim beyond dataset scale: required

## Limitations

No larger MD22/OC20-style claim is supported without local data and larger benchmarks. If simpler equivariant models outperform full SE3 attention, report that result honestly.
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
