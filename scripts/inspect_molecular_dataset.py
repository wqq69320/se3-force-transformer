#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from se3force.data.rmd17 import load_rmd17_npz
from se3force.data.md22 import load_md22_local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rmd17", "md22"], required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--molecule", required=True)
    args = parser.parse_args()

    arrays = load_rmd17_npz(args.path) if args.dataset == "rmd17" else load_md22_local(args.path)
    pos = torch.as_tensor(arrays["positions"])
    forces = torch.as_tensor(arrays["forces"])
    energies = arrays.get("energies")
    print(f"dataset={args.dataset}")
    print(f"molecule={args.molecule}")
    print(f"frames={pos.shape[0]}")
    print(f"atoms={pos.shape[1]}")
    if energies is not None:
        e = torch.as_tensor(energies, dtype=torch.float32).reshape(-1)
        print(f"energy_range=[{float(e.min()):.6g}, {float(e.max()):.6g}]")
    else:
        print("energy_range=missing")
    print(f"force_mean={float(forces.mean()):.6g}")
    print(f"force_std={float(forces.std()):.6g}")
    print("unit_length=Angstrom")
    print("unit_energy=kcal/mol unless dataset metadata says otherwise")
    print("unit_force=kcal/mol/Angstrom unless dataset metadata says otherwise")
    train = max(1, int(0.8 * pos.shape[0]))
    val = max(1, int(0.1 * pos.shape[0]))
    test = max(1, pos.shape[0] - train - val)
    print(f"recommended_split=train:{train} val:{val} test:{test}")


if __name__ == "__main__":
    main()
