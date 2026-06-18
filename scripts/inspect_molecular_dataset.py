#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from se3force.data.rmd17 import load_rmd17_npz
from se3force.data.md22 import load_md22_local
from se3force.training.molecular_overrides import add_molecular_override_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rmd17", "md22"], required=True)
    parser.add_argument("--path", default=None)
    parser.add_argument("--output", default=None)
    add_molecular_override_args(parser)
    args = parser.parse_args()

    path = args.data_path or args.path
    if not path:
        parser.error("--path or --data-path is required")
    molecule = args.molecule or "unknown"

    arrays = load_rmd17_npz(path) if args.dataset == "rmd17" else load_md22_local(path)
    if args.max_frames is not None:
        max_frames = int(args.max_frames)
        arrays["positions"] = arrays["positions"][:max_frames]
        arrays["forces"] = arrays["forces"][:max_frames]
        if arrays.get("energies") is not None:
            arrays["energies"] = arrays["energies"][:max_frames]
    pos = torch.as_tensor(arrays["positions"])
    forces = torch.as_tensor(arrays["forces"])
    energies = arrays.get("energies")
    metadata = {
        "dataset": args.dataset,
        "molecule": molecule,
        "path": path,
        "frames": int(pos.shape[0]),
        "atoms": int(pos.shape[1]),
        "unit_length": "Angstrom",
        "unit_energy": "kcal/mol unless dataset metadata says otherwise",
        "unit_force": "kcal/mol/Angstrom unless dataset metadata says otherwise",
        "split_type": args.split_type or "not_applied",
        "seed": args.seed,
        "device": args.device,
    }
    print(f"dataset={args.dataset}")
    print(f"molecule={molecule}")
    print(f"frames={pos.shape[0]}")
    print(f"atoms={pos.shape[1]}")
    if energies is not None:
        e = torch.as_tensor(energies, dtype=torch.float32).reshape(-1)
        metadata["energy_min"] = float(e.min())
        metadata["energy_max"] = float(e.max())
        print(f"energy_range=[{float(e.min()):.6g}, {float(e.max()):.6g}]")
    else:
        metadata["energy_min"] = None
        metadata["energy_max"] = None
        print("energy_range=missing")
    metadata["force_mean"] = float(forces.mean())
    metadata["force_std"] = float(forces.std())
    print(f"force_mean={float(forces.mean()):.6g}")
    print(f"force_std={float(forces.std()):.6g}")
    print("unit_length=Angstrom")
    print("unit_energy=kcal/mol unless dataset metadata says otherwise")
    print("unit_force=kcal/mol/Angstrom unless dataset metadata says otherwise")
    train = max(1, int(0.8 * pos.shape[0]))
    val = max(1, int(0.1 * pos.shape[0]))
    test = max(1, pos.shape[0] - train - val)
    metadata["recommended_split"] = {"train": train, "val": val, "test": test}
    print(f"recommended_split=train:{train} val:{val} test:{test}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
