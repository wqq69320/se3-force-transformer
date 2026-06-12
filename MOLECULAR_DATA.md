# Molecular Data

No datasets are committed to this repository. Put local files under `data/` or `datasets/`, both ignored by git.

## Supported Local Formats

rMD17 loader supports `.npz` files with flexible keys:

- positions: `R`, `coords`, `positions`
- energies: `E`, `energies`, `energy`
- forces: `F`, `forces`
- atomic numbers: `z`, `Z`, `atomic_numbers`, `nuclear_charges`

MD22 scaffold supports local `.npz` and optional HDF5 files. HDF5 requires optional `h5py`.

## Units

The code records units in metrics metadata. Defaults are:

- length: Angstrom
- energy: kcal/mol
- force: kcal/mol/Angstrom

Adjust config fields when using eV or other source units.

## Inspect A Local Dataset

```bash
python3 scripts/inspect_molecular_dataset.py --dataset rmd17 --path data/rmd17/aspirin.npz --molecule aspirin
```

Auto-downloads are intentionally not required. Prefer explicit local paths for reproducibility.
