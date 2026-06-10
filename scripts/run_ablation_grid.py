#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from se3force.evaluation.ablations import run_ablation_configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/ablation_grid.json")
    args = parser.parse_args()
    results = run_ablation_configs(args.configs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
