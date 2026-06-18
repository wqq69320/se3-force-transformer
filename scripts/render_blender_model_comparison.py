#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_visualization_common import add_camera, add_molecule, add_render_options, add_text, load_sample, parse_with, render_options_from_args, render_output, setup_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render ground truth vs SE3 vs PaiNN-lite force arrows in Blender.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    add_render_options(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = parse_with(build_parser(), argv)
    import bpy

    sample = load_sample(args.input)
    options = render_options_from_args(args)
    if args.resolution_x == 1920 and args.resolution_y == 1080:
        options.resolution_x = 2400
        options.resolution_y = 1200
    setup_scene(bpy, options=options)
    add_molecule(bpy, sample, force_key="target_forces", offset=(-5.2, 0.0, 0.0), label="Ground Truth")
    add_molecule(bpy, sample, force_key="se3_forces", offset=(0.0, 0.0, 0.0), label="SE3 prediction")
    add_molecule(bpy, sample, force_key="painn_forces", offset=(5.2, 0.0, 0.0), label="PaiNN-lite prediction")
    add_text(bpy, "Both models are SE(3)-equivariant.", (0.0, -3.7, 3.08), size=0.24)
    add_text(
        bpy,
        "Current rMD17 aspirin benchmark: PaiNN-lite learned forces better than local SE3.",
        (0.0, -3.7, 2.72),
        size=0.18,
    )
    if sample.get("metadata", {}).get("placeholder_note"):
        add_text(bpy, "Predictions are placeholder visual aids unless checkpoint metadata says otherwise.", (0.0, -3.7, 2.42), size=0.16)
    add_camera(bpy, location=(0, -10.5, 4.6), target=(0, 0, 0))
    render_output(bpy, args.output, options)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
