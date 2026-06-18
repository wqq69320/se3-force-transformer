#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_visualization_common import add_camera, add_molecule, add_render_options, load_sample, parse_with, render_options_from_args, render_output, setup_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a molecule plus one force-arrow field in Blender.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force-key", default="target_forces", choices=["target_forces", "se3_forces", "painn_forces"])
    parser.add_argument("--label", default=None)
    add_render_options(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = parse_with(build_parser(), argv)
    import bpy

    sample = load_sample(args.input)
    options = render_options_from_args(args)
    setup_scene(bpy, options=options)
    label = args.label or f"{sample.get('molecule', 'molecule')} frame {sample.get('frame_index')} - {args.force_key}"
    add_molecule(bpy, sample, force_key=args.force_key, label=label)
    add_camera(bpy)
    render_output(bpy, args.output, options)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
