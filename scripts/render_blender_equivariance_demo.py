#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_visualization_common import (
    add_arrow,
    add_camera,
    add_molecule,
    add_render_options,
    add_text,
    center_scale_positions,
    ensure_material,
    force_scale,
    load_sample,
    parse_with,
    render_options_from_args,
    render_output,
    setup_scene,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an SE(3) equivariance split-screen in Blender.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force-key", default="se3_forces", choices=["target_forces", "se3_forces", "painn_forces"])
    add_render_options(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = parse_with(build_parser(), argv)
    import bpy

    sample = load_sample(args.input)
    options = render_options_from_args(args)
    if args.resolution_x == 1920 and args.resolution_y == 1080:
        options.resolution_x = 2100
        options.resolution_y = 1100
    setup_scene(bpy, options=options)
    left_offset = (-3.8, 0.0, 0.0)
    right_offset = (3.8, 0.0, 0.0)
    rotated_force_key = "rotated_" + args.force_key
    add_molecule(bpy, sample, positions_key="positions", force_key=args.force_key, offset=left_offset, label="Original: X, F")
    add_molecule(
        bpy,
        sample,
        positions_key="rotated_positions",
        force_key=rotated_force_key,
        offset=right_offset,
        label="Transformed: X' = R X + t, F' = R F",
    )
    ghost_mat = ensure_material(bpy, "ghost_expected_rotation", (1.0, 1.0, 1.0, 0.32))
    expected_key = "expected_" + rotated_force_key
    expected_forces = sample.get(expected_key, sample[rotated_force_key])
    ghost_positions, pos_scale = center_scale_positions(sample["rotated_positions"])
    ghost_force_scale = force_scale(expected_forces)
    for pos, force in zip(ghost_positions, expected_forces):
        shifted = [pos[i] + right_offset[i] for i in range(3)]
        vector = [component * ghost_force_scale * pos_scale for component in force]
        add_arrow(bpy, shifted, vector, ghost_mat, radius=0.02, head_scale=0.14, name="ghost_RfX")
    eq = sample.get("equivariance_error")
    eq_text = "equivariance error unavailable for placeholder predictions" if eq is None else f"equivariance error = {eq:.3e}"
    add_text(bpy, "f(RX+t,Z) approx R f(X,Z)", (0.0, -3.6, 3.0), size=0.28)
    add_text(bpy, eq_text, (0.0, -3.6, 2.62), size=0.18)
    add_camera(bpy, location=(0, -9, 4.5), target=(0, 0, 0))
    render_output(bpy, args.output, options)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
