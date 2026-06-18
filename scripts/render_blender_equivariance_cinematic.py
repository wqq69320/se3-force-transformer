#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_visualization_common import (
    FORCE_COLORS,
    add_camera,
    add_force_arrows,
    add_molecule,
    add_render_options,
    animate_camera_orbit,
    animate_objects_fade,
    animate_objects_scale,
    center_scale_positions,
    encode_frame_sequence_with_ffmpeg,
    ensure_material,
    load_sample,
    parse_with,
    render_options_from_args,
    render_output,
    setup_scene,
    validate_frame_sequence,
    vector_sub,
    write_python_preview_sequence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an SE(3) equivariance cinematic Blender animation.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force-key", default="se3_forces", choices=["target_forces", "se3_forces", "painn_forces"])
    add_render_options(parser, default_frames=300)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = parse_with(build_parser(), argv)
    sample = load_sample(args.input)
    options = render_options_from_args(args)
    try:
        import bpy
    except Exception as exc:
        frame_dir = args.frame_output_dir or "outputs/visuals/frames/equivariance_cinematic_hd"
        path = write_python_preview_sequence(
            sample,
            frame_dir,
            scene="equivariance_cinematic",
            frames=options.frames,
            fps=options.fps,
            resolution_x=options.resolution_x,
            resolution_y=options.resolution_y,
        )
        print(f"Blender unavailable ({exc.__class__.__name__}: {exc}); wrote high-resolution PNG sequence fallback to {path}")
        validate_frame_sequence(path)
        encode_frame_sequence_with_ffmpeg(path, args.output, fps=options.fps)
        return

    setup_scene(bpy, options=options)
    left_offset = (-1.35, 0.0, 0.0)
    right_offset = (1.35, 0.0, 0.0)
    rotated_force_key = "rotated_" + args.force_key
    expected_key = "expected_" + rotated_force_key
    left = add_molecule(
        bpy,
        sample,
        positions_key="positions",
        force_key=args.force_key,
        offset=left_offset,
        label="Original frame",
        target_radius=1.28,
        max_arrow_length=2.80,
        arrow_radius=0.038,
        label_size=0.18,
    )
    right = add_molecule(
        bpy,
        sample,
        positions_key="rotated_positions",
        force_key=rotated_force_key,
        offset=right_offset,
        label="Transformed frame",
        target_radius=1.28,
        max_arrow_length=2.80,
        arrow_radius=0.038,
        label_size=0.18,
    )
    animate_objects_fade(left["all"], 1, 40)
    animate_objects_scale(left["arrows"], 1, 40, 0.01, 1.0)
    animate_objects_fade(right["all"], 40, 90)
    animate_objects_scale(right["all"], 40, 90, 0.01, 1.0)

    ghost_forces = sample.get(expected_key, sample[rotated_force_key])
    ghost_positions, pos_scale = center_scale_positions(sample["rotated_positions"], target_radius=1.28)
    ghost_positions = [[pos[i] + right_offset[i] for i in range(3)] for pos in ghost_positions]
    ghost_mat = ensure_material(bpy, "ghost_RfX_light_green", (0.65, 1.0, 0.65, 0.42), emission=True, emission_strength=0.25)
    ghost_objects = add_force_arrows(bpy, ghost_positions, ghost_forces, material=ghost_mat, pos_scale=pos_scale, max_arrow_length=2.35, radius=0.032, start_offset=0.23, name="ghost_RfX")
    animate_objects_fade(ghost_objects, 130, 170)
    animate_objects_scale(ghost_objects, 130, 170, 0.01, 1.0)

    residuals = [vector_sub(actual, expected) for actual, expected in zip(sample[rotated_force_key], ghost_forces)]
    residual_mat = ensure_material(bpy, "residual_red", FORCE_COLORS["residual"], emission=True, emission_strength=0.4)
    residual_objects = add_force_arrows(bpy, ghost_positions, residuals, material=residual_mat, pos_scale=pos_scale, max_arrow_length=1.65, radius=0.028, start_offset=0.23, name="residual")
    animate_objects_fade(residual_objects, 170, options.frames)
    animate_objects_scale(residual_objects, 170, options.frames, 0.01, 1.0)

    camera = add_camera(bpy, location=(0, -12.4, 5.0), target=(0, 0, 0), ortho_scale=6.8)
    animate_camera_orbit(camera, radius=12.4, height=5.0, start_frame=1, end_frame=options.frames, start_angle=-0.12, end_angle=0.22)
    render_output(bpy, args.output, options)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
