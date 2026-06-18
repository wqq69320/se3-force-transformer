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
    encode_frame_sequence_with_ffmpeg,
    ensure_material,
    load_sample,
    panel_offsets,
    parse_with,
    render_options_from_args,
    render_output,
    setup_scene,
    validate_frame_sequence,
    vector_sub,
    write_python_preview_sequence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a cinematic ground truth vs SE3 vs PaiNN-lite comparison.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    add_render_options(parser, default_frames=240)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = parse_with(build_parser(), argv)
    sample = load_sample(args.input)
    options = render_options_from_args(args)
    try:
        import bpy
    except Exception as exc:
        frame_dir = args.frame_output_dir or "outputs/visuals/frames/model_comparison_cinematic_hd"
        path = write_python_preview_sequence(
            sample,
            frame_dir,
            scene="model_comparison_cinematic",
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
    offsets = panel_offsets(3, spacing=2.35)
    panels = [
        add_molecule(bpy, sample, force_key="target_forces", offset=offsets[0], label="Ground Truth", target_radius=0.86, max_arrow_length=2.65, arrow_radius=0.028, label_size=0.13),
        add_molecule(bpy, sample, force_key="se3_forces", offset=offsets[1], label="SE3", target_radius=0.86, max_arrow_length=2.65, arrow_radius=0.028, label_size=0.13),
        add_molecule(bpy, sample, force_key="painn_forces", offset=offsets[2], label="PaiNN-lite", target_radius=0.86, max_arrow_length=2.65, arrow_radius=0.028, label_size=0.13),
    ]
    for panel in panels:
        animate_objects_fade(panel["atoms"] + panel["bonds"], 1, 40)
        animate_objects_fade(panel["arrows"], 40, 100)
        animate_objects_scale(panel["arrows"], 40, 100, 0.01, 1.0)

    residual_mat = ensure_material(bpy, "comparison_residual_red", FORCE_COLORS["residual"], emission=True, emission_strength=0.4)
    residual_objects = []
    for panel, force_key in [(panels[1], "se3_forces"), (panels[2], "painn_forces")]:
        residuals = [vector_sub(pred, target) for pred, target in zip(sample[force_key], sample["target_forces"])]
        residual_objects.extend(add_force_arrows(bpy, panel["positions"], residuals, material=residual_mat, pos_scale=panel["pos_scale"], max_arrow_length=1.70, radius=0.024, start_offset=0.20, name="comparison_residual"))
    animate_objects_fade(residual_objects, 100, 160)
    animate_objects_scale(residual_objects, 100, 160, 0.01, 1.0)

    camera = add_camera(bpy, location=(0, -13.2, 5.2), target=(0, 0, 0), ortho_scale=7.8)
    animate_camera_orbit(camera, radius=13.2, height=5.2, start_frame=1, end_frame=options.frames, start_angle=-0.08, end_angle=0.16)
    render_output(bpy, args.output, options)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
