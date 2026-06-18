#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_visualization_common import (
    add_camera,
    add_molecule,
    add_render_options,
    add_text,
    animate_camera_orbit,
    animate_objects_fade,
    animate_objects_scale,
    encode_frame_sequence_with_ffmpeg,
    load_sample,
    parse_with,
    render_options_from_args,
    render_output,
    setup_scene,
    validate_frame_sequence,
    write_python_preview_sequence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a cinematic Blender animation of aspirin and target force arrows.")
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
        frame_dir = args.frame_output_dir or "outputs/visuals/frames/force_scene_cinematic_hd"
        path = write_python_preview_sequence(
            sample,
            frame_dir,
            scene="force_scene_cinematic",
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
    molecule = add_molecule(
        bpy,
        sample,
        force_key="target_forces",
        target_radius=2.35,
        max_arrow_length=0.92,
        arrow_radius=0.050,
        label_size=0.30,
    )
    animate_objects_fade(molecule["atoms"], 1, 30)
    animate_objects_scale(molecule["atoms"], 1, 30, 0.01, 1.0)
    animate_objects_fade(molecule["bonds"], 30, 70)
    animate_objects_scale(molecule["bonds"], 30, 70, 0.01, 1.0)
    animate_objects_fade(molecule["arrows"], 70, 120)
    animate_objects_scale(molecule["arrows"], 70, 120, 0.01, 1.0)
    camera = add_camera(bpy, location=(0, -10.2, 4.7), target=(0, 0, 0), lens=38.0)
    animate_camera_orbit(camera, radius=10.2, height=4.7, start_frame=1, end_frame=options.frames, target=(0, 0, 0), start_angle=-0.45, end_angle=1.35)
    render_output(bpy, args.output, options)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
