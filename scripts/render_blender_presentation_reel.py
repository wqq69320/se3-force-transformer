#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a concat list for the SE(3) force presentation reel.")
    parser.add_argument("--output", default="outputs/visuals/se3_force_presentation_reel.mp4")
    parser.add_argument("--concat-list", default="outputs/visuals/concat_list.txt")
    parser.add_argument(
        "--clips",
        nargs="*",
        default=[
            "outputs/visuals/force_scene_cinematic.mp4",
            "outputs/visuals/se3_equivariance_cinematic.mp4",
            "outputs/visuals/model_comparison_se3_painn_cinematic.mp4",
        ],
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    concat_path = Path(args.concat_list)
    concat_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"file '{Path(clip).resolve()}'" for clip in args.clips]
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {concat_path}")
    print(f"ffmpeg -f concat -safe 0 -i {concat_path} -c copy {args.output}")


if __name__ == "__main__":
    main()
