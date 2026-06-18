#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a presentation storyboard for SE(3) force visualizations.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-stem", default="outputs/visuals/data/aspirin_frame_000_visualization")
    return parser


def storyboard_text(data_stem: str) -> str:
    json_path = f"{data_stem}.json"
    return f"""# SE(3) Force Visualization Storyboard

## Scene List

1. Interactive force field inspection
2. Cinematic molecule and target-force render
3. SE(3) equivariance split screen
4. Ground truth vs SE3 vs PaiNN-lite model comparison
5. Scientific caution close

## Asset Commands

Export visualization data:

```bash
python3 scripts/export_visualization_sample.py \\
  --data-path data/rmd17/rmd17_aspirin.npz \\
  --molecule aspirin \\
  --frame-index 0 \\
  --output {data_stem}
```

Interactive HTML:

```bash
python3 scripts/visualize_force_plotly.py \\
  --input {json_path} \\
  --force-key target_forces \\
  --output outputs/visuals/force_field_interactive.html
```

Blender target-force still:

```bash
PYTHONPATH="$PWD/scripts" /Applications/Blender.app/Contents/MacOS/Blender \\
  -b --factory-startup --python-use-system-env \\
  --python scripts/render_blender_force_scene.py -- \\
  --input {json_path} \\
  --output outputs/visuals/force_scene.png
```

Blender equivariance still:

```bash
PYTHONPATH="$PWD/scripts" /Applications/Blender.app/Contents/MacOS/Blender \\
  -b --factory-startup --python-use-system-env \\
  --python scripts/render_blender_equivariance_demo.py -- \\
  --input {json_path} \\
  --output outputs/visuals/se3_equivariance_demo.png
```

Blender model comparison:

```bash
PYTHONPATH="$PWD/scripts" /Applications/Blender.app/Contents/MacOS/Blender \\
  -b --factory-startup --python-use-system-env \\
  --python scripts/render_blender_model_comparison.py -- \\
  --input {json_path} \\
  --output outputs/visuals/model_comparison_se3_painn.png
```

Cinematic force animation:

```bash
PYTHONPATH="$PWD/scripts" /Applications/Blender.app/Contents/MacOS/Blender \\
  -b --factory-startup --python-use-system-env \\
  --python scripts/render_blender_force_cinematic.py -- \\
  --input {json_path} \\
  --output outputs/visuals/force_scene_cinematic.mp4 \\
  --frame-output-dir outputs/visuals/frames/force_scene_cinematic_hd \\
  --frames 240 \\
  --fps 30 \\
  --resolution-x 1920 \\
  --resolution-y 1080 \\
  --engine EEVEE \\
  --samples 64
```

Cinematic SE(3) equivariance animation:

```bash
PYTHONPATH="$PWD/scripts" /Applications/Blender.app/Contents/MacOS/Blender \\
  -b --factory-startup --python-use-system-env \\
  --python scripts/render_blender_equivariance_cinematic.py -- \\
  --input {json_path} \\
  --output outputs/visuals/se3_equivariance_cinematic.mp4 \\
  --frame-output-dir outputs/visuals/frames/equivariance_cinematic_hd \\
  --frames 300 \\
  --fps 30 \\
  --resolution-x 1920 \\
  --resolution-y 1080 \\
  --engine EEVEE \\
  --samples 64
```

Cinematic model comparison:

```bash
PYTHONPATH="$PWD/scripts" /Applications/Blender.app/Contents/MacOS/Blender \\
  -b --factory-startup --python-use-system-env \\
  --python scripts/render_blender_model_comparison_cinematic.py -- \\
  --input {json_path} \\
  --output outputs/visuals/model_comparison_se3_painn_cinematic.mp4 \\
  --frame-output-dir outputs/visuals/frames/model_comparison_cinematic_hd \\
  --frames 240 \\
  --fps 30 \\
  --resolution-x 1920 \\
  --resolution-y 1080 \\
  --engine EEVEE \\
  --samples 64
```

Presentation reel concat list:

```bash
python3 scripts/render_blender_presentation_reel.py \\
  --output outputs/visuals/se3_force_presentation_reel.mp4
```

Use the `_hd` frame directories and MP4s for presentation. Earlier fallback directories without the `_hd` suffix are draft assets only; they may contain short, low-resolution preview frames.

If MP4 rendering falls back to PNG frames, convert a high-resolution sequence with:

```bash
ffmpeg -y -framerate 30 \\
  -i outputs/visuals/frames/equivariance_cinematic_hd/frame_%04d.png \\
  -vf "scale=1920:1080:flags=lanczos:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p" \\
  -c:v libx264 -crf 14 -preset slow -movflags +faststart \\
  outputs/visuals/se3_equivariance_cinematic.mp4
```

## Suggested Slide Placement

| slide | visual | purpose |
|---|---|---|
| 1 | `force_scene_cinematic.mp4` | establish aspirin as a 3D molecular force problem |
| 2 | `se3_equivariance_cinematic.mp4` | show `X' = R X + t` and `F' = R F` visually |
| 3 | `force_field_interactive.html` | inspect atom-level force directions live |
| 4 | `model_comparison_se3_painn_cinematic.mp4` | compare target, SE3, and PaiNN-lite arrow fields |
| 5 | report table excerpt | connect visuals to Phase 13 quantitative evidence |

## Exact Caption Text

- "Forces are vectors: translating the molecule does not translate force arrows."
- "SE(3)-equivariance requires `f(RX+t,Z) = R f(X,Z)`."
- "Green arrows are rMD17 target forces."
- "Cyan arrows are SE3 predictions when a checkpoint is provided; otherwise they are deterministic placeholders labeled as non-scientific."
- "Purple arrows are PaiNN-lite predictions when a checkpoint is provided; otherwise they are deterministic placeholders labeled as non-scientific."
- "Current rMD17 aspirin benchmark: PaiNN-lite learned forces better than local SE3."

## Scientific Caution Notes

- These visuals are qualitative geometry diagnostics, not proof of physical correctness.
- Placeholder predictions must not be used for quantitative model claims.
- Do not claim stable molecular dynamics from arrow plots or a single selected frame.
- Do not claim SE3 is best; Phase 13 found PaiNN-lite was the best local baseline.
- Coordinates may be centered and scaled for rendering only; scientific metrics are not altered.

## Recommended Presenter Narration

Start with aspirin as a point cloud with atom identities. Explain that the model should not care where the molecule is placed in the room, but its force vectors must rotate with the molecule. Move to the split-screen scene and read the equations directly: positions transform with rotation plus translation, while forces transform only by rotation. Then use the comparison panel to separate symmetry from accuracy: both SE3 and PaiNN-lite are equivariant, but the current rMD17 aspirin benchmark shows PaiNN-lite learned the forces better under the matched budget. Close by stating that the visuals make the geometry legible, while the CSV metrics and equivariance tests carry the quantitative claims.
"""


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(storyboard_text(args.data_stem), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
