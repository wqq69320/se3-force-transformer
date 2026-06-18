import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from se3force.visualization.geometry import (
    force_scale_factor,
    infer_bonds,
    scale_forces,
    se3_transform_positions,
    se3_transform_vectors,
)


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(Path(path).parent))
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.modules.pop(name, None)
    return module


def write_fake_rmd17(path: Path) -> None:
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.0]],
            [[0.1, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.25, 0.0], [0.0, 0.0, 1.05]],
        ],
        dtype=np.float32,
    )
    forces = np.asarray(
        [
            [[0.0, 0.2, 0.0], [0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.3]],
            [[0.0, 0.1, 0.1], [0.2, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.2]],
        ],
        dtype=np.float32,
    )
    np.savez(path, R=positions, F=forces, z=np.asarray([6, 1, 8, 1], dtype=np.int64), E=np.zeros(2, dtype=np.float32))


def test_visualization_geometry_transform_and_scaling():
    positions = np.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0]])
    atomic_numbers = np.asarray([6, 1, 8])
    bonds = infer_bonds(positions, atomic_numbers)
    assert bonds.ndim == 2
    assert bonds.shape[1] == 2

    zero_forces = np.zeros((3, 3))
    factor = force_scale_factor(zero_forces)
    scaled, used_factor = scale_forces(zero_forces)
    assert np.isfinite(factor)
    assert np.isfinite(used_factor)
    assert np.isfinite(scaled).all()

    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.asarray([4.0, 5.0, 6.0])
    vectors = np.asarray([[1.0, 2.0, 3.0]])
    transformed_pos = se3_transform_positions(vectors, rotation, translation)
    transformed_vec = se3_transform_vectors(vectors, rotation)
    assert np.allclose(transformed_pos, vectors @ rotation.T + translation)
    assert np.allclose(transformed_vec, vectors @ rotation.T)
    assert not np.allclose(transformed_vec, transformed_pos)


def test_export_visualization_sample_schema_json_and_npz(tmp_path):
    exporter = load_script("export_visualization_sample", "scripts/export_visualization_sample.py")
    data_path = tmp_path / "fake_rmd17.npz"
    output_stem = tmp_path / "aspirin_frame_000_visualization"
    write_fake_rmd17(data_path)
    exporter.main(
        [
            "--data-path",
            str(data_path),
            "--molecule",
            "aspirin",
            "--frame-index",
            "0",
            "--output",
            str(output_stem),
        ]
    )
    json_path = output_stem.with_suffix(".json")
    npz_path = output_stem.with_suffix(".npz")
    assert json_path.exists()
    assert npz_path.exists()
    sample = json.loads(json_path.read_text(encoding="utf-8"))
    required = [
        "molecule",
        "frame_index",
        "unit_length",
        "unit_force",
        "positions",
        "atomic_numbers",
        "target_forces",
        "se3_forces",
        "painn_forces",
        "rotation_matrix",
        "translation",
        "rotated_positions",
        "rotated_target_forces",
        "rotated_se3_forces",
        "rotated_painn_forces",
        "expected_rotated_se3_forces",
        "expected_rotated_painn_forces",
        "metadata",
    ]
    for key in required:
        assert key in sample
    assert sample["metadata"]["se3_prediction_type"] == "placeholder_non_scientific"
    with np.load(npz_path) as data:
        for key in [
            "positions",
            "atomic_numbers",
            "target_forces",
            "se3_forces",
            "painn_forces",
            "rotation_matrix",
            "expected_rotated_se3_forces",
        ]:
            assert key in data.files
        assert data["positions"].shape == (4, 3)


def test_plotly_html_generation_without_plotly_requirement(tmp_path):
    exporter = load_script("export_visualization_sample_for_plot", "scripts/export_visualization_sample.py")
    plotter = load_script("visualize_force_plotly", "scripts/visualize_force_plotly.py")
    data_path = tmp_path / "fake_rmd17.npz"
    output_stem = tmp_path / "sample"
    write_fake_rmd17(data_path)
    exporter.main(["--data-path", str(data_path), "--molecule", "aspirin", "--frame-index", "0", "--output", str(output_stem)])
    html_path = tmp_path / "force_field_interactive.html"
    plotter.main(
        [
            "--input",
            str(output_stem.with_suffix(".json")),
            "--force-key",
            "target_forces",
            "--output",
            str(html_path),
            "--show-rotated",
        ]
    )
    html = html_path.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "target_forces" in html


def test_blender_scripts_import_and_parse_without_blender():
    force_scene = load_script("render_blender_force_scene", "scripts/render_blender_force_scene.py")
    equivariance = load_script("render_blender_equivariance_demo", "scripts/render_blender_equivariance_demo.py")
    comparison = load_script("render_blender_model_comparison", "scripts/render_blender_model_comparison.py")
    force_cinematic = load_script("render_blender_force_cinematic", "scripts/render_blender_force_cinematic.py")
    equivariance_cinematic = load_script("render_blender_equivariance_cinematic", "scripts/render_blender_equivariance_cinematic.py")
    comparison_cinematic = load_script("render_blender_model_comparison_cinematic", "scripts/render_blender_model_comparison_cinematic.py")
    reel = load_script("render_blender_presentation_reel", "scripts/render_blender_presentation_reel.py")
    assert force_scene.build_parser().parse_args(["--input", "in.json", "--output", "out.png"]).force_key == "target_forces"
    assert equivariance.build_parser().parse_args(["--input", "in.json", "--output", "out.png"]).force_key == "se3_forces"
    assert comparison.build_parser().parse_args(["--input", "in.json", "--output", "out.png"]).output == "out.png"
    force_args = force_cinematic.build_parser().parse_args(["--input", "in.json", "--output", "out.mp4"])
    equivariance_args = equivariance_cinematic.build_parser().parse_args(["--input", "in.json", "--output", "out.mp4"])
    comparison_args = comparison_cinematic.build_parser().parse_args(["--input", "in.json", "--output", "out.mp4", "--engine", "CYCLES"])
    assert force_args.frames == 240
    assert force_args.resolution_x == 1920
    assert force_args.resolution_y == 1080
    assert force_args.fps == 30
    assert force_args.samples == 64
    assert equivariance_args.frames == 300
    assert equivariance_cinematic.build_parser().parse_args(["--input", "in.json", "--output", "out.mp4", "--frames", "12"]).frames == 12
    assert comparison_args.frames == 240
    assert comparison_args.engine == "CYCLES"
    assert reel.build_parser().parse_args(["--output", "reel.mp4"]).output == "reel.mp4"


def test_blender_common_animation_utilities_importable():
    common = load_script("blender_visualization_common", "scripts/blender_visualization_common.py")
    parser = common.argparse.ArgumentParser()
    common.add_render_options(parser)
    args = parser.parse_args(["--frames", "24", "--fps", "12", "--engine", "EEVEE", "--frame-output-dir", "frames"])
    options = common.render_options_from_args(args)
    assert options.frames == 24
    assert options.fps == 12
    assert options.engine == "EEVEE"
    assert options.frame_output_dir == "frames"
    assert common.panel_offsets(3, spacing=2.0) == [(-2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    assert common.vector_sub([3, 2, 1], [1, 1, 1]) == [2.0, 1.0, 0.0]


def test_cinematic_frame_validation_and_ffmpeg_command(tmp_path):
    from PIL import Image

    common = load_script("blender_visualization_common_validation", "scripts/blender_visualization_common.py")
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for index in range(1, 4):
        Image.new("RGB", (420, 445), color=(12, 16, 22)).save(frame_dir / f"frame_{index:04d}.png")

    report = common.validate_frame_sequence(frame_dir, warn=False)
    assert report["frame_count"] == 3
    assert report["width"] == 420
    assert report["height"] == 445
    assert any("width" in warning for warning in report["warnings"])
    assert any("height" in warning for warning in report["warnings"])
    assert any("frame count" in warning for warning in report["warnings"])

    command_text = common.format_shell_command(common.build_ffmpeg_command(frame_dir, tmp_path / "out.mp4", fps=30))
    assert "scale=1920:1080:flags=lanczos:force_original_aspect_ratio=decrease" in command_text
    assert "pad=1920:1080:(ow-iw)/2:(oh-ih)/2" in command_text
    assert "setsar=1,format=yuv420p" in command_text
    assert "-crf 14" in command_text
    assert "-preset slow" in command_text


def test_storyboard_and_readme_include_cinematic_commands(tmp_path):
    storyboard = load_script("make_visualization_storyboard", "scripts/make_visualization_storyboard.py")
    output = tmp_path / "storyboard.md"
    storyboard.main(["--output", str(output)])
    text = output.read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    for content in (text, readme):
        assert "--factory-startup --python-use-system-env" in content
        assert "render_blender_force_cinematic.py" in content
        assert "render_blender_equivariance_cinematic.py" in content
        assert "render_blender_model_comparison_cinematic.py" in content
        assert "--frame-output-dir outputs/visuals/frames/force_scene_cinematic_hd" in content
        assert "--resolution-x 1920" in content
        assert "--resolution-y 1080" in content
        assert "--samples 64" in content
        assert "draft assets only" in content
        assert "-crf 14" in content
        assert "-preset slow" in content
        assert "scale=1920:1080:flags=lanczos:force_original_aspect_ratio=decrease" in content
        assert "pad=1920:1080:(ow-iw)/2:(oh-ih)/2" in content
    assert "ffmpeg -y -framerate 30" in text
