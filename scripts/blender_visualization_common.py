from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ATOM_COLORS = {
    1: (0.86, 0.86, 0.86, 1.0),
    6: (0.02, 0.02, 0.025, 1.0),
    7: (0.10, 0.22, 0.95, 1.0),
    8: (0.95, 0.05, 0.04, 1.0),
}

ATOM_RADII = {1: 0.18, 6: 0.28, 7: 0.27, 8: 0.26}
FORCE_COLORS = {
    "target_forces": (0.10, 0.90, 0.35, 1.0),
    "se3_forces": (0.05, 0.82, 1.0, 1.0),
    "painn_forces": (0.62, 0.25, 1.0, 1.0),
    "residual": (1.0, 0.05, 0.04, 1.0),
    "ghost": (1.0, 1.0, 1.0, 0.28),
}


@dataclass
class RenderOptions:
    resolution_x: int = 1920
    resolution_y: int = 1080
    fps: int = 30
    frames: int = 240
    engine: str = "EEVEE"
    device: str = "CPU"
    samples: int = 64
    transparent: bool = False
    frame_output_dir: str | None = None


def blender_script_args(argv: list[str] | None = None) -> list[str]:
    raw = list(sys.argv if argv is None else argv)
    return raw[raw.index("--") + 1 :] if "--" in raw else raw[1:]


def add_render_options(parser: argparse.ArgumentParser, *, default_frames: int = 240) -> None:
    parser.add_argument("--resolution-x", type=int, default=1920)
    parser.add_argument("--resolution-y", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=default_frames)
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE")
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--frame-output-dir")


def render_options_from_args(args: argparse.Namespace) -> RenderOptions:
    return RenderOptions(
        resolution_x=int(getattr(args, "resolution_x", 1920)),
        resolution_y=int(getattr(args, "resolution_y", 1080)),
        fps=int(getattr(args, "fps", 30)),
        frames=int(getattr(args, "frames", 240)),
        engine=str(getattr(args, "engine", "EEVEE")),
        device=str(getattr(args, "device", "CPU")),
        samples=int(getattr(args, "samples", 64)),
        transparent=bool(getattr(args, "transparent", False)),
        frame_output_dir=getattr(args, "frame_output_dir", None),
    )


def load_sample(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atom_color(z: int) -> tuple[float, float, float, float]:
    return ATOM_COLORS.get(int(z), (0.5, 0.5, 0.5, 1.0))


def atom_radius(z: int) -> float:
    return ATOM_RADII.get(int(z), 0.25)


def infer_bonds(positions: list[list[float]], atomic_numbers: list[int]) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = positions[j][0] - positions[i][0]
            dy = positions[j][1] - positions[i][1]
            dz = positions[j][2] - positions[i][2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            cutoff = 1.25 * (atom_radius(atomic_numbers[i]) / 0.35 + atom_radius(atomic_numbers[j]) / 0.35) + 0.15
            if 0.15 < distance <= cutoff:
                bonds.append((i, j))
    return bonds


def center_scale_positions(positions: list[list[float]], target_radius: float = 3.0) -> tuple[list[list[float]], float]:
    n = len(positions)
    center = [sum(p[k] for p in positions) / n for k in range(3)]
    centered = [[p[k] - center[k] for k in range(3)] for p in positions]
    max_radius = max((math.sqrt(sum(v * v for v in p)) for p in centered), default=1.0)
    scale = target_radius / max(max_radius, 1e-12)
    return [[scale * v for v in p] for p in centered], scale


def force_scale(forces: list[list[float]], max_arrow_length: float = 1.0) -> float:
    norms = sorted(math.sqrt(sum(v * v for v in f)) for f in forces)
    if not norms:
        return 1.0
    index = min(len(norms) - 1, max(0, int(0.9 * (len(norms) - 1))))
    ref = max(norms[index], 1e-12)
    return max_arrow_length / ref


def vector_sub(a, b) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def ensure_material(
    bpy,
    name: str,
    color: tuple[float, float, float, float],
    *,
    emission: bool = False,
    emission_strength: float = 0.0,
):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        material.diffuse_color = color
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = color
            bsdf.inputs["Roughness"].default_value = 0.42
            bsdf.inputs["Alpha"].default_value = color[3]
            if emission and "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = color
            if emission and "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = emission_strength
        material.blend_method = "BLEND" if color[3] < 1.0 else "OPAQUE"
    return material


def clear_scene(bpy) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def configure_engine(bpy, options: RenderOptions) -> None:
    scene = bpy.context.scene
    if options.engine == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = options.samples
        scene.cycles.device = options.device
    else:
        for engine_name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "EEVEE"):
            try:
                scene.render.engine = engine_name
                break
            except TypeError:
                continue
        if hasattr(scene, "eevee"):
            if hasattr(scene.eevee, "taa_render_samples"):
                scene.eevee.taa_render_samples = options.samples
            if hasattr(scene.eevee, "taa_samples"):
                scene.eevee.taa_samples = max(16, min(options.samples, 128))
            if hasattr(scene.eevee, "use_gtao"):
                scene.eevee.use_gtao = True
            if hasattr(scene.eevee, "gtao_distance"):
                scene.eevee.gtao_distance = 3.0
            if hasattr(scene.eevee, "gtao_factor"):
                scene.eevee.gtao_factor = 1.2
    scene.render.fps = options.fps
    scene.frame_start = 1
    scene.frame_end = max(1, options.frames)
    scene.render.resolution_x = options.resolution_x
    scene.render.resolution_y = options.resolution_y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = options.transparent


def setup_scene(bpy, *, resolution: tuple[int, int] = (1800, 1200), options: RenderOptions | None = None) -> None:
    clear_scene(bpy)
    scene = bpy.context.scene
    active_options = options or RenderOptions(resolution_x=resolution[0], resolution_y=resolution[1], engine="CYCLES", samples=96)
    configure_engine(bpy, active_options)
    scene.world.color = (0.015, 0.017, 0.022)
    bpy.ops.object.light_add(type="AREA", location=(0, -4, 7))
    key = bpy.context.object
    key.name = "Key Area Light"
    key.data.energy = 980
    key.data.size = 6.5
    bpy.ops.object.light_add(type="AREA", location=(-5, 3, 4))
    fill = bpy.context.object
    fill.name = "Cool Fill Light"
    fill.data.energy = 220
    fill.data.size = 5.0


def add_camera(bpy, location=(0, -8, 4.2), target=(0, 0, 0), *, lens: float = 44.0, ortho_scale: float | None = None) -> None:
    from mathutils import Vector

    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    if ortho_scale is not None:
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = ortho_scale
    else:
        camera.data.lens = lens
    bpy.context.scene.camera = camera
    return camera


def add_sphere(bpy, position, radius: float, material) -> None:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius, location=position)
    obj = bpy.context.object
    obj.data.materials.append(material)
    return obj


def add_cylinder_between(bpy, start, end, radius: float, material, name: str = "bond") -> None:
    from mathutils import Vector

    start_v = Vector(start)
    end_v = Vector(end)
    delta = end_v - start_v
    length = delta.length
    if length <= 1e-8:
        return
    mid = start_v + 0.5 * delta
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=length, location=mid)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(material)
    return obj


def add_cone_between(bpy, start, end, radius: float, material, name: str = "arrow_head") -> None:
    from mathutils import Vector

    start_v = Vector(start)
    end_v = Vector(end)
    delta = end_v - start_v
    length = delta.length
    if length <= 1e-8:
        return
    mid = start_v + 0.5 * delta
    bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=radius, radius2=0.0, depth=length, location=mid)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(material)
    return obj


def add_arrow(
    bpy,
    start,
    vector,
    material,
    *,
    radius: float = 0.035,
    head_scale: float = 0.24,
    start_offset: float = 0.0,
    name: str = "force_arrow",
) -> list:
    length = math.sqrt(sum(v * v for v in vector))
    if length <= 1e-8:
        return []
    direction = [v / length for v in vector]
    visible_start = [start[i] + direction[i] * start_offset for i in range(3)]
    shaft_len = max(0.0, length - head_scale)
    shaft_end = [visible_start[i] + direction[i] * shaft_len for i in range(3)]
    end = [visible_start[i] + vector[i] for i in range(3)]
    objects = []
    shaft = add_cylinder_between(bpy, visible_start, shaft_end, radius, material, name=name + "_shaft")
    head = add_cone_between(bpy, shaft_end, end, radius * 3.2, material, name=name + "_head")
    if shaft is not None:
        objects.append(shaft)
    if head is not None:
        objects.append(head)
    return objects


def add_text(bpy, text: str, location, *, size: float = 0.22, align: str = "CENTER") -> None:
    material = ensure_material(bpy, "text_white", (0.92, 0.94, 0.96, 1.0))
    bpy.ops.object.text_add(location=location, rotation=(math.radians(68), 0, 0))
    obj = bpy.context.object
    obj.data.body = text
    obj.data.align_x = align
    obj.data.size = size
    obj.data.materials.append(material)
    return obj


def add_empty(bpy, name: str, location=(0.0, 0.0, 0.0)):
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=location)
    obj = bpy.context.object
    obj.name = name
    return obj


def set_parent(objects: list, parent) -> None:
    for obj in objects:
        obj.parent = parent


def animate_object_scale(obj, start_frame: int, end_frame: int, start_scale=0.0, end_scale=1.0) -> None:
    obj.scale = (start_scale, start_scale, start_scale)
    obj.keyframe_insert(data_path="scale", frame=start_frame)
    obj.scale = (end_scale, end_scale, end_scale)
    obj.keyframe_insert(data_path="scale", frame=end_frame)


def animate_objects_scale(objects: list, start_frame: int, end_frame: int, start_scale=0.0, end_scale=1.0) -> None:
    for obj in objects:
        animate_object_scale(obj, start_frame, end_frame, start_scale, end_scale)


def _materials_for_object(obj) -> list:
    data = getattr(obj, "data", None)
    if data is None or not hasattr(data, "materials"):
        return []
    return [mat for mat in data.materials if mat is not None]


def set_material_alpha(material, alpha: float) -> None:
    material.diffuse_color = (material.diffuse_color[0], material.diffuse_color[1], material.diffuse_color[2], alpha)
    if material.use_nodes:
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None and "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
    material.blend_method = "BLEND" if alpha < 1.0 else "OPAQUE"


def animate_object_fade(obj, start_frame: int, end_frame: int, start_alpha=0.0, end_alpha=1.0) -> None:
    for material in _materials_for_object(obj):
        set_material_alpha(material, start_alpha)
        material.keyframe_insert(data_path="diffuse_color", frame=start_frame)
        set_material_alpha(material, end_alpha)
        material.keyframe_insert(data_path="diffuse_color", frame=end_frame)


def animate_objects_fade(objects: list, start_frame: int, end_frame: int, start_alpha=0.0, end_alpha=1.0) -> None:
    for obj in objects:
        animate_object_fade(obj, start_frame, end_frame, start_alpha, end_alpha)


def animate_camera_orbit(
    camera,
    *,
    radius: float,
    height: float,
    start_frame: int,
    end_frame: int,
    target=(0.0, 0.0, 0.0),
    start_angle=0.0,
    end_angle=2.0 * math.pi,
    steps: int = 7,
) -> None:
    from mathutils import Vector

    steps = max(2, int(steps))
    for index in range(steps):
        t = index / max(steps - 1, 1)
        frame = round(start_frame + t * (end_frame - start_frame))
        angle = start_angle + t * (end_angle - start_angle)
        camera.location = (radius * math.sin(angle), -radius * math.cos(angle), height)
        direction = Vector(target) - camera.location
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
    if camera.animation_data is not None and camera.animation_data.action is not None:
        for curve in camera.animation_data.action.fcurves:
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "LINEAR"


def panel_offsets(count: int, spacing: float = 5.2) -> list[tuple[float, float, float]]:
    if count <= 1:
        return [(0.0, 0.0, 0.0)]
    start = -0.5 * spacing * (count - 1)
    return [(start + i * spacing, 0.0, 0.0) for i in range(count)]


def add_force_arrows(
    bpy,
    positions: list[list[float]],
    forces: list[list[float]],
    *,
    material,
    pos_scale: float = 1.0,
    max_arrow_length: float = 1.0,
    radius: float = 0.035,
    start_offset: float = 0.0,
    name: str = "force_arrow",
) -> list:
    f_scale = force_scale(forces, max_arrow_length=max_arrow_length)
    objects = []
    for pos, force in zip(positions, forces):
        vec = [component * f_scale * pos_scale for component in force]
        objects.extend(add_arrow(bpy, pos, vec, material, radius=radius, start_offset=start_offset, name=name))
    return objects


def add_molecule(
    bpy,
    sample: dict,
    *,
    positions_key: str = "positions",
    force_key: str = "target_forces",
    offset=(0.0, 0.0, 0.0),
    label: str | None = None,
    arrow_color_key: str | None = None,
    target_radius: float = 3.0,
    max_arrow_length: float = 1.12,
    arrow_radius: float = 0.048,
    label_size: float = 0.28,
) -> dict[str, object]:
    positions_raw = sample[positions_key]
    atomic_numbers = [int(z) for z in sample["atomic_numbers"]]
    positions, pos_scale = center_scale_positions(positions_raw, target_radius=target_radius)
    positions = [[p[i] + offset[i] for i in range(3)] for p in positions]
    bonds = infer_bonds(positions_raw, atomic_numbers)
    bond_mat = ensure_material(bpy, "bond_soft_gray", (0.75, 0.75, 0.78, 1.0))
    atom_objects = []
    bond_objects = []
    arrow_objects = []
    text_objects = []
    for i, j in bonds:
        obj = add_cylinder_between(bpy, positions[i], positions[j], 0.048, bond_mat, name="bond")
        if obj is not None:
            bond_objects.append(obj)
    for pos, z in zip(positions, atomic_numbers):
        mat = ensure_material(bpy, f"atom_{z}", atom_color(z))
        atom_objects.append(add_sphere(bpy, pos, atom_radius(z) * 1.08, mat))
    forces = sample[force_key]
    f_scale = force_scale(forces, max_arrow_length=max_arrow_length)
    arrow_key = arrow_color_key or force_key
    arrow_mat = ensure_material(
        bpy,
        f"arrow_{arrow_key}",
        FORCE_COLORS.get(arrow_key, FORCE_COLORS["target_forces"]),
        emission=True,
        emission_strength=0.35,
    )
    for pos, force, z in zip(positions, forces, atomic_numbers):
        vec = [component * f_scale * pos_scale for component in force]
        arrow_objects.extend(
            add_arrow(
                bpy,
                pos,
                vec,
                arrow_mat,
                radius=arrow_radius,
                head_scale=0.30,
                start_offset=atom_radius(z) * 1.24,
                name=arrow_key,
            )
        )
    if label:
        text_objects.append(add_text(bpy, label, (offset[0], offset[1] - 3.35, offset[2] + 2.85), size=label_size))
    return {
        "atoms": atom_objects,
        "bonds": bond_objects,
        "arrows": arrow_objects,
        "texts": text_objects,
        "positions": positions,
        "pos_scale": pos_scale,
        "all": atom_objects + bond_objects + arrow_objects + text_objects,
    }


def render_frame_sequence(bpy, output_dir: str | Path, *, frames: int, frame_pattern: str = "frame_####.png") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for old_frame in output_path.glob("frame_*.png"):
        old_frame.unlink()
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_path / frame_pattern)
    scene.frame_start = 1
    scene.frame_end = frames
    bpy.ops.render.render(animation=True)
    return output_path


def build_ffmpeg_command(
    frame_dir: str | Path,
    output: str | Path,
    *,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> list[str]:
    frame_pattern = str(Path(frame_dir) / "frame_%04d.png")
    video_filter = (
        f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p"
    )
    return [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-preset",
        "slow",
        "-movflags",
        "+faststart",
        str(output),
    ]


def format_shell_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def print_ffmpeg_command(frame_dir: str | Path, output: str | Path, *, fps: int = 30) -> list[str]:
    command = build_ffmpeg_command(frame_dir, output, fps=fps)
    print("ffmpeg command for the high-resolution PNG sequence:", flush=True)
    print(format_shell_command(command), flush=True)
    return command


def encode_frame_sequence_with_ffmpeg(frame_dir: str | Path, output: str | Path, *, fps: int = 30) -> bool:
    command = print_ffmpeg_command(frame_dir, output, fps=fps)
    if shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg was not found on PATH; PNG frames were written but MP4 encoding was skipped.")
        return False
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"WARNING: ffmpeg exited with code {result.returncode}; PNG frames remain available at {frame_dir}.")
        return False
    return True


def validate_frame_sequence(
    frame_dir: str | Path,
    *,
    min_width: int = 1280,
    min_height: int = 720,
    min_frames: int = 150,
    warn: bool = True,
) -> dict[str, object]:
    path = Path(frame_dir)
    frames = sorted(path.glob("frame_*.png"))
    width = 0
    height = 0
    warnings: list[str] = []
    if frames:
        width, height = image_size(frames[0])
    else:
        warnings.append(f"frame count 0 is below {min_frames}")
    if width < min_width:
        warnings.append(f"frame width {width} is below {min_width}")
    if height < min_height:
        warnings.append(f"frame height {height} is below {min_height}")
    if len(frames) < min_frames:
        warnings.append(f"frame count {len(frames)} is below {min_frames}")
    if warn:
        for message in warnings:
            print(f"WARNING: {path}: {message}")
    return {
        "frame_dir": str(path),
        "frame_count": len(frames),
        "width": width,
        "height": height,
        "warnings": warnings,
    }


def image_size(path: str | Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return png_size(path)


def png_size(path: str | Path) -> tuple[int, int]:
    with Path(path).open("rb") as handle:
        header = handle.read(24)
    png_signature = b"\x89PNG\r\n\x1a\n"
    if len(header) >= 24 and header[:8] == png_signature and header[12:16] == b"IHDR":
        return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
    return 0, 0


def render_output(bpy, output: str | Path, options: RenderOptions | None = None) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    active_options = options or RenderOptions(frames=72, engine="CYCLES", samples=96, resolution_x=scene.render.resolution_x, resolution_y=scene.render.resolution_y)
    configure_engine(bpy, active_options)
    scene.render.filepath = str(output_path)
    if output_path.suffix.lower() == ".mp4":
        if active_options.frame_output_dir is not None:
            frame_dir = render_frame_sequence(bpy, active_options.frame_output_dir, frames=active_options.frames)
            validate_frame_sequence(frame_dir)
            encode_frame_sequence_with_ffmpeg(frame_dir, output_path, fps=active_options.fps)
            return
        scene.frame_start = 1
        scene.frame_end = max(1, active_options.frames)
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        try:
            bpy.ops.render.render(animation=True)
        except Exception:
            frame_dir = render_frame_sequence(bpy, output_path.parent / "frames" / output_path.stem, frames=active_options.frames)
            print(f"MP4 render failed; wrote PNG sequence to {frame_dir}")
            validate_frame_sequence(frame_dir)
            print_ffmpeg_command(frame_dir, output_path, fps=active_options.fps)
    else:
        scene.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(write_still=True)


def parse_with(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    return parser.parse_args(blender_script_args(argv))


def write_python_preview_sequence(
    sample: dict,
    output_dir: str | Path,
    *,
    scene: str,
    frames: int = 240,
    fps: int = 30,
    resolution_x: int = 1920,
    resolution_y: int = 1080,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    width = int(resolution_x)
    height = int(resolution_y)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for old_frame in output_path.glob("frame_*.png"):
        old_frame.unlink()

    atomic_numbers = [int(z) for z in sample["atomic_numbers"]]
    bonds = infer_bonds(sample["positions"], atomic_numbers)
    min_dim = min(width, height)
    title_font = _load_pillow_font(ImageFont, int(height * 0.046), bold=True)
    label_font = _load_pillow_font(ImageFont, int(height * 0.033), bold=True)
    body_font = _load_pillow_font(ImageFont, int(height * 0.026), bold=False)
    small_font = _load_pillow_font(ImageFont, int(height * 0.021), bold=False)
    background = _make_hd_background(Image, ImageDraw, width, height)

    def atom_rgba(z: int, alpha: int = 255) -> tuple[int, int, int, int]:
        palette = {
            1: (232, 232, 232, alpha),
            6: (54, 59, 66, alpha),
            7: (58, 92, 240, alpha),
            8: (235, 70, 60, alpha),
        }
        return palette.get(int(z), (150, 150, 150, alpha))

    def text(draw, xy, value: str, font, fill=(238, 242, 246, 255), align: str = "center") -> None:
        bbox = draw.textbbox((0, 0), value, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if align == "center":
            origin = (xy[0] - text_width / 2, xy[1] - text_height / 2)
        elif align == "right":
            origin = (xy[0] - text_width, xy[1] - text_height / 2)
        else:
            origin = (xy[0], xy[1] - text_height / 2)
        shadow = (origin[0] + 2, origin[1] + 2)
        draw.text(shadow, value, font=font, fill=(0, 0, 0, min(180, fill[3])))
        draw.text(origin, value, font=font, fill=fill)

    def project(point, *, yaw: float, elevation: float, center, scale: float) -> tuple[float, float, float]:
        x, y, z = point
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        x1 = cos_y * x + sin_y * y
        y1 = -sin_y * x + cos_y * y
        z1 = z
        cos_e = math.cos(elevation)
        sin_e = math.sin(elevation)
        depth = cos_e * y1 - sin_e * z1
        vertical = sin_e * y1 + cos_e * z1
        return center[0] + x1 * scale, center[1] - vertical * scale, depth

    def draw_arrow(draw, start, end, color, width_px: int) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 3.0:
            return
        ux = dx / length
        uy = dy / length
        px = -uy
        py = ux
        head_len = max(width_px * 3.2, min_dim * 0.022)
        head_width = max(width_px * 2.7, min_dim * 0.017)
        shaft_end = (end[0] - ux * head_len * 0.55, end[1] - uy * head_len * 0.55)
        glow = (color[0], color[1], color[2], min(90, color[3]))
        draw.line([start, shaft_end], fill=glow, width=max(width_px + 8, width_px * 2))
        draw.line([start, shaft_end], fill=color, width=width_px)
        head = [
            end,
            (end[0] - ux * head_len + px * head_width * 0.5, end[1] - uy * head_len + py * head_width * 0.5),
            (end[0] - ux * head_len - px * head_width * 0.5, end[1] - uy * head_len - py * head_width * 0.5),
        ]
        draw.polygon(head, fill=color)

    def draw_panel(
        image,
        *,
        center,
        yaw: float,
        positions_key: str,
        force_key: str,
        force_color: tuple[int, int, int, int],
        label: str,
        label_color: tuple[int, int, int, int],
        target_radius: float,
        panel_scale: float,
        label_y: float | None = None,
        force_progress: float = 1.0,
        max_arrow_length: float = 1.0,
        residual_forces: list[list[float]] | None = None,
        residual_progress: float = 0.0,
        ghost_forces: list[list[float]] | None = None,
        ghost_progress: float = 0.0,
    ) -> None:
        positions, _ = center_scale_positions(sample[positions_key], target_radius=target_radius)
        elevation = math.radians(21.0)
        items: list[tuple[float, int, str, object]] = []
        for i, j in bonds:
            p0 = project(positions[i], yaw=yaw, elevation=elevation, center=center, scale=panel_scale)
            p1 = project(positions[j], yaw=yaw, elevation=elevation, center=center, scale=panel_scale)
            items.append(((p0[2] + p1[2]) * 0.5, 0, "bond", (p0, p1)))
        arrow_width = max(7, int(min_dim * 0.009))
        add_arrow_items(items, positions, sample[force_key], force_progress, force_color, max_arrow_length, center, panel_scale, yaw, elevation, arrow_width)
        if ghost_forces is not None and ghost_progress > 0.0:
            add_arrow_items(items, positions, ghost_forces, ghost_progress, (150, 255, 170, 155), max_arrow_length * 0.92, center, panel_scale, yaw, elevation, max(5, int(arrow_width * 0.76)))
        if residual_forces is not None and residual_progress > 0.0:
            add_arrow_items(items, positions, residual_forces, residual_progress, (255, 58, 58, 240), 0.72, center, panel_scale, yaw, elevation, max(5, int(arrow_width * 0.72)))
        for pos, z in zip(positions, atomic_numbers):
            screen = project(pos, yaw=yaw, elevation=elevation, center=center, scale=panel_scale)
            radius = max(8, int(panel_scale * atom_radius(z) * 1.18))
            items.append((screen[2], 2, "atom", (screen, radius, atom_rgba(z))))
        draw = ImageDraw.Draw(image, "RGBA")
        for _, _, item_type, payload in sorted(items, key=lambda item: (item[0], item[1])):
            if item_type == "bond":
                p0, p1 = payload
                draw.line([(p0[0], p0[1]), (p1[0], p1[1])], fill=(178, 184, 190, 210), width=max(5, int(min_dim * 0.006)))
            elif item_type == "arrow":
                start, end, color, width_px = payload
                draw_arrow(draw, start, end, color, width_px)
            else:
                screen, radius, color = payload
                cx, cy, _ = screen
                draw.ellipse((cx - radius - 3, cy - radius - 3, cx + radius + 3, cy + radius + 3), fill=(0, 0, 0, 155))
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=(235, 240, 245, 190), width=max(2, int(radius * 0.11)))
                highlight = max(3, int(radius * 0.32))
                draw.ellipse((cx - radius * 0.42, cy - radius * 0.48, cx - radius * 0.42 + highlight, cy - radius * 0.48 + highlight), fill=(255, 255, 255, 82))
        if label:
            active_label_y = label_y if label_y is not None else min(center[1] + target_radius * panel_scale + min_dim * 0.09, height * 0.82)
            text(draw, (center[0], active_label_y), label, label_font, fill=label_color)

    def add_arrow_items(items, positions, forces, progress, color, max_arrow_length, center, panel_scale, yaw, elevation, width_px) -> None:
        progress = smoothstep(progress)
        if progress <= 0.0:
            return
        scale = force_scale(forces, max_arrow_length=max_arrow_length)
        for pos, force in zip(positions, forces):
            end_3d = [pos[i] + force[i] * scale * progress for i in range(3)]
            start = project(pos, yaw=yaw, elevation=elevation, center=center, scale=panel_scale)
            end = project(end_3d, yaw=yaw, elevation=elevation, center=center, scale=panel_scale)
            items.append((max(start[2], end[2]), 1, "arrow", ((start[0], start[1]), (end[0], end[1]), color, width_px)))

    for frame in range(1, int(frames) + 1):
        phase = 0.0 if frames <= 1 else (frame - 1) / (frames - 1)
        image = background.copy()
        draw = ImageDraw.Draw(image, "RGBA")
        text(draw, (width * 0.5, height * 0.07), _scene_title(scene), title_font)
        if scene == "equivariance_cinematic":
            yaw = -0.46 + 0.82 * phase
            text(draw, (width * 0.5, height * 0.135), "X' = R X + t    |    F' = R F", body_font, fill=(206, 222, 235, 255))
            expected = sample.get("expected_rotated_se3_forces", sample.get("rotated_se3_forces", sample["se3_forces"]))
            rotated = sample.get("rotated_se3_forces", sample["se3_forces"])
            residual = [vector_sub(actual, exp) for actual, exp in zip(rotated, expected)]
            draw_panel(
                image,
                center=(width * 0.28, height * 0.55),
                yaw=yaw,
                positions_key="positions",
                force_key="se3_forces",
                force_color=(28, 203, 255, 245),
                label="Original",
                label_color=(210, 242, 255, 255),
                target_radius=2.25,
                panel_scale=min_dim * 0.122,
                force_progress=1.0,
                max_arrow_length=0.92,
            )
            draw_panel(
                image,
                center=(width * 0.72, height * 0.55),
                yaw=yaw,
                positions_key="rotated_positions",
                force_key="rotated_se3_forces",
                force_color=(28, 203, 255, 245),
                label="Rotated and translated",
                label_color=(210, 242, 255, 255),
                target_radius=2.25,
                panel_scale=min_dim * 0.122,
                force_progress=(phase - 0.18) / 0.24,
                max_arrow_length=0.92,
                ghost_forces=expected,
                ghost_progress=(phase - 0.48) / 0.18,
                residual_forces=residual,
                residual_progress=(phase - 0.68) / 0.20,
            )
            text(draw, (width * 0.5, height * 0.92), "Green ghost arrows show the rotated expectation; red residuals should stay small.", small_font, fill=(196, 205, 214, 245))
        elif scene == "model_comparison_cinematic":
            yaw = -0.34 + 0.68 * phase
            progress = (phase - 0.12) / 0.32
            residual_progress = (phase - 0.58) / 0.24
            panels = [
                (0.19, "target_forces", (45, 232, 122, 245), "Ground truth", None),
                (0.50, "se3_forces", (28, 203, 255, 245), "SE3", "se3_forces"),
                (0.81, "painn_forces", (184, 104, 255, 245), "PaiNN-lite", "painn_forces"),
            ]
            for xfrac, force_key, color, label, residual_key in panels:
                residual = None
                if residual_key is not None:
                    residual = [vector_sub(pred, target) for pred, target in zip(sample[residual_key], sample["target_forces"])]
                draw_panel(
                    image,
                    center=(width * xfrac, height * 0.55),
                    yaw=yaw,
                    positions_key="positions",
                    force_key=force_key,
                    force_color=color,
                    label=label,
                    label_color=color,
                    target_radius=1.86,
                    panel_scale=min_dim * 0.096,
                    force_progress=progress,
                    max_arrow_length=0.78,
                    residual_forces=residual,
                    residual_progress=residual_progress,
                )
            text(draw, (width * 0.5, height * 0.135), "Same molecular frame, three force fields; red arrows visualize prediction residuals.", body_font, fill=(206, 222, 235, 255))
            if sample.get("metadata", {}).get("placeholder_note"):
                text(draw, (width * 0.5, height * 0.92), "Placeholder predictions are qualitative diagnostics, not quantitative model evidence.", small_font, fill=(230, 214, 170, 255))
        else:
            yaw = -0.68 + 2.05 * phase
            draw_panel(
                image,
                center=(width * 0.5, height * 0.56),
                yaw=yaw,
                positions_key="positions",
                force_key="target_forces",
                force_color=(45, 232, 122, 245),
                label="",
                label_color=(160, 255, 195, 255),
                target_radius=2.30,
                panel_scale=min_dim * 0.155,
                force_progress=(phase - 0.16) / 0.34,
                max_arrow_length=1.0,
            )
            text(draw, (width * 0.5, height * 0.135), "Atom identities stay fixed while force vectors live in 3D space.", body_font, fill=(206, 222, 235, 255))
        image.convert("RGB").save(output_path / f"frame_{frame:04d}.png", quality=96)
    return output_path


def _load_pillow_font(ImageFont, size: int, *, bold: bool):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _make_hd_background(Image, ImageDraw, width: int, height: int):
    image = Image.new("RGBA", (width, height), (13, 16, 22, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(9 + 10 * (1.0 - t))
        g = int(12 + 12 * (1.0 - t))
        b = int(18 + 20 * (1.0 - t))
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))
    draw.rectangle((0, 0, width, int(height * 0.17)), fill=(4, 7, 12, 90))
    draw.rectangle((0, int(height * 0.86), width, height), fill=(4, 7, 12, 76))
    return image


def _scene_title(scene: str) -> str:
    titles = {
        "force_scene_cinematic": "SE(3) molecular force field",
        "equivariance_cinematic": "SE(3) equivariance check",
        "model_comparison_cinematic": "Force prediction comparison",
    }
    return titles.get(scene, scene.replace("_", " "))


def smoothstep(value: float) -> float:
    x = min(1.0, max(0.0, float(value)))
    return x * x * (3.0 - 2.0 * x)
