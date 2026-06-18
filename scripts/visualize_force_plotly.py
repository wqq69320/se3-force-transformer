#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from se3force.visualization.geometry import atom_color, infer_bonds, scale_forces, scale_positions_for_rendering

FORCE_KEYS = {"target_forces", "se3_forces", "painn_forces"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an interactive 3D force-field HTML artifact.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--force-key", required=True, choices=sorted(FORCE_KEYS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--show-rotated", action="store_true")
    parser.add_argument("--show-residual", action="store_true")
    return parser


def load_sample(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def panel_data(sample: dict, force_key: str, *, rotated: bool = False, x_offset: float = 0.0) -> dict:
    pos_key = "rotated_positions" if rotated else "positions"
    if rotated:
        force_key = "rotated_" + force_key
    positions = np.asarray(sample[pos_key], dtype=float)
    forces = np.asarray(sample[force_key], dtype=float)
    centered, position_scale = scale_positions_for_rendering(positions)
    scaled_forces, force_scale = scale_forces(forces)
    centered[:, 0] += x_offset
    return {
        "positions": centered,
        "forces": scaled_forces * position_scale,
        "raw_forces": forces,
        "bonds": infer_bonds(positions, sample["atomic_numbers"]),
        "position_scale": position_scale,
        "force_scale": force_scale,
        "title": "rotated" if rotated else "original",
    }


def traces_for_panel(sample: dict, panel: dict, force_key: str, *, show_residual: bool = False) -> list[dict]:
    positions = panel["positions"]
    forces = panel["forces"]
    atomic_numbers = np.asarray(sample["atomic_numbers"], dtype=int)
    colors = [f"rgb({int(255*r)},{int(255*g)},{int(255*b)})" for r, g, b in (atom_color(int(z)) for z in atomic_numbers)]
    hover = [
        f"atom {i}<br>Z={int(z)}<br>|F|={np.linalg.norm(panel['raw_forces'][i]):.4g}"
        for i, z in enumerate(atomic_numbers)
    ]
    traces: list[dict] = [
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": positions[:, 0].tolist(),
            "y": positions[:, 1].tolist(),
            "z": positions[:, 2].tolist(),
            "text": hover,
            "hoverinfo": "text",
            "marker": {"size": 7, "color": colors, "line": {"color": "white", "width": 0.5}},
            "name": f"atoms {panel['title']}",
        }
    ]
    for i, j in panel["bonds"]:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "x": [positions[i, 0], positions[j, 0]],
                "y": [positions[i, 1], positions[j, 1]],
                "z": [positions[i, 2], positions[j, 2]],
                "line": {"color": "rgba(210,210,210,0.62)", "width": 5},
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    line_z: list[float | None] = []
    for start, vec in zip(positions, forces):
        end = start + vec
        line_x.extend([start[0], end[0], None])
        line_y.extend([start[1], end[1], None])
        line_z.extend([start[2], end[2], None])
    traces.append(
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": line_x,
            "y": line_y,
            "z": line_z,
            "line": {"color": force_color(force_key), "width": 5},
            "name": f"{force_key} arrows {panel['title']}",
        }
    )
    tips = positions + forces
    traces.append(
        {
            "type": "cone",
            "x": tips[:, 0].tolist(),
            "y": tips[:, 1].tolist(),
            "z": tips[:, 2].tolist(),
            "u": forces[:, 0].tolist(),
            "v": forces[:, 1].tolist(),
            "w": forces[:, 2].tolist(),
            "anchor": "tip",
            "sizemode": "absolute",
            "sizeref": 0.22,
            "colorscale": [[0, force_color(force_key)], [1, force_color(force_key)]],
            "showscale": False,
            "name": f"{force_key} cone {panel['title']}",
        }
    )
    if show_residual and force_key != "target_forces":
        residual = np.asarray(sample[force_key], dtype=float) - np.asarray(sample["target_forces"], dtype=float)
        residual_scaled, _ = scale_forces(residual, max_arrow_length=0.7)
        residual_scaled *= panel["position_scale"]
        rx: list[float | None] = []
        ry: list[float | None] = []
        rz: list[float | None] = []
        for start, vec in zip(positions, residual_scaled):
            end = start + vec
            rx.extend([start[0], end[0], None])
            ry.extend([start[1], end[1], None])
            rz.extend([start[2], end[2], None])
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "x": rx,
                "y": ry,
                "z": rz,
                "line": {"color": "red", "width": 3},
                "name": "residual to target",
            }
        )
    return traces


def force_color(force_key: str) -> str:
    return {
        "target_forces": "rgb(80, 220, 120)",
        "se3_forces": "rgb(65, 210, 235)",
        "painn_forces": "rgb(170, 90, 255)",
    }.get(force_key, "rgb(80, 220, 120)")


def build_figure_dict(sample: dict, force_key: str, *, show_rotated: bool = False, show_residual: bool = False) -> dict:
    panels = [panel_data(sample, force_key, rotated=False, x_offset=-3.8 if show_rotated else 0.0)]
    if show_rotated:
        panels.append(panel_data(sample, force_key, rotated=True, x_offset=3.8))
    traces: list[dict] = []
    for panel in panels:
        traces.extend(traces_for_panel(sample, panel, force_key, show_residual=show_residual))
    title = f"{sample.get('molecule', 'molecule')} frame {sample.get('frame_index')} - {force_key}"
    if sample.get("metadata", {}).get("placeholder_note"):
        title += " (placeholder predictions marked in metadata)"
    return {
        "data": traces,
        "layout": {
            "title": title,
            "paper_bgcolor": "#101214",
            "plot_bgcolor": "#101214",
            "font": {"color": "#f2f2f2"},
            "scene": {
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False},
                "aspectmode": "data",
                "bgcolor": "#101214",
            },
            "margin": {"l": 0, "r": 0, "t": 48, "b": 0},
        },
    }


def write_html_with_plotly_package(fig_dict: dict, output: Path) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot
    except Exception:
        return False
    fig = go.Figure(data=fig_dict["data"], layout=fig_dict["layout"])
    plot(fig, filename=str(output), auto_open=False, include_plotlyjs="cdn")
    return True


def write_html_fallback(fig_dict: dict, output: Path) -> None:
    payload = json.dumps(fig_dict)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <title>SE3 Force Visualization</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>html, body, #plot {{ width: 100%; height: 100%; margin: 0; background: #101214; }}</style>
</head>
<body>
  <div id=\"plot\"></div>
  <script>
    const figure = {payload};
    Plotly.newPlot('plot', figure.data, figure.layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sample = load_sample(args.input)
    fig_dict = build_figure_dict(sample, args.force_key, show_rotated=args.show_rotated, show_residual=args.show_residual)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not write_html_with_plotly_package(fig_dict, output):
        write_html_fallback(fig_dict, output)
        print("plotly is not installed; wrote CDN-backed Plotly HTML fallback")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
