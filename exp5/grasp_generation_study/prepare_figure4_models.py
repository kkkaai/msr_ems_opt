#!/usr/bin/env python3
"""Prepare the eight Figure 4 CMA-ES reconstructions for grasp generation."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "msr_matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "exp2_3/results/e2_multi_5x4_f005_c001/json/cmaes"
OUTPUT_ROOT = REPO_ROOT / "exp5/grasp_generation_study/results/figure4_numbered_cmaes"
TARGET_MAX_DIM_M = 0.20
OBJECTS = [
    "HamburgerSauce",
    "Heart1",
    "Slotted_Screwdriver",
    "FizzyTabletsCalcium",
    "MelforBottle",
    "Deodorant",
    "CokePlasticLarge",
    "Moon",
]


def signed_power(value: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(value) * np.abs(value) ** exponent


def sample_superquadric(sq: dict, n_eta: int = 64, n_omega: int = 128) -> np.ndarray:
    e1, e2 = (float(x) for x in sq["shape"])
    a1, a2, a3 = (float(x) for x in sq["scale"])
    eta = np.linspace(-np.pi / 2, np.pi / 2, n_eta)
    omega = np.linspace(-np.pi, np.pi, n_omega)
    eta_grid, omega_grid = np.meshgrid(eta, omega, indexing="ij")

    ce = signed_power(np.cos(eta_grid), e1)
    se = signed_power(np.sin(eta_grid), e1)
    co = signed_power(np.cos(omega_grid), e2)
    so = signed_power(np.sin(omega_grid), e2)
    local = np.stack((a1 * ce * co, a2 * ce * so, a3 * se), axis=-1)

    rotation = Rotation.from_euler("ZYX", sq["euler"]).as_matrix()
    translation = np.asarray(sq["translation"], dtype=float)
    return local.reshape(-1, 3) @ rotation.T + translation


def read_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PLY header: {path}")
            text = line.decode("ascii", errors="ignore").strip()
            header.append(text)
            if text == "end_header":
                break

        fmt = next(x for x in header if x.startswith("format ")).split()[1]
        count = int(next(x for x in header if x.startswith("element vertex ")).split()[-1])
        properties = []
        in_vertex = False
        for line in header:
            if line.startswith("element vertex "):
                in_vertex = True
                continue
            if line.startswith("element ") and in_vertex:
                break
            if in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) == 3:
                    properties.append((parts[2], parts[1]))

        if fmt == "ascii":
            points = np.loadtxt(handle, max_rows=count, usecols=(0, 1, 2), dtype=float)
            return np.atleast_2d(points)

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format {fmt}: {path}")

        type_map = {
            "char": "<i1",
            "uchar": "<u1",
            "int8": "<i1",
            "uint8": "<u1",
            "short": "<i2",
            "ushort": "<u2",
            "int16": "<i2",
            "uint16": "<u2",
            "int": "<i4",
            "uint": "<u4",
            "int32": "<i4",
            "uint32": "<u4",
            "float": "<f4",
            "float32": "<f4",
            "double": "<f8",
            "float64": "<f8",
        }
        dtype = np.dtype([(name, type_map[kind]) for name, kind in properties])
        vertices = np.fromfile(handle, dtype=dtype, count=count)
        return np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(float)


def equalize_axes(ax, points: np.ndarray) -> None:
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    center = (pmin + pmax) / 2
    radius = max(float(np.max(pmax - pmin)) / 2, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def layer_ids(counts: list[int], total: int) -> list[int]:
    result: list[int] = []
    for layer, count in enumerate(counts):
        result.extend([layer] * max(0, int(count)))
    result.extend([len(counts)] * max(0, total - len(result)))
    return result[:total]


def transformed_json(data: dict, center_mm: np.ndarray, scale_m_per_mm: float) -> dict:
    result = json.loads(json.dumps(data))
    for index, sq in enumerate(result["superquadrics"], start=1):
        sq["id"] = index
        sq["source_scale_mm"] = list(sq["scale"])
        sq["source_translation_mm"] = list(sq["translation"])
        sq["scale"] = [float(x) * scale_m_per_mm for x in sq["scale"]]
        translation = np.asarray(sq["translation"], dtype=float)
        sq["translation"] = ((translation - center_mm) * scale_m_per_mm).tolist()
    result["grasp_model_transform"] = {
        "source_unit": "mm",
        "target_unit": "m",
        "aabb_center_mm": center_mm.tolist(),
        "uniform_scale_m_per_mm": float(scale_m_per_mm),
        "target_max_dimension_m": TARGET_MAX_DIM_M,
        "formula": "x_grasp_m = uniform_scale_m_per_mm * (x_source_mm - aabb_center_mm)",
    }
    return result


def render_object(
    object_name: str,
    data: dict,
    surfaces: list[np.ndarray],
    output_path: Path,
) -> None:
    all_points = np.concatenate(surfaces, axis=0)
    layers = layer_ids(data.get("layer_best_num_superquadrics", []), len(surfaces))
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(20, len(surfaces))))
    camera_azim = -62.0
    camera_elev = 18.0
    azim_rad = np.deg2rad(camera_azim)
    elev_rad = np.deg2rad(camera_elev)
    camera_vector = np.array(
        [
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ]
    )
    scene_extent = float(np.max(np.ptp(all_points, axis=0)))

    fig = plt.figure(figsize=(8.2, 7.4), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    for index, (sq, points) in enumerate(zip(data["superquadrics"], surfaces), start=1):
        grid = points.reshape(64, 128, 3)
        color = colors[index - 1]
        ax.plot_surface(
            grid[:, :, 0],
            grid[:, :, 1],
            grid[:, :, 2],
            color=color,
            alpha=0.50,
            linewidth=0,
            antialiased=True,
            shade=True,
        )
        center = np.asarray(sq["translation"], dtype=float)
        front_point = points[np.argmax(points @ camera_vector)]
        label_position = front_point + camera_vector * (0.025 * scene_extent)
        ax.plot(
            [center[0], label_position[0]],
            [center[1], label_position[1]],
            [center[2], label_position[2]],
            color=color,
            linewidth=1.0,
            alpha=0.9,
        )
        ax.text(
            label_position[0],
            label_position[1],
            label_position[2],
            f"SQ{index}",
            fontsize=8,
            weight="bold",
            color="black",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": color, "alpha": 0.94},
        )

    equalize_axes(ax, all_points)
    ax.view_init(elev=camera_elev, azim=camera_azim)
    ax.set_title(f"{object_name}: numbered CMA-ES reconstruction", fontsize=13, pad=12)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    legend_text = "  ".join(f"SQ{i + 1}: L{layers[i]}" for i in range(len(surfaces)))
    fig.text(0.5, 0.025, legend_text, ha="center", va="bottom", fontsize=7, wrap=True)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_contact_sheet(paths: list[Path], output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    images = [Image.open(path).convert("RGB") for path in paths]
    thumb_w, thumb_h = 720, 650
    sheet = Image.new("RGB", (thumb_w * 2, thumb_h * 4), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (path, image) in enumerate(zip(paths, images)):
        image.thumbnail((thumb_w, thumb_h - 28))
        x = (index % 2) * thumb_w + (thumb_w - image.width) // 2
        y = (index // 2) * thumb_h + 24
        sheet.paste(image, (x, y))
        draw.text(((index % 2) * thumb_w + 10, (index // 2) * thumb_h + 6), path.stem, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    numbered_paths: list[Path] = []
    dimension_rows: list[dict] = []
    primitive_rows: list[dict] = []

    for object_name in OBJECTS:
        json_path = RESULTS_ROOT / f"{object_name}_25k__seed0.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        surfaces = [sample_superquadric(sq) for sq in data["superquadrics"]]
        reconstructed_points = np.concatenate(surfaces, axis=0)
        rec_min = reconstructed_points.min(axis=0)
        rec_max = reconstructed_points.max(axis=0)
        rec_extent = rec_max - rec_min
        center_mm = (rec_min + rec_max) / 2
        scale_m_per_mm = TARGET_MAX_DIM_M / float(rec_extent.max())

        input_points = read_ply_xyz(Path(data["input_path"]))
        input_extent = input_points.max(axis=0) - input_points.min(axis=0)

        image_path = OUTPUT_ROOT / "numbered" / f"{object_name}_numbered.png"
        render_object(object_name, data, surfaces, image_path)
        numbered_paths.append(image_path)

        scaled = transformed_json(data, center_mm, scale_m_per_mm)
        scaled_path = OUTPUT_ROOT / "scaled_json_20cm" / f"{object_name}_20cm.json"
        scaled_path.parent.mkdir(parents=True, exist_ok=True)
        scaled_path.write_text(json.dumps(scaled, indent=2), encoding="utf-8")

        dimension_rows.append(
            {
                "object": object_name,
                "num_superquadrics": len(surfaces),
                "input_x_mm": input_extent[0],
                "input_y_mm": input_extent[1],
                "input_z_mm": input_extent[2],
                "reconstruction_x_mm": rec_extent[0],
                "reconstruction_y_mm": rec_extent[1],
                "reconstruction_z_mm": rec_extent[2],
                "reconstruction_max_mm": rec_extent.max(),
                "uniform_scale_m_per_mm": scale_m_per_mm,
                "scaled_x_m": rec_extent[0] * scale_m_per_mm,
                "scaled_y_m": rec_extent[1] * scale_m_per_mm,
                "scaled_z_m": rec_extent[2] * scale_m_per_mm,
            }
        )

        layers = layer_ids(data.get("layer_best_num_superquadrics", []), len(surfaces))
        for index, sq in enumerate(data["superquadrics"], start=1):
            scaled_sq = scaled["superquadrics"][index - 1]
            primitive_rows.append(
                {
                    "object": object_name,
                    "sq_id": f"SQ{index}",
                    "layer": layers[index - 1],
                    "shape_epsilon_1": sq["shape"][0],
                    "shape_epsilon_2": sq["shape"][1],
                    "scale_a1_mm": sq["scale"][0],
                    "scale_a2_mm": sq["scale"][1],
                    "scale_a3_mm": sq["scale"][2],
                    "center_x_mm": sq["translation"][0],
                    "center_y_mm": sq["translation"][1],
                    "center_z_mm": sq["translation"][2],
                    "scaled_a1_m": scaled_sq["scale"][0],
                    "scaled_a2_m": scaled_sq["scale"][1],
                    "scaled_a3_m": scaled_sq["scale"][2],
                    "scaled_center_x_m": scaled_sq["translation"][0],
                    "scaled_center_y_m": scaled_sq["translation"][1],
                    "scaled_center_z_m": scaled_sq["translation"][2],
                }
            )

    for filename, rows in (
        ("model_dimensions_and_scaling.csv", dimension_rows),
        ("primitive_id_map.csv", primitive_rows),
    ):
        path = OUTPUT_ROOT / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    make_contact_sheet(numbered_paths, OUTPUT_ROOT / "figure4_eight_objects_numbered.png")
    print(f"Wrote outputs to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
