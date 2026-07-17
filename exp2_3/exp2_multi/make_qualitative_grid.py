#!/usr/bin/env python3
import argparse
import csv
import io
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "msr_matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_ply_xyz(path: Path) -> np.ndarray:
    try:
        from plyfile import PlyData  # type: ignore

        ply = PlyData.read(str(path))
        vertex = ply["vertex"]
        return np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(float)
    except Exception:
        pass

    try:
        import open3d as o3d  # type: ignore

        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points, dtype=float)
        if points.ndim == 2 and points.shape[1] == 3 and points.shape[0] > 0:
            return points
    except Exception:
        pass

    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY file: unexpected EOF before end_header: {path}")
            line_str = line.decode("utf-8", errors="ignore").strip()
            header_lines.append(line_str)
            if line_str == "end_header":
                break

        format_line = next((ln for ln in header_lines if ln.startswith("format ")), "")
        vertex_line = next((ln for ln in header_lines if ln.startswith("element vertex ")), "")
        if not vertex_line:
            raise ValueError(f"Invalid PLY file: missing vertex element: {path}")
        vertex_count = int(vertex_line.split()[-1])

        property_lines = [ln for ln in header_lines if ln.startswith("property ")]
        property_specs = []
        for ln in property_lines:
            parts = ln.split()
            if len(parts) != 3:
                continue
            _, ptype, pname = parts
            property_specs.append((ptype, pname))

        if "ascii" in format_line:
            points = np.loadtxt(f, dtype=float, usecols=(0, 1, 2), max_rows=vertex_count)
            if points.ndim == 1:
                points = points.reshape(1, 3)
            return points

        if "binary_little_endian" in format_line:
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
            dtype_fields = []
            for ptype, pname in property_specs:
                if ptype not in type_map:
                    raise RuntimeError(f"Unsupported binary PLY property type '{ptype}' in {path}")
                dtype_fields.append((pname, type_map[ptype]))
            data = np.fromfile(f, dtype=np.dtype(dtype_fields), count=vertex_count)
            return np.column_stack((data["x"], data["y"], data["z"])).astype(float)

        raise RuntimeError(f"Unsupported PLY format in {path}: {format_line}")


def load_ordered_objects(labels_csv: Path, objects_available: set[str]) -> list[str]:
    ordered: list[str] = []
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            stem = Path(row["path"]).stem
            if row.get("label", "").strip().lower() != "multi":
                continue
            if stem in objects_available:
                ordered.append(stem)
    return ordered


def discover_method_images(vis_dir: Path, method: str, seed: int) -> dict[str, Path]:
    method_dir = vis_dir / method
    pattern = f"*__seed{seed}.png"
    return {
        img.name.split("__seed", 1)[0]: img
        for img in sorted(method_dir.glob(pattern))
        if img.is_file()
    }


def safe_float(value, default=float("inf")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_seed_spec(value: str) -> tuple[str, int | None]:
    value = value.strip().lower()
    if value == "best":
        return "best", None
    if value in {"best-available", "best_available"}:
        return "best_available", None
    return "fixed", int(value)


def parse_seed_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def object_name_from_seeded_name(name: str) -> str:
    return name.split("__seed", 1)[0]


def seed_from_seeded_name(name: str) -> int:
    return int(name.split("__seed", 1)[1].split(".", 1)[0])


def choose_best_json(
    json_dir: Path,
    obj: str,
    candidate_seeds: list[int],
) -> tuple[Path, int]:
    candidates: list[tuple[float, int, Path]] = []
    for js in sorted(json_dir.glob(f"{obj}__seed*.json")):
        seed = seed_from_seeded_name(js.stem)
        if candidate_seeds and seed not in candidate_seeds:
            continue
        data = json.loads(js.read_text(encoding="utf-8"))
        fit = safe_float(data.get("best_fitness"), float("inf"))
        candidates.append((fit, seed, js))
    if not candidates:
        raise FileNotFoundError(f"No candidate JSON found for {obj} in {json_dir}")
    candidates.sort(key=lambda x: (x[0], x[1], str(x[2])))
    _, seed, js = candidates[0]
    return js, seed


def choose_best_available_image(
    json_dir: Path,
    vis_dir: Path,
    obj: str,
    candidate_seeds: list[int],
) -> tuple[Path, int]:
    candidates: list[tuple[float, int, Path]] = []
    for js in sorted(json_dir.glob(f"{obj}__seed*.json")):
        seed = seed_from_seeded_name(js.stem)
        if candidate_seeds and seed not in candidate_seeds:
            continue
        img_path = vis_dir / f"{obj}__seed{seed}.png"
        if not img_path.exists():
            continue
        data = json.loads(js.read_text(encoding="utf-8"))
        fit = safe_float(data.get("best_fitness"), float("inf"))
        candidates.append((fit, seed, img_path))
    if not candidates:
        raise FileNotFoundError(f"No candidate PNG found for {obj} in {vis_dir}")
    candidates.sort(key=lambda x: (x[0], x[1], str(x[2])))
    _, seed, img_path = candidates[0]
    return img_path, seed


def ensure_rendered_image(
    json_path: Path,
    image_path: Path,
    visualize_mode: str,
    timeout_sec: int,
) -> Path:
    if image_path.exists():
        return image_path

    import render_images_from_json as rij

    image_path.parent.mkdir(parents=True, exist_ok=True)
    success, err = rij._render_one_isolated(
        json_path=json_path,
        image_path=image_path,
        visualize_mode=visualize_mode,
        arc_length=0.2,
        point_size=0.001,
        offscreen=False,
        timeout_sec=timeout_sec,
    )
    if not success or not image_path.exists():
        raise RuntimeError(f"Failed to render missing image for {json_path.name}: {err}")
    return image_path


def build_selected_method_images(
    results_dir: Path,
    method: str,
    seed_mode: str,
    fixed_seed: int | None,
    candidate_seeds: list[int],
    render_missing_best: bool,
    render_visualize_mode: str,
    render_timeout_sec: int,
) -> dict[str, Path]:
    vis_dir = results_dir / "vis" / method
    json_dir = results_dir / "json" / method

    if seed_mode == "fixed":
        if fixed_seed is None:
            raise ValueError("fixed_seed is required when seed_mode='fixed'")
        return discover_method_images(results_dir / "vis", method, fixed_seed)

    selected: dict[str, Path] = {}
    objects_in_json = sorted({object_name_from_seeded_name(js.stem) for js in json_dir.glob("*.json")})

    if seed_mode == "best_available":
        for obj in objects_in_json:
            try:
                img_path, _ = choose_best_available_image(json_dir, vis_dir, obj, candidate_seeds)
            except FileNotFoundError:
                continue
            selected[obj] = img_path
        return selected

    for obj in objects_in_json:
        best_json, best_seed = choose_best_json(json_dir, obj, candidate_seeds)
        img_path = vis_dir / f"{obj}__seed{best_seed}.png"
        if img_path.exists():
            selected[obj] = img_path
            continue
        if not render_missing_best:
            continue
        selected[obj] = ensure_rendered_image(
            json_path=best_json,
            image_path=img_path,
            visualize_mode=render_visualize_mode,
            timeout_sec=render_timeout_sec,
        )
    return selected


def render_point_cloud_image(
    ply_path: Path,
    cell_px: int,
    max_points: int,
    rng: np.random.Generator,
    elev: float,
    azim: float,
) -> Image.Image:
    points = read_ply_xyz(ply_path)
    if points.shape[0] > max_points:
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]

    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0, keepdims=True)
    points = points - center
    scale = np.max(np.ptp(points, axis=0))
    if scale > 0:
        points = points / scale

    fig = plt.figure(figsize=(cell_px / 100.0, cell_px / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.25, c="#1f77b4", alpha=0.9)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    lim = 0.55
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def fit_image_to_cell(image: Image.Image, cell_px: int, inner_pad: int) -> Image.Image:
    target = Image.new("RGB", (cell_px, cell_px), "white")
    max_w = cell_px - 2 * inner_pad
    max_h = cell_px - 2 * inner_pad
    src = image.convert("RGB")
    src.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    off_x = (cell_px - src.width) // 2
    off_y = (cell_px - src.height) // 2
    target.paste(src, (off_x, off_y))
    return target


def draw_text_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = left + (right - left - text_w) // 2
    y = top + (bottom - top - text_h) // 2
    draw.text((x, y), text, fill="black", font=font)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compose a qualitative 39x4 comparison figure from one exp2 results directory."
    )
    parser.add_argument("results_dir", type=str, help="Path like exp2_3/results/e2_multi_5x4_f005_c001")
    parser.add_argument("--output", type=str, default="", help="Optional output PNG path.")
    parser.add_argument("--ply-root", type=str, default="data/KIT_ObjectModels_25k_ply")
    parser.add_argument("--labels", type=str, default="data/kit_superquadric_labels.csv")
    parser.add_argument(
        "--seed",
        type=str,
        default="0",
        help="Use a fixed seed index such as 0, 'best' for the true lowest-fitness seed, or 'best-available' for the lowest-fitness seed that already has a PNG.",
    )
    parser.add_argument(
        "--candidate-seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated candidate seeds used when --seed best.",
    )
    parser.add_argument(
        "--render-missing-best",
        action="store_true",
        help="When --seed best and the chosen PNG is missing, render it from the corresponding JSON.",
    )
    parser.add_argument(
        "--render-visualize-mode",
        type=str,
        default="with_points",
        choices=["reconstruction_only", "with_points"],
        help="Visualization mode used only when rendering missing best-seed images from JSON.",
    )
    parser.add_argument(
        "--render-timeout-sec",
        type=int,
        default=120,
        help="Timeout per missing-image render when --render-missing-best is enabled.",
    )
    parser.add_argument("--cell-px", type=int, default=320, help="Square cell size in pixels.")
    parser.add_argument("--label-width", type=int, default=180, help="Left label column width in pixels.")
    parser.add_argument("--header-height", type=int, default=70, help="Header row height in pixels.")
    parser.add_argument("--gap", type=int, default=8, help="Gap between cells in pixels.")
    parser.add_argument("--inner-pad", type=int, default=6, help="Padding inside each image cell in pixels.")
    parser.add_argument("--max-points", type=int, default=10000, help="Max rendered points for original cloud.")
    parser.add_argument("--elev", type=float, default=22.0, help="3D view elevation for original cloud.")
    parser.add_argument("--azim", type=float, default=42.0, help="3D view azimuth for original cloud.")
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument("--row-font-size", type=int, default=18)
    parser.add_argument("--limit-rows", type=int, default=0, help="Debug helper. 0 means all rows.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = (repo_root / args.results_dir).resolve()
    vis_dir = results_dir / "vis"
    ply_root = (repo_root / args.ply_root).resolve()
    labels_csv = (repo_root / args.labels).resolve()

    seed_mode, fixed_seed = parse_seed_spec(args.seed)
    candidate_seeds = parse_seed_list(args.candidate_seeds)

    grid_map = build_selected_method_images(
        results_dir=results_dir,
        method="grid",
        seed_mode=seed_mode,
        fixed_seed=fixed_seed,
        candidate_seeds=candidate_seeds,
        render_missing_best=args.render_missing_best,
        render_visualize_mode=args.render_visualize_mode,
        render_timeout_sec=args.render_timeout_sec,
    )
    pso_map = build_selected_method_images(
        results_dir=results_dir,
        method="pso",
        seed_mode=seed_mode,
        fixed_seed=fixed_seed,
        candidate_seeds=candidate_seeds,
        render_missing_best=args.render_missing_best,
        render_visualize_mode=args.render_visualize_mode,
        render_timeout_sec=args.render_timeout_sec,
    )
    cmaes_map = build_selected_method_images(
        results_dir=results_dir,
        method="cmaes",
        seed_mode=seed_mode,
        fixed_seed=fixed_seed,
        candidate_seeds=candidate_seeds,
        render_missing_best=args.render_missing_best,
        render_visualize_mode=args.render_visualize_mode,
        render_timeout_sec=args.render_timeout_sec,
    )
    common_objects = set(grid_map) & set(pso_map) & set(cmaes_map)
    if not common_objects:
        raise FileNotFoundError(f"No common PNG triplets found under {vis_dir}")

    ordered_objects = load_ordered_objects(labels_csv, common_objects)
    if args.limit_rows > 0:
        ordered_objects = ordered_objects[: args.limit_rows]
    if not ordered_objects:
        raise RuntimeError("No ordered multi-object rows matched the available visualization triplets.")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (results_dir / "figures" / f"qualitative_grid_seed{args.seed}.png").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas_w = args.gap + args.label_width + args.gap + 4 * (args.cell_px + args.gap)
    canvas_h = args.gap + args.header_height + args.gap + len(ordered_objects) * (args.cell_px + args.gap)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        header_font = ImageFont.truetype("Arial.ttf", args.font_size)
    except Exception:
        header_font = ImageFont.load_default()
    try:
        row_font = ImageFont.truetype("Arial.ttf", args.row_font_size)
    except Exception:
        row_font = ImageFont.load_default()

    x0 = args.gap + args.label_width + args.gap
    headers = ["Original Point Cloud", "Grid", "PSO", "CMA-ES"]
    for idx, text in enumerate(headers):
        left = x0 + idx * (args.cell_px + args.gap)
        right = left + args.cell_px
        draw_text_center(draw, (left, args.gap, right, args.gap + args.header_height), text, header_font)

    rng = np.random.default_rng(42)
    for row_idx, obj in enumerate(ordered_objects):
        top = args.gap + args.header_height + args.gap + row_idx * (args.cell_px + args.gap)
        bottom = top + args.cell_px

        label_text = obj.replace("_25k", "")
        draw_text_center(draw, (args.gap, top, args.gap + args.label_width, bottom), label_text, row_font)

        ply_path = ply_root / f"{obj}.ply"
        original_img = render_point_cloud_image(
            ply_path=ply_path,
            cell_px=args.cell_px,
            max_points=args.max_points,
            rng=rng,
            elev=args.elev,
            azim=args.azim,
        )
        method_imgs = [
            original_img,
            Image.open(grid_map[obj]),
            Image.open(pso_map[obj]),
            Image.open(cmaes_map[obj]),
        ]

        for col_idx, img in enumerate(method_imgs):
            left = x0 + col_idx * (args.cell_px + args.gap)
            cell_img = fit_image_to_cell(img, args.cell_px, args.inner_pad)
            canvas.paste(cell_img, (left, top))

    canvas.save(output_path)
    print(f"Saved qualitative grid to: {output_path}")
    print(f"Rows: {len(ordered_objects)}")
    print(f"Columns: 4")
    print(f"Seed mode: {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
