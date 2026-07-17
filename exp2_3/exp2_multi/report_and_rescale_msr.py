#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


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


def write_ascii_ply(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.9f} {p[1]:.9f} {p[2]:.9f}\n")


def write_scaled_obj(src_obj: Path, dst_obj: Path, factor: float) -> None:
    dst_obj.parent.mkdir(parents=True, exist_ok=True)
    with src_obj.open("r", encoding="utf-8", errors="ignore") as fin, dst_obj.open(
        "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in fin:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    x = float(parts[1]) * factor
                    y = float(parts[2]) * factor
                    z = float(parts[3]) * factor
                    tail = ""
                    if len(parts) > 4:
                        tail = " " + " ".join(parts[4:])
                    fout.write(f"v {x:.9f} {y:.9f} {z:.9f}{tail}\n")
                    continue
            fout.write(line)


def parse_objects_csv(value: str) -> set[str]:
    if not value.strip():
        return set()
    return {x.strip() for x in value.split(",") if x.strip()}


def object_name_from_json_name(name: str) -> str:
    return name.split("__seed", 1)[0]


def compute_dims(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    ext = pmax - pmin
    return pmin, pmax, ext


def maybe_scale_distance_fields(data: dict, factor: float) -> None:
    if "fitness_details" in data and isinstance(data["fitness_details"], dict):
        fd = data["fitness_details"]
        if "distance_fit" in fd:
            fd["distance_fit"] = float(fd["distance_fit"]) * factor
        if "fitness" in fd:
            obj_cfg = data.get("objective_config", {})
            lambda_cov = float(obj_cfg.get("lambda_cov", 0.0))
            lambda_out = float(obj_cfg.get("lambda_out", 0.0))
            lambda_comp = float(obj_cfg.get("lambda_comp", 0.0))
            cov = float(fd.get("coverage_ratio", 0.0))
            out = float(fd.get("residual_unexplained_ratio", 0.0))
            comp = float(fd.get("num_superquadrics", 0.0))
            fd["fitness"] = (
                float(fd.get("distance_fit", 0.0))
                + lambda_cov * (1.0 - cov)
                + lambda_out * out
                + lambda_comp * comp
            )

    if "summary" in data and isinstance(data["summary"], dict) and "distance_fit" in data["summary"]:
        data["summary"]["distance_fit"] = float(data["summary"]["distance_fit"]) * factor

    if "layer_best_distance_fit" in data and isinstance(data["layer_best_distance_fit"], list):
        data["layer_best_distance_fit"] = [
            (float(x) * factor) if isinstance(x, (int, float)) else x for x in data["layer_best_distance_fit"]
        ]

    if "layer_best_fitness" in data and isinstance(data["layer_best_fitness"], list):
        cfg = data.get("objective_config", {})
        lambda_cov = float(cfg.get("lambda_cov", 0.0))
        lambda_out = float(cfg.get("lambda_out", 0.0))
        lambda_comp = float(cfg.get("lambda_comp", 0.0))
        cov_list = data.get("layer_best_coverage_ratio", [])
        out_list = data.get("layer_best_residual_unexplained_ratio", [])
        comp_list = data.get("layer_best_num_superquadrics", [])
        new_fit = []
        for i, old in enumerate(data["layer_best_fitness"]):
            if not isinstance(old, (int, float)):
                new_fit.append(old)
                continue
            dist = data.get("layer_best_distance_fit", [])[i] if i < len(data.get("layer_best_distance_fit", [])) else old
            cov = cov_list[i] if i < len(cov_list) else 0.0
            out = out_list[i] if i < len(out_list) else 0.0
            comp = comp_list[i] if i < len(comp_list) else 0.0
            new_fit.append(float(dist) + lambda_cov * (1.0 - float(cov)) + lambda_out * float(out) + lambda_comp * float(comp))
        data["layer_best_fitness"] = new_fit

    if "best_fitness" in data and isinstance(data["best_fitness"], (int, float)):
        fd = data.get("fitness_details", {})
        if isinstance(fd, dict):
            data["best_fitness"] = float(fd.get("fitness", data["best_fitness"]))


def scale_result_json(data: dict, factor: float, new_input_path: Path | None) -> dict:
    scaled = json.loads(json.dumps(data))

    for sq in scaled.get("superquadrics", []):
        sq["scale"] = [float(x) * factor for x in sq.get("scale", [])]
        sq["translation"] = [float(x) * factor for x in sq.get("translation", [])]

    norm = scaled.get("normalization")
    if isinstance(norm, dict):
        meta = norm.get("meta")
        if isinstance(meta, dict):
            if "center" in meta:
                meta["center"] = [float(x) * factor for x in meta["center"]]
            if "scale" in meta:
                meta["scale"] = float(meta["scale"]) * factor

    if new_input_path is not None:
        scaled["input_path"] = str(new_input_path.resolve())

    scaled["physical_rescale"] = {
        "applied": True,
        "factor": float(factor),
        "source_input_path": data.get("input_path", ""),
    }
    maybe_scale_distance_fields(scaled, factor)
    return scaled


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report physical sizes of E2 MSR inputs and optionally rescale both point clouds and MSR JSON."
    )
    parser.add_argument("--results-root", type=str, required=True, help="e.g. exp2_3/results/e2_multi_8x4_f005_c001")
    parser.add_argument("--method", type=str, default="pso", choices=["grid", "pso", "cmaes"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objects", type=str, default="", help="Optional comma-separated object basenames without _25k.")
    parser.add_argument("--report-csv", type=str, default="", help="Optional report CSV path.")
    parser.add_argument("--target-max-dim", type=float, default=0.0, help="If >0, scale each object so its max AABB extent equals this value.")
    parser.add_argument("--output-root", type=str, default="", help="Required when --target-max-dim > 0. Writes scaled ply/json here.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_root = (repo_root / args.results_root).resolve()
    json_dir = results_root / "json" / args.method
    obj_root = repo_root / "data" / "KIT_ObjectModels_25k_obj"
    selected = parse_objects_csv(args.objects)

    json_paths = sorted(json_dir.glob(f"*__seed{args.seed}.json"))
    if selected:
        json_paths = [p for p in json_paths if object_name_from_json_name(p.stem).replace("_25k", "") in selected]

    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in {json_dir} for seed={args.seed}.")

    do_rescale = float(args.target_max_dim) > 0.0
    if do_rescale and not args.output_root:
        raise ValueError("--output-root is required when --target-max-dim > 0.")

    output_root = (repo_root / args.output_root).resolve() if args.output_root else None
    rows: list[dict] = []

    for jp in json_paths:
        data = json.loads(jp.read_text(encoding="utf-8"))
        obj = object_name_from_json_name(jp.stem)
        input_path = Path(str(data.get("input_path", ""))).expanduser().resolve()
        points = read_ply_xyz(input_path)
        _, _, ext = compute_dims(points)
        max_dim = float(np.max(ext))
        diag = float(np.linalg.norm(ext))
        norm = data.get("normalization", {})
        fixed = data.get("fixed_hyperparameters", {})
        factor = (float(args.target_max_dim) / max_dim) if do_rescale and max_dim > 0 else 1.0

        row = {
            "object": obj.replace("_25k", ""),
            "json_name": jp.name,
            "input_path": str(input_path),
            "dx": float(ext[0]),
            "dy": float(ext[1]),
            "dz": float(ext[2]),
            "max_dim": max_dim,
            "diag": diag,
            "output_in_original_scale": bool(norm.get("output_in_original_scale", False)),
            "rescale_flag": bool(fixed.get("rescale", False)),
            "norm_method": str(norm.get("method", "")),
            "norm_center_x": float(norm.get("meta", {}).get("center", [np.nan, np.nan, np.nan])[0]),
            "norm_center_y": float(norm.get("meta", {}).get("center", [np.nan, np.nan, np.nan])[1]),
            "norm_center_z": float(norm.get("meta", {}).get("center", [np.nan, np.nan, np.nan])[2]),
            "norm_scale": float(norm.get("meta", {}).get("scale", np.nan)),
            "target_max_dim": float(args.target_max_dim) if do_rescale else "",
            "scale_factor": factor if do_rescale else "",
            "scaled_dx": float(ext[0]) * factor if do_rescale else "",
            "scaled_dy": float(ext[1]) * factor if do_rescale else "",
            "scaled_dz": float(ext[2]) * factor if do_rescale else "",
        }
        rows.append(row)

        if do_rescale:
            scaled_points = np.asarray(points, dtype=float) * factor
            scaled_ply = output_root / "ply" / f"{obj}.ply"
            src_obj = obj_root / f"{obj}.obj"
            scaled_obj = output_root / "obj" / f"{obj}.obj"
            scaled_json = output_root / "json" / args.method / jp.name
            write_ascii_ply(scaled_ply, scaled_points)
            if src_obj.exists():
                write_scaled_obj(src_obj, scaled_obj, factor=factor)
                src_mtl = src_obj.with_suffix(".mtl")
                if src_mtl.exists():
                    scaled_obj.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_mtl, scaled_obj.with_suffix(".mtl"))
            scaled_data = scale_result_json(data, factor=factor, new_input_path=scaled_ply)
            scaled_json.parent.mkdir(parents=True, exist_ok=True)
            scaled_json.write_text(json.dumps(scaled_data, indent=2), encoding="utf-8")

    report_path = (
        (repo_root / args.report_csv).resolve()
        if args.report_csv
        else (results_root / f"scale_report_{args.method}_seed{args.seed}.csv").resolve()
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved report to: {report_path}")
    print(f"Objects: {len(rows)}")
    if do_rescale:
        print(f"Saved scaled outputs to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
