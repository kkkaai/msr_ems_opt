#!/usr/bin/env python3
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

HEX_LAYER_COLORS = ["00FF80", "00FFFF", "007FFF", "0000FF", "7F00FF"]
RGB_LAYER_COLORS = [tuple(int(c[i:i+2], 16)/255.0 for i in (0, 2, 4)) for c in HEX_LAYER_COLORS]


def spow(x: np.ndarray, e: float) -> np.ndarray:
    return np.sign(x) * (np.abs(x) ** e)


def layer_color(layer_idx: int) -> tuple[float, float, float]:
    return RGB_LAYER_COLORS[layer_idx % len(RGB_LAYER_COLORS)]


def set_axes_equal(ax, points: np.ndarray | None = None):
    if points is None or points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius <= 1e-12:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def read_ply_xyz(path: Path) -> np.ndarray:
    try:
        from plyfile import PlyData
        ply = PlyData.read(str(path))
        v = ply["vertex"]
        return np.column_stack((v["x"], v["y"], v["z"]))
    except Exception:
        pass

    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY (EOF before end_header): {path}")
            s = line.decode("utf-8", errors="ignore").strip()
            header_lines.append(s)
            if s == "end_header":
                break

        fmt = next((ln for ln in header_lines if ln.startswith("format ")), "")
        if "ascii" not in fmt:
            raise RuntimeError("Binary PLY requires plyfile package")
        vertex_line = next((ln for ln in header_lines if ln.startswith("element vertex ")), "")
        if not vertex_line:
            raise ValueError(f"No vertex element in PLY: {path}")
        n = int(vertex_line.split()[-1])
        pts = np.loadtxt(f, dtype=float, usecols=(0, 1, 2), max_rows=n)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        return pts


def build_layer_ids(summary: dict, num_quadrics: int) -> list[int]:
    counts = summary.get("point_inlier_count_per_layer", {})
    layer_ids = []
    for k in sorted((int(x) for x in counts.keys())):
        layer_ids.extend([k] * int(counts[str(k)]))
    if len(layer_ids) < num_quadrics:
        fallback = layer_ids[-1] if layer_ids else 0
        layer_ids.extend([fallback] * (num_quadrics - len(layer_ids)))
    return layer_ids[:num_quadrics]


def render_result(json_path: Path, png_path: Path, max_points: int = 5000, draw_points: bool = True) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    sqs = data.get("superquadrics", [])
    inp = Path(data["input_path"]).expanduser()
    pts = read_ply_xyz(inp)

    if pts.shape[0] > max_points:
        idx = np.random.default_rng(42).choice(pts.shape[0], size=max_points, replace=False)
        pts_plot = pts[idx]
    else:
        pts_plot = pts

    fig = plt.figure(figsize=(7, 7), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    if draw_points:
        ax.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2], s=0.2, c="#8a8a8a", alpha=0.25)

    layer_ids = build_layer_ids(data, len(sqs))

    eta = np.linspace(-math.pi / 2.0, math.pi / 2.0, 28)
    omega = np.linspace(-math.pi, math.pi, 56)
    ETA, OMEGA = np.meshgrid(eta, omega)

    for i, sq in enumerate(sqs):
        e1, e2 = [float(x) for x in sq["shape"]]
        a1, a2, a3 = [float(x) for x in sq["scale"]]
        euler = np.array(sq["euler"], dtype=float)
        t = np.array(sq["translation"], dtype=float)

        ce = spow(np.cos(ETA), e1)
        se = spow(np.sin(ETA), e1)
        co = spow(np.cos(OMEGA), e2)
        so = spow(np.sin(OMEGA), e2)

        X = a1 * ce * co
        Y = a2 * ce * so
        Z = a3 * se

        P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        rot = R.from_euler("ZYX", euler).as_matrix()
        P = P @ rot.T + t

        Xw = P[:, 0].reshape(X.shape)
        Yw = P[:, 1].reshape(Y.shape)
        Zw = P[:, 2].reshape(Z.shape)

        color = layer_color(layer_ids[i] if i < len(layer_ids) else 0)
        ax.plot_surface(Xw, Yw, Zw, color=color, linewidth=0, alpha=0.45, antialiased=False)

    ax.set_title(f"{json_path.stem} | SQ={len(sqs)} | fitness={data.get('best_fitness', float('nan')):.4f}", fontsize=9)
    ax.set_axis_off()
    set_axes_equal(ax, pts_plot)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch reconstruct KIT multi-label objects with run_hierarchical_ems_pso_3params.py and save screenshots."
    )
    parser.add_argument("--labels", type=str, default="data/kit_superquadric_labels.csv")
    parser.add_argument("--ply-root", type=str, default="data/KIT_ObjectModels_25k_ply")
    parser.add_argument("--output", type=str, default="results/kit_multi_pso3_20260320")
    parser.add_argument("--swarmsize", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--MaxLayer", type=int, default=5)
    parser.add_argument("--GlobalNormalize", type=str, default="True")
    parser.add_argument("--GlobalNormMethod", type=str, default="ems_matlab")
    parser.add_argument("--OutputInOriginalScale", type=str, default="True")
    parser.add_argument(
        "--FitnessMode",
        type=str,
        default="distance_coverage_outlier_complexity",
        choices=["legacy", "distance_coverage", "distance_coverage_outlier_complexity"],
        help="Objective mode forwarded to run_hierarchical_ems_pso_3params.py.",
    )
    parser.add_argument(
        "--LambdaCov",
        type=float,
        default=0.25,
        help="Coverage penalty weight for non-legacy objectives.",
    )
    parser.add_argument(
        "--LambdaOut",
        type=float,
        default=0.15,
        help="Residual unexplained ratio penalty weight for distance_coverage_outlier_complexity.",
    )
    parser.add_argument(
        "--LambdaComp",
        type=float,
        default=0.01,
        help="Complexity penalty weight for distance_coverage_outlier_complexity.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-points-render", type=int, default=5000)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    labels_path = (repo_root / args.labels).resolve()
    ply_root = (repo_root / args.ply_root).resolve()
    out_root = (repo_root / args.output).resolve()
    json_dir = out_root / "json"
    log_dir = out_root / "log"
    png_dir = out_root / "png"
    for d in [out_root, json_dir, log_dir, png_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    with labels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("label", "").strip().lower() == "multi":
                rows.append(r)

    runner = (repo_root / "src" / "run_hierarchical_ems_pso_3params.py").resolve()
    summary = []

    for i, r in enumerate(rows, start=1):
        rel = r["path"].strip()
        ply_path = ply_root / rel
        stem = Path(rel).stem
        json_out = json_dir / f"{stem}.json"
        log_out = log_dir / f"{stem}.log"
        png_out = png_dir / f"{stem}.png"

        if args.skip_existing and json_out.exists() and png_out.exists():
            summary.append({"model": stem, "status": "SKIP", "reason": "existing", "json": str(json_out), "png": str(png_out)})
            print(f"[{i:02d}/{len(rows):02d}] SKIP {stem}")
            continue

        if not ply_path.exists():
            summary.append({"model": stem, "status": "FAIL", "reason": f"missing_ply:{ply_path}"})
            print(f"[{i:02d}/{len(rows):02d}] FAIL {stem} (missing ply)")
            continue

        cmd = [
            sys.executable,
            str(runner),
            str(ply_path),
            "--out", str(json_out),
            "--runtime",
            "--swarmsize", str(args.swarmsize),
            "--maxiter", str(args.maxiter),
            "--MaxLayer", str(args.MaxLayer),
            "--GlobalNormalize", str(args.GlobalNormalize),
            "--GlobalNormMethod", str(args.GlobalNormMethod),
            "--OutputInOriginalScale", str(args.OutputInOriginalScale),
            "--Rescale", "False",
            "--FitnessMode", str(args.FitnessMode),
            "--LambdaCov", str(args.LambdaCov),
            "--LambdaOut", str(args.LambdaOut),
            "--LambdaComp", str(args.LambdaComp),
        ]

        print(f"[{i:02d}/{len(rows):02d}] RUN  {stem}")
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        log_out.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

        if proc.returncode != 0:
            summary.append({"model": stem, "status": "FAIL", "reason": f"returncode:{proc.returncode}", "log": str(log_out)})
            print(f"[{i:02d}/{len(rows):02d}] FAIL {stem} rc={proc.returncode}")
            continue

        try:
            render_result(json_out, png_out, max_points=args.max_points_render, draw_points=True)
        except Exception as e:
            summary.append({"model": stem, "status": "FAIL", "reason": f"render:{e}", "json": str(json_out), "log": str(log_out)})
            print(f"[{i:02d}/{len(rows):02d}] FAIL {stem} render")
            continue

        metrics = {}
        try:
            d = json.loads(json_out.read_text(encoding="utf-8"))
            metrics = {
                "fitness": d.get("best_fitness"),
                "num_superquadrics": d.get("summary", {}).get("num_superquadrics"),
                "inlier_points_total": d.get("summary", {}).get("inlier_points_total"),
            }
        except Exception:
            pass

        summary.append({"model": stem, "status": "OK", **metrics, "json": str(json_out), "png": str(png_out), "log": str(log_out)})
        print(f"[{i:02d}/{len(rows):02d}] OK   {stem}")

    summary_csv = out_root / "summary.csv"
    fieldnames = sorted({k for row in summary for k in row.keys()}) if summary else ["model", "status"]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    ok = sum(1 for x in summary if x.get("status") == "OK")
    fail = sum(1 for x in summary if x.get("status") == "FAIL")
    skip = sum(1 for x in summary if x.get("status") == "SKIP")
    print("\nSummary")
    print(f"- total: {len(summary)}")
    print(f"- ok   : {ok}")
    print(f"- fail : {fail}")
    print(f"- skip : {skip}")
    print(f"- out  : {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
