#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "msr_matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


OBJECTS = [
    "CokePlasticLarge_25k",
    "Deodorant_25k",
    "FizzyTabletsCalcium_25k",
    "HamburgerSauce_25k",
    "Heart1_25k",
    "MelforBottle_25k",
    "Moon_25k",
    "Slotted_Screwdriver_25k",
]

RUNNERS = {
    "grid": "run_hierarchical_ems_grid_4params.py",
    "pso": "run_hierarchical_ems_pso_4params.py",
    "cmaes": "run_hierarchical_ems_cmaes_4params.py",
}

COLORS = {
    "grid": "#4477AA",
    "pso": "#228833",
    "cmaes": "#CC6677",
}


def parse_list(value: str, cast=str) -> list:
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def sigma_tag(sigma: float) -> str:
    return f"sigma_{sigma:.3f}".replace(".", "p")


def read_ply_xyz(path: Path) -> np.ndarray:
    try:
        from plyfile import PlyData

        ply = PlyData.read(str(path))
        vertex = ply["vertex"]
        return np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(float)
    except Exception:
        pass

    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY: {path}")
            text = line.decode("utf-8", errors="ignore").strip()
            header.append(text)
            if text == "end_header":
                break
        fmt = next((x for x in header if x.startswith("format ")), "")
        vertex_line = next((x for x in header if x.startswith("element vertex ")), "")
        if "ascii" not in fmt or not vertex_line:
            raise RuntimeError(f"Install plyfile to read this PLY format: {path}")
        count = int(vertex_line.split()[-1])
        points = np.loadtxt(f, dtype=float, usecols=(0, 1, 2), max_rows=count)
        return points.reshape(-1, 3)


def write_ply_xyz(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        np.savetxt(f, points, fmt="%.9f")


def clean_bbox_diagonal(points: np.ndarray) -> float:
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    return max(diagonal, 1e-12)


def generate_noisy_clouds(
    repo: Path,
    output: Path,
    objects: list[str],
    sigmas: list[float],
    seeds: list[int],
    overwrite: bool,
) -> None:
    clean_root = repo / "data" / "KIT_ObjectModels_25k_ply"
    noisy_root = output / "pointclouds"
    manifest_rows = []
    object_iter = tqdm(objects, desc="Generate noisy clouds", unit="object") if tqdm is not None else objects
    for obj in object_iter:
        clean_path = clean_root / f"{obj}.ply"
        clean = read_ply_xyz(clean_path)
        reference_scale = clean_bbox_diagonal(clean)
        for sigma in sigmas:
            for seed in seeds:
                out = noisy_root / sigma_tag(sigma) / f"seed_{seed}" / f"{obj}.ply"
                object_seed = (int(seed) + zlib.crc32(obj.encode("utf-8"))) % (2**32)
                rng = np.random.default_rng(object_seed)
                noisy = clean + rng.normal(0.0, sigma * reference_scale, size=clean.shape)
                if overwrite or not out.exists():
                    write_ply_xyz(out, noisy)
                manifest_rows.append(
                    {
                        "object": obj,
                        "noise_sigma_normalized": sigma,
                        "noise_seed": seed,
                        "clean_path": str(clean_path),
                        "noisy_path": str(out),
                        "clean_bbox_diagonal": reference_scale,
                        "noise_std_original_units": sigma * reference_scale,
                        "num_points": len(clean),
                    }
                )
    write_csv(output / "noise_manifest.csv", manifest_rows)


def common_runner_args(min_coverage: float, lambda_min_coverage: float) -> list[str]:
    return [
        "--runtime",
        "--GlobalNormalize", "True",
        "--GlobalNormMethod", "ems_matlab",
        "--OutputInOriginalScale", "True",
        "--Rescale", "False",
        "--AdaptiveUpperBound", "True",
        "--MaxLayer", "5",
        "--TauIn", "0.1",
        "--TauSplit", "0.8",
        "--MinClusterRatio", "0.0008",
        "--MaxIterationEM", "20",
        "--ToleranceEM", "0.001",
        "--RelativeToleranceEM", "0.2",
        "--MaxOptiIterations", "2",
        "--MaxiSwitch", "2",
        "--FitnessMode", "distance_coverage_outlier_complexity",
        "--LambdaCov", "0.05",
        "--LambdaOut", "0.0",
        "--LambdaComp", "0.001",
        "--MinCoverage", str(min_coverage),
        "--LambdaMinCoverage", str(lambda_min_coverage),
        "--OutlierRatioMin", "0.1",
        "--OutlierRatioMax", "0.95",
        "--SigmaMin", "0.0",
        "--SigmaMax", "0.8",
        "--EpsMin", "1.0",
        "--EpsMax", "3.0",
        "--MinPointsMin", "10",
        "--MinPointsMax", "120",
    ]


def method_args(method: str, seed: int, profile: str) -> list[str]:
    if profile == "smoke":
        if method == "grid":
            return ["--gridSteps", "2"]
        if method == "pso":
            return [
                "--swarmsize", "3", "--maxiter", "2",
                "--omega", "0.72", "--c1", "1.49", "--c2", "1.49",
                "--tol", "1e-6", "--patience", "5",
                "--fitnessThreshold", "1e-4", "--seed", str(seed),
            ]
        return [
            "--popsize", "4", "--maxiter", "2", "--cmaSigma", "0.15",
            "--tol", "1e-6", "--patience", "5",
            "--fitnessThreshold", "1e-4", "--seed", str(seed),
        ]

    if method == "grid":
        return ["--gridSteps", "5"]
    if method == "pso":
        return [
            "--swarmsize", "25", "--maxiter", "25",
            "--omega", "0.72", "--c1", "1.49", "--c2", "1.49",
            "--tol", "1e-6", "--patience", "5",
            "--fitnessThreshold", "1e-4", "--seed", str(seed),
        ]
    return [
        "--popsize", "25", "--maxiter", "25", "--cmaSigma", "0.15",
        "--tol", "1e-6", "--patience", "5",
        "--fitnessThreshold", "1e-4", "--seed", str(seed),
    ]


def run_reconstructions(
    repo: Path,
    output: Path,
    objects: list[str],
    sigmas: list[float],
    seeds: list[int],
    methods: list[str],
    profile: str,
    skip_existing: bool,
    min_coverage: float,
    lambda_min_coverage: float,
) -> None:
    jobs = []
    for sigma in sigmas:
        for seed in seeds:
            noisy_dir = output / "pointclouds" / sigma_tag(sigma) / f"seed_{seed}"
            for obj in objects:
                for method in methods:
                    # Grid is deterministic for a fixed noisy cloud. It still runs once
                    # for every noise seed because each seed defines a different cloud.
                    jobs.append((sigma, seed, obj, method, noisy_dir / f"{obj}.ply"))

    print(f"Reconstruction jobs: {len(jobs)} ({profile})", flush=True)
    job_iter = tqdm(jobs, desc=f"Reconstruct ({profile})", unit="run", dynamic_ncols=True) if tqdm is not None else jobs
    for index, (sigma, seed, obj, method, noisy_path) in enumerate(job_iter, 1):
        if tqdm is not None:
            job_iter.set_postfix_str(f"{sigma_tag(sigma)} | seed={seed} | {method} | {obj}")
        run_dir = output / "runs" / sigma_tag(sigma) / f"seed_{seed}" / method
        log_dir = output / "logs" / sigma_tag(sigma) / f"seed_{seed}" / method
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        out_json = run_dir / f"{obj}.json"
        out_log = log_dir / f"{obj}.log"
        if skip_existing and out_json.exists():
            if tqdm is None:
                print(f"[{index}/{len(jobs)}] REUSE {sigma_tag(sigma)} seed={seed} {method} {obj}", flush=True)
            else:
                tqdm.write(f"REUSE {sigma_tag(sigma)} seed={seed} {method} {obj}")
            continue
        runner = repo / "src" / RUNNERS[method]
        cmd = [
            sys.executable,
            str(runner),
            str(noisy_path),
            "--out",
            str(out_json),
            *common_runner_args(min_coverage, lambda_min_coverage),
            *method_args(method, seed, profile),
        ]
        if tqdm is None:
            print(f"[{index}/{len(jobs)}] START {sigma_tag(sigma)} seed={seed} {method} {obj}", flush=True)
        proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        out_log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        if proc.returncode != 0 or not out_json.exists():
            message = f"FAIL {sigma_tag(sigma)} seed={seed} {method} {obj}; return={proc.returncode}; log={out_log}"
            if tqdm is not None:
                tqdm.write(message)
            else:
                print(f"[{index}/{len(jobs)}] {message}", flush=True)
        elif tqdm is None:
            print(f"[{index}/{len(jobs)}] OK", flush=True)


def signed_power(value: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(value) * np.abs(value) ** exponent


def superquadric_mesh(sq: dict, n_eta: int = 64, n_omega: int = 128) -> tuple[np.ndarray, np.ndarray]:
    eps1, eps2 = [max(0.007, float(x)) for x in sq["shape"]]
    a1, a2, a3 = [float(x) for x in sq["scale"]]
    eta = np.linspace(-math.pi / 2, math.pi / 2, n_eta)
    omega = np.linspace(-math.pi, math.pi, n_omega, endpoint=False)
    ee, oo = np.meshgrid(eta, omega, indexing="ij")
    ce = signed_power(np.cos(ee), eps1)
    se = signed_power(np.sin(ee), eps1)
    co = signed_power(np.cos(oo), eps2)
    so = signed_power(np.sin(oo), eps2)
    local = np.stack([a1 * ce * co, a2 * ce * so, a3 * se], axis=-1)
    rot = Rotation.from_euler("ZYX", sq["euler"]).as_matrix()
    vertices = local.reshape(-1, 3) @ rot.T + np.asarray(sq["translation"], dtype=float)
    faces = []
    for i in range(n_eta - 1):
        for j in range(n_omega):
            j2 = (j + 1) % n_omega
            a = i * n_omega + j
            b = i * n_omega + j2
            c = (i + 1) * n_omega + j
            d = (i + 1) * n_omega + j2
            faces.append((a, c, b))
            faces.append((b, c, d))
    return vertices, np.asarray(faces, dtype=np.int64)


def sample_mesh(vertices: np.ndarray, faces: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    valid = area > 1e-14
    tri = tri[valid]
    area = area[valid]
    if len(area) == 0:
        return np.repeat(vertices[:1], count, axis=0)
    probs = area / area.sum()
    chosen = tri[rng.choice(len(tri), size=count, p=probs)]
    u = rng.random(count)
    v = rng.random(count)
    fold = u + v > 1.0
    u[fold] = 1.0 - u[fold]
    v[fold] = 1.0 - v[fold]
    return chosen[:, 0] + u[:, None] * (chosen[:, 1] - chosen[:, 0]) + v[:, None] * (
        chosen[:, 2] - chosen[:, 0]
    )


def sample_reconstruction(data: dict, count: int, seed: int) -> np.ndarray:
    quadrics = data.get("superquadrics", [])
    if not quadrics:
        return np.empty((0, 3), dtype=float)
    rng = np.random.default_rng(seed)
    meshes = [superquadric_mesh(sq) for sq in quadrics]
    areas = []
    for vertices, faces in meshes:
        tri = vertices[faces]
        areas.append(float((0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)).sum()))
    areas = np.asarray(areas)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        areas = np.ones(len(meshes))
    raw = count * areas / areas.sum()
    counts = np.maximum(32, np.floor(raw).astype(int))
    while counts.sum() > count:
        idx = int(np.argmax(counts))
        if counts[idx] <= 32:
            break
        counts[idx] -= 1
    while counts.sum() < count:
        counts[int(np.argmax(raw - counts))] += 1
    return np.concatenate(
        [sample_mesh(vertices, faces, int(n), rng) for (vertices, faces), n in zip(meshes, counts)],
        axis=0,
    )


def sample_reconstruction_by_primitive(data: dict, count: int, seed: int) -> list[np.ndarray]:
    quadrics = data.get("superquadrics", [])
    if not quadrics:
        return []
    rng = np.random.default_rng(seed)
    meshes = [superquadric_mesh(sq) for sq in quadrics]
    areas = np.asarray(
        [
            float(
                (
                    0.5
                    * np.linalg.norm(
                        np.cross(
                            vertices[faces][:, 1] - vertices[faces][:, 0],
                            vertices[faces][:, 2] - vertices[faces][:, 0],
                        ),
                        axis=1,
                    )
                ).sum()
            )
            for vertices, faces in meshes
        ]
    )
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        areas = np.ones(len(meshes))
    raw = count * areas / areas.sum()
    counts = np.maximum(32, np.floor(raw).astype(int))
    while counts.sum() > count:
        idx = int(np.argmax(counts))
        if counts[idx] <= 32:
            break
        counts[idx] -= 1
    while counts.sum() < count:
        counts[int(np.argmax(raw - counts))] += 1
    return [
        sample_mesh(vertices, faces, int(n), rng)
        for (vertices, faces), n in zip(meshes, counts)
    ]


def evaluate_pair(clean: np.ndarray, surface: np.ndarray, thresholds: list[float]) -> dict:
    reference_scale = clean_bbox_diagonal(clean)
    center = np.mean(clean, axis=0)
    clean_norm = (clean - center) / reference_scale
    surface_norm = (surface - center) / reference_scale
    tree_surface = cKDTree(surface_norm)
    tree_clean = cKDTree(clean_norm)
    d_clean_to_surface = tree_surface.query(clean_norm, workers=-1)[0]
    d_surface_to_clean = tree_clean.query(surface_norm, workers=-1)[0]
    result = {
        "clean_to_reconstruction_mean": float(np.mean(d_clean_to_surface)),
        "reconstruction_to_clean_mean": float(np.mean(d_surface_to_clean)),
        "chamfer_l1": float(np.mean(d_clean_to_surface) + np.mean(d_surface_to_clean)),
        "clean_to_reconstruction_p95": float(np.quantile(d_clean_to_surface, 0.95)),
        "reconstruction_to_clean_p95": float(np.quantile(d_surface_to_clean, 0.95)),
        "reference_bbox_diagonal": reference_scale,
    }
    for threshold in thresholds:
        key = f"{threshold:.3f}".replace(".", "p")
        recall = float(np.mean(d_clean_to_surface < threshold))
        precision = float(np.mean(d_surface_to_clean < threshold))
        fscore = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        result[f"recall_{key}"] = recall
        result[f"precision_{key}"] = precision
        result[f"fscore_{key}"] = fscore
    return result


def evaluate_assigned_inliers(
    clean: np.ndarray,
    data: dict,
    thresholds: list[float],
    surface_samples: int,
) -> dict:
    assignments = data.get("primitive_assignments", [])
    surfaces = sample_reconstruction_by_primitive(data, surface_samples, seed=12345)
    if len(assignments) != len(surfaces) or not assignments:
        return {}

    reference_scale = clean_bbox_diagonal(clean)
    center = np.mean(clean, axis=0)
    clean_norm = (clean - center) / reference_scale
    clean_to_surface_parts = []
    surface_to_clean_parts = []
    valid_inlier_indices = []

    for primitive_id, (assignment, surface) in enumerate(zip(assignments, surfaces)):
        if int(assignment.get("primitive_id", primitive_id)) != primitive_id:
            raise ValueError("primitive_assignments are not aligned with superquadrics.")
        indices = np.asarray(assignment.get("inlier_indices", []), dtype=np.int64)
        indices = indices[(indices >= 0) & (indices < len(clean))]
        if len(indices) == 0 or len(surface) == 0:
            continue
        assigned_clean = clean_norm[indices]
        surface_norm = (surface - center) / reference_scale
        clean_to_surface_parts.append(
            cKDTree(surface_norm).query(assigned_clean, workers=-1)[0]
        )
        surface_to_clean_parts.append(
            cKDTree(assigned_clean).query(surface_norm, workers=-1)[0]
        )
        valid_inlier_indices.append(indices)

    if not clean_to_surface_parts or not surface_to_clean_parts:
        return {}

    d_clean_to_surface = np.concatenate(clean_to_surface_parts)
    d_surface_to_clean = np.concatenate(surface_to_clean_parts)
    unique_indices = np.unique(np.concatenate(valid_inlier_indices))
    result = {
        "inlier_clean_to_reconstruction_mean": float(np.mean(d_clean_to_surface)),
        "inlier_reconstruction_to_clean_mean": float(np.mean(d_surface_to_clean)),
        "inlier_chamfer_l1": float(
            np.mean(d_clean_to_surface) + np.mean(d_surface_to_clean)
        ),
        "inlier_clean_to_reconstruction_p95": float(
            np.quantile(d_clean_to_surface, 0.95)
        ),
        "inlier_reconstruction_to_clean_p95": float(
            np.quantile(d_surface_to_clean, 0.95)
        ),
        "recorded_inlier_count": int(sum(len(x) for x in valid_inlier_indices)),
        "recorded_unique_inlier_count": int(len(unique_indices)),
        "recorded_unique_inlier_coverage": float(len(unique_indices) / len(clean)),
    }
    for threshold in thresholds:
        key = f"{threshold:.3f}".replace(".", "p")
        recall = float(np.mean(d_clean_to_surface < threshold))
        precision = float(np.mean(d_surface_to_clean < threshold))
        fscore = (
            0.0
            if precision + recall == 0
            else 2.0 * precision * recall / (precision + recall)
        )
        result[f"inlier_recall_{key}"] = recall
        result[f"inlier_precision_{key}"] = precision
        result[f"inlier_fscore_{key}"] = fscore
    return result


def clean_json_path(repo: Path, method: str, obj: str, seed: int) -> Path:
    return (
        repo
        / "exp2_3"
        / "results"
        / "e2_multi_5x4_f005_c001"
        / "json"
        / method
        / f"{obj}__seed{seed}.json"
    )


def render_cloud(
    path: Path,
    clean: np.ndarray,
    noisy: np.ndarray | None,
    surface: np.ndarray | None,
    title: str,
    limits: tuple[np.ndarray, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5.2, 5.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    rng = np.random.default_rng(0)
    if noisy is not None:
        idx = rng.choice(len(noisy), size=min(6000, len(noisy)), replace=False)
        ax.scatter(noisy[idx, 0], noisy[idx, 1], noisy[idx, 2], s=1.0, c="#D62728", alpha=0.35, depthshade=False)
    elif surface is None:
        idx = rng.choice(len(clean), size=min(6000, len(clean)), replace=False)
        ax.scatter(clean[idx, 0], clean[idx, 1], clean[idx, 2], s=1.2, c="#333333", alpha=0.8, depthshade=False)
    if surface is not None:
        idx = rng.choice(len(surface), size=min(10000, len(surface)), replace=False)
        ax.scatter(surface[idx, 0], surface[idx, 1], surface[idx, 2], s=1.0, c="#1F77B4", alpha=0.55, depthshade=False)
    lo, hi = limits
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.maximum(hi - lo, 1e-9))
    ax.view_init(elev=18, azim=-70)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def evaluate_and_render(
    repo: Path,
    output: Path,
    objects: list[str],
    sigmas: list[float],
    seeds: list[int],
    methods: list[str],
    thresholds: list[float],
    surface_samples: int,
) -> None:
    clean_root = repo / "data" / "KIT_ObjectModels_25k_ply"
    rows = []
    # Include clean 5x4 reconstructions to enable degradation metrics.
    conditions = [(0.0, 0)] + [(sigma, seed) for sigma in sigmas for seed in seeds]
    object_iter = tqdm(objects, desc="Evaluate and render", unit="object", dynamic_ncols=True) if tqdm is not None else objects
    for obj in object_iter:
        if tqdm is not None:
            object_iter.set_postfix_str(obj)
        clean = read_ply_xyz(clean_root / f"{obj}.ply")
        margin = 0.08 * np.max(np.ptp(clean, axis=0))
        limits = (clean.min(axis=0) - margin, clean.max(axis=0) + margin)
        render_cloud(output / "figures" / obj / "clean_pointcloud.png", clean, None, None, f"{obj}: clean point cloud", limits)
        for sigma, seed in conditions:
            if sigma == 0.0:
                noisy = clean
            else:
                noisy_path = output / "pointclouds" / sigma_tag(sigma) / f"seed_{seed}" / f"{obj}.ply"
                noisy = read_ply_xyz(noisy_path)
                render_cloud(
                    output / "figures" / obj / sigma_tag(sigma) / f"seed_{seed}" / "noisy_pointcloud.png",
                    clean,
                    noisy,
                    None,
                    f"{obj}: noisy point cloud ($\\sigma_n={sigma}$)",
                    limits,
                )
            for method in methods:
                if sigma == 0.0:
                    json_path = clean_json_path(repo, method, obj, seed=0)
                    eval_seed = 0
                else:
                    json_path = output / "runs" / sigma_tag(sigma) / f"seed_{seed}" / method / f"{obj}.json"
                    eval_seed = seed
                if not json_path.exists():
                    continue
                data = json.loads(json_path.read_text(encoding="utf-8"))
                surface = sample_reconstruction(data, surface_samples, seed=12345)
                if len(surface) == 0:
                    continue
                metrics = evaluate_pair(clean, surface, thresholds)
                inlier_metrics = evaluate_assigned_inliers(
                    clean, data, thresholds, surface_samples
                )
                details = data.get("fitness_details", {})
                timing = data.get("timing", {})
                row = {
                    "object": obj,
                    "noise_sigma": sigma,
                    "noise_seed": seed,
                    "method": method,
                    "json_path": str(json_path),
                    "objective": details.get("fitness", data.get("best_fitness")),
                    "noisy_distance_fit": details.get("distance_fit"),
                    "noisy_coverage": details.get("coverage_ratio"),
                    "num_superquadrics": details.get("num_superquadrics", len(data.get("superquadrics", []))),
                    "runtime_ms": timing.get("total_runtime_ms"),
                    **metrics,
                    **inlier_metrics,
                }
                rows.append(row)
                fig_dir = output / "figures" / obj / sigma_tag(sigma) / f"seed_{eval_seed}" / method
                render_cloud(
                    fig_dir / "reconstruction_only.png",
                    clean,
                    None,
                    surface,
                    f"{method.upper()}: reconstruction",
                    limits,
                )
                render_cloud(
                    fig_dir / "reconstruction_with_noisy_points.png",
                    clean,
                    noisy,
                    surface,
                    f"{method.upper()}: reconstruction + input",
                    limits,
                )
    write_csv(output / "metrics_raw.csv", rows)
    add_degradation_and_summaries(output, rows)


def add_degradation_and_summaries(output: Path, rows: list[dict]) -> None:
    clean_map = {
        (r["object"], r["method"]): r
        for r in rows
        if float(r["noise_sigma"]) == 0.0
    }
    enriched = []
    for row in rows:
        row = dict(row)
        baseline = clean_map.get((row["object"], row["method"]))
        if baseline is not None:
            row["delta_chamfer_l1"] = float(row["chamfer_l1"]) - float(baseline["chamfer_l1"])
            for key in list(row):
                if key.startswith("fscore_"):
                    row[f"delta_{key}"] = float(baseline[key]) - float(row[key])
        enriched.append(row)
    write_csv(output / "metrics_with_degradation.csv", enriched)

    groups: dict[tuple[float, str], list[dict]] = {}
    for row in enriched:
        groups.setdefault((float(row["noise_sigma"]), str(row["method"])), []).append(row)
    summary = []
    numeric_keys = [
        "chamfer_l1",
        "clean_to_reconstruction_mean",
        "reconstruction_to_clean_mean",
        "clean_to_reconstruction_p95",
        "reconstruction_to_clean_p95",
        "delta_chamfer_l1",
        "num_superquadrics",
        "runtime_ms",
    ]
    numeric_keys.extend(sorted({k for r in enriched for k in r if k.startswith(("precision_", "recall_", "fscore_", "delta_fscore_"))}))
    numeric_keys.extend(
        sorted(
            {
                k
                for r in enriched
                for k in r
                if k.startswith(("inlier_", "recorded_"))
            }
        )
    )
    for (sigma, method), group in sorted(groups.items()):
        item = {"noise_sigma": sigma, "method": method, "n_runs": len(group)}
        for key in numeric_keys:
            vals = np.asarray([float(r[key]) for r in group if r.get(key) not in (None, "")], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                item[f"{key}_mean"] = float(vals.mean())
                item[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                item[f"{key}_median"] = float(np.median(vals))
                item[f"{key}_q1"] = float(np.quantile(vals, 0.25))
                item[f"{key}_q3"] = float(np.quantile(vals, 0.75))
        summary.append(item)
    write_csv(output / "metrics_summary.csv", summary)
    plot_summary(output, enriched)


def plot_summary(output: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=180)
    sigmas = sorted({float(r["noise_sigma"]) for r in rows})
    for method in RUNNERS:
        method_rows = [r for r in rows if r["method"] == method]
        x, cd_med, cd_q1, cd_q3, fs_med, fs_q1, fs_q3, n_med, n_q1, n_q3 = ([] for _ in range(10))
        for sigma in sigmas:
            group = [r for r in method_rows if float(r["noise_sigma"]) == sigma]
            if not group:
                continue
            x.append(sigma)
            for key, med, q1, q3 in [
                ("chamfer_l1", cd_med, cd_q1, cd_q3),
                ("fscore_0p010", fs_med, fs_q1, fs_q3),
                ("num_superquadrics", n_med, n_q1, n_q3),
            ]:
                vals = np.asarray([float(r[key]) for r in group])
                med.append(np.median(vals))
                q1.append(np.quantile(vals, 0.25))
                q3.append(np.quantile(vals, 0.75))
        color = COLORS[method]
        for ax, med, q1, q3 in [
            (axes[0], cd_med, cd_q1, cd_q3),
            (axes[1], fs_med, fs_q1, fs_q3),
            (axes[2], n_med, n_q1, n_q3),
        ]:
            ax.plot(x, med, marker="o", label=method.upper(), color=color)
            ax.fill_between(x, q1, q3, color=color, alpha=0.15)
    axes[0].set_ylabel("Clean-reference Chamfer-L1")
    axes[1].set_ylabel(r"Surface F-score ($\tau=0.01$)")
    axes[2].set_ylabel("Number of superquadrics")
    for ax in axes:
        ax.set_xlabel(r"Noise level $\sigma_n$")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig_dir = output / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "noise_metrics_summary.png", bbox_inches="tight")
    fig.savefig(fig_dir / "noise_metrics_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_config(output: Path, args: argparse.Namespace) -> None:
    config = {
        "objects": args.objects,
        "noise_sigmas_normalized": args.sigmas,
        "noise_seeds": args.seeds,
        "methods": args.methods,
        "profile": args.profile,
        "surface_samples": args.surface_samples,
        "minimum_layer_coverage": args.min_coverage,
        "minimum_coverage_penalty_weight": args.lambda_min_coverage,
        "fscore_thresholds_normalized": args.thresholds,
        "clean_reference_results": "exp2_3/results/e2_multi_5x4_f005_c001",
        "noise_definition": "Per-axis Gaussian noise standard deviation is sigma_n times the clean point-cloud bounding-box diagonal.",
        "optimizer_configuration": {
            "grid": "5^4 points per active layer",
            "pso": "swarmsize=25, maxiter=25",
            "cmaes": "popsize=25, maxiter=25",
        },
        "objective": "distance_fit + 0.05*(1-coverage) + 0.001*N_sq",
    }
    (output / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Noise robustness experiment for the eight Figure 4 KIT objects.")
    parser.add_argument("--stage", choices=["generate", "run", "evaluate", "all"], default="all")
    parser.add_argument("--output", default="exp4_noise/results/noise_8objects_5x4")
    parser.add_argument("--objects", default=",".join(OBJECTS))
    parser.add_argument("--sigmas", default="0.005,0.01")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--methods", default="grid,pso,cmaes")
    parser.add_argument("--profile", choices=["5x4", "smoke"], default="5x4")
    parser.add_argument("--thresholds", default="0.01,0.02")
    parser.add_argument("--surface-samples", type=int, default=20000)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--lambda-min-coverage", type=float, default=0.0)
    parser.add_argument("--overwrite-noise", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output = (repo / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.objects = parse_list(args.objects)
    args.sigmas = parse_list(args.sigmas, float)
    args.seeds = parse_list(args.seeds, int)
    args.methods = parse_list(args.methods)
    args.thresholds = parse_list(args.thresholds, float)
    for method in args.methods:
        if method not in RUNNERS:
            raise ValueError(f"Unsupported method: {method}")
    write_config(output, args)

    if args.stage in {"generate", "all"}:
        generate_noisy_clouds(repo, output, args.objects, args.sigmas, args.seeds, args.overwrite_noise)
    if args.stage in {"run", "all"}:
        run_reconstructions(
            repo,
            output,
            args.objects,
            args.sigmas,
            args.seeds,
            args.methods,
            args.profile,
            skip_existing=not args.no_skip_existing,
            min_coverage=args.min_coverage,
            lambda_min_coverage=args.lambda_min_coverage,
        )
    if args.stage in {"evaluate", "all"}:
        evaluate_and_render(
            repo,
            output,
            args.objects,
            args.sigmas,
            args.seeds,
            args.methods,
            args.thresholds,
            args.surface_samples,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
