#!/usr/bin/env python3
"""Generate pregrasp-initialized grasp poses for the Figure 4 SQ models."""

from __future__ import annotations

import argparse
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

from grasp_generator.optimize import GraspOptimizer, GraspResult
from grasp_generator.pregrasp import generate_pregrasps
from grasp_generator.simple_hand import ParallelJawHand, ParallelJawPose
from grasp_generator.superquadric_geometry import SuperquadricModel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "exp5/grasp_generation_study/results/figure4_numbered_cmaes/scaled_json_20cm"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "exp5/grasp_generation_study/results/pregrasp_grasps"


def pose_to_dict(pose: ParallelJawPose) -> dict:
    return {
        "translation_m": [float(x) for x in pose.translation],
        "rotation_matrix": pose.rotation_matrix.tolist(),
        "width_m": float(pose.width),
    }


def result_to_dict(result: GraspResult) -> dict:
    return {
        "object": result.object_name,
        "target_sq": f"SQ{result.target_sq}",
        "pregrasp": pose_to_dict(result.pregrasp),
        "grasp": pose_to_dict(result.grasp),
        "objective": result.objective,
        "contact_distance_m": result.contact_distance_m,
        "max_penetration_proxy_m": result.max_penetration_proxy_m,
        "q1_proxy": result.q1_proxy,
        "success": result.success,
    }


def diverse_select(results: list[GraspResult], count: int) -> list[GraspResult]:
    ordered = sorted(
        results,
        key=lambda item: (
            not item.success,
            -item.q1_proxy,
            item.max_penetration_proxy_m,
            item.contact_distance_m,
            item.objective,
        ),
    )
    selected: list[GraspResult] = []
    for item in ordered:
        if len(selected) >= count:
            break
        keep = True
        for old in selected:
            translation_gap = np.linalg.norm(item.grasp.translation - old.grasp.translation)
            width_gap = abs(item.grasp.width - old.grasp.width)
            same_target = item.target_sq == old.target_sq
            if same_target and translation_gap < 0.015 and width_gap < 0.01:
                keep = False
                break
        if keep:
            selected.append(item)
    if len(selected) < count:
        seen = {id(item) for item in selected}
        for item in ordered:
            if len(selected) >= count:
                break
            if id(item) not in seen:
                selected.append(item)
                seen.add(id(item))
    return selected


def render_result(model: SuperquadricModel, hand: ParallelJawHand, result: GraspResult, output_path: Path) -> None:
    fig = plt.figure(figsize=(6.2, 5.8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(20, len(model.superquadrics))))
    for index, sq in enumerate(model.superquadrics):
        points = model.surface_points_by_id[sq.id].reshape(36, 72, 3)
        ax.plot_surface(
            points[:, :, 0],
            points[:, :, 1],
            points[:, :, 2],
            color=colors[index],
            alpha=0.34 if sq.id != result.target_sq else 0.58,
            linewidth=0,
            antialiased=True,
        )
        ax.text(*sq.translation, f"SQ{sq.id}", fontsize=7, weight="bold")

    for pose, color, label in (
        (result.pregrasp, "#777777", "pregrasp"),
        (result.grasp, "#d62728", "optimized"),
    ):
        tips, palm = hand.segment_points(pose)
        ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2], c=color, s=24, label=label)
        ax.plot([tips[0, 0], tips[1, 0]], [tips[0, 1], tips[1, 1]], [tips[0, 2], tips[1, 2]], c=color, lw=2.0)
        for tip in tips:
            ax.plot([tip[0], palm[0]], [tip[1], palm[1]], [tip[2], palm[2]], c=color, lw=1.2, alpha=0.8)

    all_points = np.vstack((model.surface_points, hand.collision_points(result.pregrasp), hand.collision_points(result.grasp)))
    pmin = all_points.min(axis=0)
    pmax = all_points.max(axis=0)
    center = 0.5 * (pmin + pmax)
    radius = max(float(np.max(pmax - pmin)) / 2, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18.0, azim=-58.0)
    ax.set_title(f"{result.object_name} SQ{result.target_sq}: pregrasp to grasp", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_object(path: Path, output_root: Path, num_grasps: int, max_candidates: int, maxiter: int) -> list[GraspResult]:
    model = SuperquadricModel.from_json(path)
    hand = ParallelJawHand()
    optimizer = GraspOptimizer(model, hand)
    pregrasps = generate_pregrasps(model)
    results = [optimizer.optimize(candidate, maxiter=maxiter) for candidate in pregrasps[:max_candidates]]
    selected = diverse_select(results, num_grasps)
    object_dir = output_root / "visualizations" / model.object_name
    if object_dir.exists():
        for old_png in object_dir.glob("*.png"):
            old_png.unlink()
    for index, result in enumerate(selected, start=1):
        render_result(model, hand, result, object_dir / f"grasp_{index:02d}_SQ{result.target_sq}.png")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-grasps", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--maxiter", type=int, default=180)
    parser.add_argument("--objects", nargs="*", default=["all"])
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_root.glob("*_20cm.json"))
    if args.objects != ["all"]:
        wanted = set(args.objects)
        paths = [path for path in paths if path.name.replace("_20cm.json", "") in wanted]

    all_results: list[GraspResult] = []
    for path in paths:
        selected = run_object(path, args.output_root, args.num_grasps, args.max_candidates, args.maxiter)
        all_results.extend(selected)
        print(f"{path.name}: selected {len(selected)} grasps")

    json_path = args.output_root / "selected_grasps.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([result_to_dict(result) for result in all_results], handle, indent=2)

    csv_path = args.output_root / "selected_grasps_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "object",
            "grasp_id",
            "target_sq",
            "success",
            "objective",
            "contact_distance_mm",
            "max_penetration_proxy_mm",
            "q1_proxy",
            "width_mm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        counters: dict[str, int] = {}
        for result in all_results:
            counters[result.object_name] = counters.get(result.object_name, 0) + 1
            writer.writerow(
                {
                    "object": result.object_name,
                    "grasp_id": counters[result.object_name],
                    "target_sq": f"SQ{result.target_sq}",
                    "success": int(result.success),
                    "objective": f"{result.objective:.8f}",
                    "contact_distance_mm": f"{1000.0 * result.contact_distance_m:.4f}",
                    "max_penetration_proxy_mm": f"{1000.0 * result.max_penetration_proxy_m:.4f}",
                    "q1_proxy": f"{result.q1_proxy:.4f}",
                    "width_mm": f"{1000.0 * result.grasp.width:.4f}",
                }
            )

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
