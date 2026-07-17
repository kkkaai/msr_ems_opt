#!/usr/bin/env python3
"""Generate pregrasp-initialized grasps with hand/folding_hand_right kinematics."""

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from grasp_generator.folding_hand import (
    RIGHT_CONTROL_LIMITS,
    RIGHT_CONTROL_NAMES,
    FoldingHandPose,
    FoldingHandRightKinematics,
)
from grasp_generator.pregrasp import generate_pregrasps
from grasp_generator.simple_hand import make_rotation
from grasp_generator.superquadric_geometry import SuperquadricModel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "exp5/grasp_generation_study/results/figure4_numbered_cmaes/scaled_json_20cm"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "exp5/grasp_generation_study/results/folding_hand_right_pregrasp_grasps"


def make_hand_pregrasp(model: SuperquadricModel, target_sq: int, approach_axis: np.ndarray, closing_axis: np.ndarray) -> FoldingHandPose:
    hand = FoldingHandRightKinematics.default()
    sq = model.sq_by_id(target_sq)
    anchors = model.target_anchor_pair(target_sq, closing_axis)
    midpoint = 0.5 * (anchors["left_anchor"] + anchors["right_anchor"])
    rotation = make_rotation(closing_axis, approach_axis)
    controls = hand.open_controls()
    local_probe = FoldingHandPose(translation=np.zeros(3), rotation_matrix=np.eye(3), controls=controls)
    local_contact_center = hand.contact_points(local_probe).mean(axis=0)
    desired_contact_center = midpoint + approach_axis * 0.035
    translation = desired_contact_center - rotation @ local_contact_center
    return FoldingHandPose(translation=translation, rotation_matrix=rotation, controls=controls)


def optimize_candidate(
    model: SuperquadricModel,
    hand: FoldingHandRightKinematics,
    pregrasp: FoldingHandPose,
    target_sq: int,
    maxiter: int,
) -> dict:
    target = model.sq_by_id(target_sq)
    target_points = model.surface_points_by_id[target_sq]
    target_tree = cKDTree(target_points)
    start = pregrasp.as_vector()
    L = model.reference_length
    contact_indices = np.array([0, 1, 2, 3, 4])
    bounds = [
        (start[0] - 0.08, start[0] + 0.08),
        (start[1] - 0.08, start[1] + 0.08),
        (start[2] - 0.08, start[2] + 0.08),
        (start[3] - 0.5, start[3] + 0.5),
        (start[4] - 0.5, start[4] + 0.5),
        (start[5] - 0.5, start[5] + 0.5),
    ] + [RIGHT_CONTROL_LIMITS[name] for name in RIGHT_CONTROL_NAMES]

    def unpack(vector: np.ndarray) -> FoldingHandPose:
        pose = FoldingHandPose.from_vector(vector)
        pose.controls = np.array([np.clip(v, *RIGHT_CONTROL_LIMITS[name]) for v, name in zip(pose.controls, RIGHT_CONTROL_NAMES)])
        return pose

    def nearest_target_distance(points: np.ndarray) -> np.ndarray:
        return target_tree.query(points)[0]

    def energy(vector: np.ndarray) -> float:
        pose = unpack(vector)
        contacts = hand.contact_points(pose)
        contact_distance = nearest_target_distance(contacts[contact_indices])
        collision = hand.collision_points(pose)
        collision_distance, _ = model.nearest_surface_distance(collision)
        inside = model.union_inside_margin(collision) > 0.0
        penetration = collision_distance[inside] if np.any(inside) else np.array([0.0])
        close_ref = hand.mid_closed_controls()
        joint_ref = 0.02 * float(np.mean(((pose.controls - close_ref) / np.pi) ** 2))
        drift = 0.01 * float(np.sum((pose.translation - start[:3]) ** 2)) / (L**2)
        return (
            100.0 * float(np.mean((contact_distance / L) ** 2))
            + 100.0 * float(np.mean((penetration / L) ** 2))
            + joint_ref
            + drift
        )

    result = minimize(energy, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-8})
    grasp = unpack(result.x)
    contacts = hand.contact_points(grasp)
    contact_distance = nearest_target_distance(contacts)
    collision = hand.collision_points(grasp)
    collision_distance, _ = model.nearest_surface_distance(collision)
    inside = model.union_inside_margin(collision) > 0.0
    max_pen = float(np.max(collision_distance[inside])) if np.any(inside) else 0.0
    normals = target.normal_at(contacts[:2])
    q1_proxy = max(0.0, -float(np.dot(normals[0], normals[1])))
    success = float(np.mean(contact_distance)) <= 0.006 and max_pen <= 0.004 and q1_proxy >= 0.10
    return {
        "object": model.object_name,
        "target_sq": target_sq,
        "pregrasp": pregrasp,
        "grasp": grasp,
        "objective": float(result.fun),
        "contact_distance_m": float(np.mean(contact_distance)),
        "max_penetration_proxy_m": max_pen,
        "q1_proxy": q1_proxy,
        "success": success,
    }


def select_results(results: list[dict], count: int) -> list[dict]:
    ordered = sorted(
        results,
        key=lambda row: (
            not row["success"],
            -row["q1_proxy"],
            row["max_penetration_proxy_m"],
            row["contact_distance_m"],
            row["objective"],
        ),
    )
    return ordered[:count]


def pose_dict(pose: FoldingHandPose) -> dict:
    return {
        "translation_m": pose.translation.tolist(),
        "rotation_matrix": pose.rotation_matrix.tolist(),
        "controls": {name: float(value) for name, value in zip(RIGHT_CONTROL_NAMES, pose.controls)},
    }


def render(model: SuperquadricModel, hand: FoldingHandRightKinematics, row: dict, output_path: Path) -> None:
    fig = plt.figure(figsize=(6.2, 5.8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(20, len(model.superquadrics))))
    for index, sq in enumerate(model.superquadrics):
        points = model.surface_points_by_id[sq.id].reshape(36, 72, 3)
        ax.plot_surface(points[:, :, 0], points[:, :, 1], points[:, :, 2], color=colors[index], alpha=0.24, linewidth=0)
    for pose, color, label in ((row["pregrasp"], "#777777", "pregrasp"), (row["grasp"], "#d62728", "optimized")):
        contacts = hand.contact_points(pose)
        palm = pose.translation
        ax.scatter(contacts[:, 0], contacts[:, 1], contacts[:, 2], c=color, s=20, label=label)
        for point in contacts:
            ax.plot([palm[0], point[0]], [palm[1], point[1]], [palm[2], point[2]], c=color, lw=1.0, alpha=0.8)
    for triangles in hand.transformed_mesh_triangles(row["grasp"], max_triangles_per_mesh=700):
        mesh = Poly3DCollection(
            triangles,
            facecolor=(0.18, 0.34, 0.82, 0.78),
            edgecolor=(0.04, 0.08, 0.18, 0.22),
            linewidth=0.08,
        )
        ax.add_collection3d(mesh)
    all_points = np.vstack((model.surface_points, hand.collision_points(row["pregrasp"]), hand.collision_points(row["grasp"])))
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    radius = max(float(np.max(np.ptp(all_points, axis=0))) / 2, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-58)
    ax.set_title(f"{model.object_name} folding hand SQ{row['target_sq']}", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_object(path: Path, output_root: Path, num_grasps: int, max_candidates: int, maxiter: int, skip_render: bool) -> list[dict]:
    model = SuperquadricModel.from_json(path)
    hand = FoldingHandRightKinematics.default()
    candidates = generate_pregrasps(model, target_limit=min(5, len(model.superquadrics)))
    results = []
    for candidate in candidates[:max_candidates]:
        pregrasp = make_hand_pregrasp(model, candidate.target_sq, candidate.approach_axis, candidate.closing_axis)
        results.append(optimize_candidate(model, hand, pregrasp, candidate.target_sq, maxiter))
    selected = select_results(results, num_grasps)
    if not skip_render:
        object_dir = output_root / "visualizations" / model.object_name
        if object_dir.exists():
            for old_png in object_dir.glob("*.png"):
                old_png.unlink()
        for index, row in enumerate(selected, start=1):
            render(model, hand, row, object_dir / f"grasp_{index:02d}_SQ{row['target_sq']}.png")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-grasps", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--objects", nargs="*", default=["all"])
    parser.add_argument("--skip-render", action="store_true", help="Write JSON/CSV without generating PNG visualizations.")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_root.glob("*_20cm.json"))
    if args.objects != ["all"]:
        wanted = set(args.objects)
        paths = [path for path in paths if path.name.replace("_20cm.json", "") in wanted]

    all_rows = []
    for path in paths:
        rows = run_object(path, args.output_root, args.num_grasps, args.max_candidates, args.maxiter, args.skip_render)
        all_rows.extend(rows)
        print(f"{path.name}: selected {len(rows)} folding-hand grasps")

    with (args.output_root / "selected_folding_hand_grasps.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "object": row["object"],
                    "target_sq": f"SQ{row['target_sq']}",
                    "pregrasp": pose_dict(row["pregrasp"]),
                    "grasp": pose_dict(row["grasp"]),
                    "objective": row["objective"],
                    "contact_distance_m": row["contact_distance_m"],
                    "max_penetration_proxy_m": row["max_penetration_proxy_m"],
                    "q1_proxy": row["q1_proxy"],
                    "success": row["success"],
                }
                for row in all_rows
            ],
            handle,
            indent=2,
        )

    with (args.output_root / "selected_folding_hand_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["object", "grasp_id", "target_sq", "success", "objective", "contact_distance_mm", "max_penetration_proxy_mm", "q1_proxy"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        counters: dict[str, int] = {}
        for row in all_rows:
            counters[row["object"]] = counters.get(row["object"], 0) + 1
            writer.writerow(
                {
                    "object": row["object"],
                    "grasp_id": counters[row["object"]],
                    "target_sq": f"SQ{row['target_sq']}",
                    "success": int(row["success"]),
                    "objective": f"{row['objective']:.8f}",
                    "contact_distance_mm": f"{1000.0 * row['contact_distance_m']:.4f}",
                    "max_penetration_proxy_mm": f"{1000.0 * row['max_penetration_proxy_m']:.4f}",
                    "q1_proxy": f"{row['q1_proxy']:.4f}",
                }
            )


if __name__ == "__main__":
    main()
