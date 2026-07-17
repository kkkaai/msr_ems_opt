#!/usr/bin/env python3
"""Render the folding hand model meshes in open and half-closed poses."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "msr_matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from grasp_generator.folding_hand import FoldingHandPose, FoldingHandRightKinematics


REPO_ROOT = Path(__file__).resolve().parents[2]
HAND_ROOT = REPO_ROOT / "hand/folding_hand_right"
OUTPUT_ROOT = REPO_ROOT / "exp5/grasp_generation_study/results/folding_hand_right_model"


MESH_BY_BODY = {
    "base_link": "base_link.STL",
    "th_1": "th_1.STL",
    "th_2": "th_2.STL",
    "th_3": "th_3.STL",
    "th_4": "th_4.STL",
    "ff_1": "ff_1.STL",
    "ff_2": "ff_2.STL",
    "ff_3": "ff_3.STL",
    "mf_1": "mf_1.STL",
    "mf_2": "mf_2.STL",
    "rf_1": "rf_1.STL",
    "rf_2": "rf_2.STL",
    "rf_3": "rf_3.STL",
    "lf_1": "lf_1.STL",
    "lf_2": "lf_2.STL",
    "lf_3": "lf_3.STL",
}


def read_stl_triangles(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if raw[:5].lower() == b"solid" and b"facet normal" in raw[:512].lower():
        vertices = []
        for line in raw.decode("utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0] == "vertex":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.asarray(vertices, dtype=float).reshape(-1, 3, 3)

    count = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attr", "<u2"),
        ]
    )
    data = np.frombuffer(raw, dtype=dtype, count=count, offset=84)
    return np.asarray(data["vertices"], dtype=float)


def transformed_meshes(hand: FoldingHandRightKinematics, pose: FoldingHandPose, stride: int) -> list[np.ndarray]:
    transforms = hand.forward_kinematics(pose)
    meshes = []
    for body, mesh_name in MESH_BY_BODY.items():
        triangles = read_stl_triangles(HAND_ROOT / "meshes" / mesh_name)
        if stride > 1 and len(triangles) > stride:
            triangles = triangles[::stride]
        translation, rotation = transforms[body]
        meshes.append(triangles @ rotation.T + translation)
    return meshes


def equal_axes(ax, points: np.ndarray) -> None:
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    center = 0.5 * (pmin + pmax)
    radius = max(float(np.max(pmax - pmin)) / 2, 0.04)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def render_pose(name: str, pose: FoldingHandPose, output_path: Path, stride: int) -> None:
    hand = FoldingHandRightKinematics.default()
    meshes = transformed_meshes(hand, pose, stride)
    all_points = np.concatenate([mesh.reshape(-1, 3) for mesh in meshes], axis=0)

    fig = plt.figure(figsize=(7.2, 6.8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(meshes)))
    for mesh, color in zip(meshes, colors):
        collection = Poly3DCollection(mesh, linewidths=0.05, alpha=0.82)
        collection.set_facecolor(color)
        collection.set_edgecolor((0.15, 0.15, 0.15, 0.08))
        ax.add_collection3d(collection)

    contacts = hand.contact_points(pose)
    ax.scatter(contacts[:, 0], contacts[:, 1], contacts[:, 2], c="#d62728", s=28, depthshade=False)
    for idx, point in enumerate(contacts, start=1):
        ax.text(point[0], point[1], point[2], f"T{idx}", fontsize=8, weight="bold")

    equal_axes(ax, all_points)
    ax.view_init(elev=18, azim=-58)
    ax.set_title(f"folding_hand_right: {name}", fontsize=12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--stride", type=int, default=5, help="Triangle subsampling stride for faster rendering.")
    args = parser.parse_args()

    hand = FoldingHandRightKinematics.default()
    open_pose = FoldingHandPose(np.zeros(3), np.eye(3), hand.open_controls())
    closed_pose = FoldingHandPose(np.zeros(3), np.eye(3), hand.mid_closed_controls())
    render_pose("open_pose", open_pose, args.output_root / "folding_hand_right_open_pose.png", args.stride)
    render_pose("half_closed_pose", closed_pose, args.output_root / "folding_hand_right_half_closed_pose.png", args.stride)
    print(args.output_root / "folding_hand_right_open_pose.png")
    print(args.output_root / "folding_hand_right_half_closed_pose.png")


if __name__ == "__main__":
    main()
