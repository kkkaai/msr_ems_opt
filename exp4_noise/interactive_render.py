#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_ems_pso_core import add_external_src_path, read_ply_xyz
from layer_visualization import layer_color
from mayavi_export import save_mayavi_figure


def layer_ids(num_sq: int, counts: list[int]) -> list[int]:
    ids = []
    for layer, count in enumerate(counts):
        ids.extend([layer] * max(0, int(count)))
    ids.extend([len(counts)] * max(0, num_sq - len(ids)))
    return ids[:num_sq]


def draw_points(mlab, points: np.ndarray, color: tuple[float, float, float], point_size: int) -> None:
    cloud = mlab.points3d(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        mode="point",
        color=color,
    )
    cloud.actor.property.point_size = int(point_size)


def save_camera(path: Path, mlab) -> None:
    azimuth, elevation, distance, focalpoint = mlab.view()
    payload = {
        "azimuth": float(azimuth),
        "elevation": float(elevation),
        "distance": float(distance),
        "focalpoint": [float(x) for x in focalpoint],
        "roll": float(mlab.roll()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_camera(path: Path, mlab) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    mlab.view(
        azimuth=float(data["azimuth"]),
        elevation=float(data["elevation"]),
        distance=float(data["distance"]),
        focalpoint=tuple(data["focalpoint"]),
    )
    mlab.roll(float(data.get("roll", 0.0)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactively select and save a Mayavi view.")
    parser.add_argument("--json", type=str, default="", help="Reconstruction JSON; required unless mode=points_only.")
    parser.add_argument("--points", type=str, required=True, help="Clean or noisy PLY displayed in the scene.")
    parser.add_argument(
        "--mode",
        choices=["points_only", "reconstruction_only", "with_points"],
        default="with_points",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--camera-in", type=str, default="")
    parser.add_argument("--camera-out", type=str, default="")
    parser.add_argument("--point-size", type=int, default=3)
    parser.add_argument("--arc-length", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()

    add_external_src_path(REPO)
    from EMS.superquadrics import superquadric
    from mayavi import mlab

    points_path = Path(args.points).resolve()
    points = read_ply_xyz(points_path)
    reconstruction = None
    if args.mode != "points_only":
        if not args.json:
            raise ValueError("--json is required for reconstruction modes.")
        reconstruction = json.loads(Path(args.json).resolve().read_text(encoding="utf-8"))

    fig = mlab.figure(size=(args.width, args.height), bgcolor=(1, 1, 1))
    if reconstruction is not None:
        quadrics = reconstruction.get("superquadrics", [])
        counts = [int(x) for x in reconstruction.get("layer_best_num_superquadrics", [])]
        ids = layer_ids(len(quadrics), counts)
        for index, item in enumerate(quadrics):
            sq = superquadric(item["shape"], item["scale"], item["euler"], item["translation"])
            sq.showSuperquadric(
                arclength=float(args.arc_length),
                color=layer_color(ids[index]),
                opacity=0.55,
            )
    if args.mode in {"points_only", "with_points"}:
        draw_points(mlab, points, color=(0.85, 0.1, 0.1), point_size=args.point_size)

    mlab.view(azimuth=0.0, elevation=0.0, distance="auto")
    fig.scene.reset_zoom()
    if args.camera_in:
        load_camera(Path(args.camera_in).resolve(), mlab)
    fig.scene.render()

    output = Path(args.output).resolve()
    camera_out = Path(args.camera_out).resolve() if args.camera_out else None

    def on_keypress(vtk_interactor, _event):
        key = vtk_interactor.GetKeySym().lower()
        if key == "s":
            output.parent.mkdir(parents=True, exist_ok=True)
            save_mayavi_figure(fig, output, mlab=mlab)
            if camera_out is not None:
                save_camera(camera_out, mlab)
            print(f"Saved image: {output}", flush=True)
            if camera_out is not None:
                print(f"Saved camera: {camera_out}", flush=True)
        elif key == "q":
            mlab.close(fig)

    fig.scene.interactor.add_observer("KeyPressEvent", on_keypress)
    print("Drag to rotate, scroll to zoom, press 's' to save, and press 'q' to close.", flush=True)
    mlab.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
