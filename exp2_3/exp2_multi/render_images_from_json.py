#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import platform
import sys
import traceback
from pathlib import Path

from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hierarchical_ems_pso_core import add_external_src_path, read_ply_xyz, str2bool
from layer_visualization import layer_color
from mayavi_export import save_mayavi_figure


def _parse_methods(value: str) -> list[str]:
    methods = [m.strip().lower() for m in value.split(",") if m.strip()]
    if not methods:
        raise ValueError("Empty --methods.")
    return methods


def _layer_ids(num_sq: int, layer_counts: list[int]) -> list[int]:
    ids: list[int] = []
    for layer, c in enumerate(layer_counts):
        ids.extend([layer] * max(0, int(c)))
    if len(ids) < num_sq:
        ids.extend([len(layer_counts)] * (num_sq - len(ids)))
    return ids[:num_sq]


def _auto_point_size(points, fallback: float = 0.003) -> float:
    try:
        import numpy as np  # local import to keep module import lightweight

        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
            return fallback
        span = np.ptp(pts, axis=0)
        diag = float(np.linalg.norm(span))
        if not np.isfinite(diag) or diag <= 0.0:
            return fallback
        # Empirically stable for KIT object scale after current preprocessing.
        return max(0.002, min(0.03, 0.008 * diag))
    except Exception:
        return fallback


def _draw_points(mlab, points, point_size: float) -> None:
    """
    Draw point cloud in a robust way for screenshots.
    Using mode='point' with pixel-size control is much less likely to appear blank
    than tiny sphere glyphs in points3d default mode.
    """
    import numpy as np

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
        return

    cloud = mlab.points3d(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        mode="point",
        color=(1, 0, 0),
    )
    # points3d 'point' mode uses actor point_size in pixels.
    px = int(max(2, min(8, round(1000.0 * float(point_size)))))
    cloud.actor.property.point_size = px


def _render_one(
    json_path: Path,
    image_path: Path,
    visualize_mode: str,
    arc_length: float,
    point_size: float,
    offscreen: bool,
) -> tuple[bool, str]:
    # Ensure the external EMS package is importable even when this module is
    # called programmatically instead of through this script's CLI entrypoint.
    add_external_src_path(_REPO_ROOT)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    sq_list = data.get("superquadrics", [])
    if visualize_mode != "points_only" and not sq_list:
        return False, "no superquadrics in json"

    input_path = Path(data.get("input_path", ""))
    if not input_path.exists():
        return False, f"input_path missing: {input_path}"

    # Delay import until external EMS path is injected.
    from EMS.superquadrics import superquadric  # type: ignore
    from mayavi import mlab  # type: ignore

    layer_counts = [int(x) for x in data.get("layer_best_num_superquadrics", []) if int(x) >= 0]
    layer_ids = _layer_ids(len(sq_list), layer_counts)

    if offscreen:
        mlab.options.offscreen = True

    fig = mlab.figure(size=(900, 900), bgcolor=(1, 1, 1))
    try:
        if visualize_mode in {"reconstruction_only", "with_points"}:
            for i, sqd in enumerate(sq_list):
                sq = superquadric(
                    sqd["shape"],
                    sqd["scale"],
                    sqd["euler"],
                    sqd["translation"],
                )
                layer = layer_ids[i] if i < len(layer_ids) else 0
                sq.showSuperquadric(arclength=arc_length, color=layer_color(layer), opacity=0.5)

        if visualize_mode in {"with_points", "points_only"}:
            pts = read_ply_xyz(input_path)
            ps = float(point_size) if float(point_size) > 0 else _auto_point_size(pts)
            _draw_points(mlab, pts, ps)

        # Force camera to frame the rendered scene before export.
        mlab.view(azimuth=0.0, elevation=0.0, distance="auto")
        scene = getattr(fig, "scene", None)
        if scene is not None and getattr(scene, "camera", None) is not None:
            scene.reset_zoom()
            scene.render()

        save_mayavi_figure(fig, image_path, mlab=mlab)
    finally:
        mlab.close(fig)

    return True, ""


def _render_worker(
    json_path: str,
    image_path: str,
    visualize_mode: str,
    arc_length: float,
    point_size: float,
    offscreen: bool,
    queue: "mp.Queue[tuple[bool, str]]",
) -> None:
    try:
        res = _render_one(
            Path(json_path),
            Path(image_path),
            visualize_mode=visualize_mode,
            arc_length=arc_length,
            point_size=point_size,
            offscreen=offscreen,
        )
        queue.put(res)
    except Exception:
        queue.put((False, traceback.format_exc()))


def _render_one_isolated(
    json_path: Path,
    image_path: Path,
    visualize_mode: str,
    arc_length: float,
    point_size: float,
    offscreen: bool,
    timeout_sec: int,
) -> tuple[bool, str]:
    ctx = mp.get_context("spawn")
    queue: "mp.Queue[tuple[bool, str]]" = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_render_worker,
        args=(
            str(json_path),
            str(image_path),
            visualize_mode,
            arc_length,
            point_size,
            offscreen,
            queue,
        ),
    )
    proc.start()
    proc.join(timeout=timeout_sec if timeout_sec > 0 else None)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False, f"worker timeout after {timeout_sec}s"

    if proc.exitcode != 0:
        return False, f"worker crashed with exit code {proc.exitcode} (possible VTK/Mayavi segfault)"

    if queue.empty():
        return False, "worker exited without returning result"

    return queue.get()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render visualization images from existing E2 json results (without re-running optimization)."
    )
    parser.add_argument("--results-root", type=str, required=True, help="E2 result root, e.g. exp2_3/results/e2_multi_8x4_f005_c001")
    parser.add_argument("--methods", type=str, default="grid,pso,cmaes")
    parser.add_argument("--seed", type=int, default=0, help="Only render json files matching __seed{seed}.json")
    parser.add_argument("--image-format", type=str, default="png", choices=["png", "eps"])
    parser.add_argument(
        "--visualizeMode",
        type=str,
        default="with_points",
        choices=["reconstruction_only", "with_points", "points_only"],
    )
    parser.add_argument("--arcLength", type=float, default=0.2)
    parser.add_argument("--pointSize", type=float, default=0.001)
    parser.add_argument("--OffscreenRender", type=str2bool, default=False)
    parser.add_argument("--isolateProcess", type=str2bool, default=True, help="Render each image in a spawned child process.")
    parser.add_argument("--workerTimeoutSec", type=int, default=0, help="Optional per-image timeout in seconds; 0 means no timeout.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of images to render.")
    args = parser.parse_args()
    offscreen = bool(args.OffscreenRender)
    if offscreen and platform.system().lower() == "darwin":
        print("WARN: macOS VTK build usually does not support OSMesa offscreen rendering; forcing --OffscreenRender false.")
        offscreen = False

    add_external_src_path(_REPO_ROOT)

    root = (_REPO_ROOT / args.results_root).resolve()
    json_root = root / "json"
    vis_root = root / "vis"
    methods = _parse_methods(args.methods)

    all_jobs: list[tuple[Path, Path]] = []
    for m in methods:
        in_dir = json_root / m
        out_dir = vis_root / m
        out_dir.mkdir(parents=True, exist_ok=True)
        for js in sorted(in_dir.glob(f"*__seed{args.seed}.json")):
            out = out_dir / f"{js.stem}.{args.image_format}"
            if (not args.overwrite) and out.exists():
                continue
            all_jobs.append((js, out))

    if args.limit > 0:
        all_jobs = all_jobs[: args.limit]

    print(f"Render jobs: {len(all_jobs)}")
    ok = 0
    fail = 0
    for js, out in tqdm(all_jobs, desc="Render", unit="img", dynamic_ncols=True):
        if bool(args.isolateProcess):
            success, err = _render_one_isolated(
                js,
                out,
                visualize_mode=args.visualizeMode,
                arc_length=args.arcLength,
                point_size=args.pointSize,
                offscreen=offscreen,
                timeout_sec=int(args.workerTimeoutSec),
            )
        else:
            success, err = _render_one(
                js,
                out,
                visualize_mode=args.visualizeMode,
                arc_length=args.arcLength,
                point_size=args.pointSize,
                offscreen=offscreen,
            )
        if success:
            ok += 1
        else:
            fail += 1
            print(f"FAIL {js.name}: {err}")

    print("Done")
    print(f"- ok  : {ok}")
    print(f"- fail: {fail}")
    print(f"- out : {vis_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
