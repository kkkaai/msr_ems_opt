#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


def parse_methods(methods_str: str) -> list[str]:
    methods = [m.strip().lower() for m in methods_str.split(",") if m.strip()]
    valid = {"grid", "pso", "cmaes"}
    for m in methods:
        if m not in valid:
            raise ValueError(f"Unsupported method '{m}'. Supported: {sorted(valid)}")
    return methods


def parse_seeds(seeds_str: str) -> set[int]:
    if not seeds_str.strip():
        return set()
    return {int(s.strip()) for s in seeds_str.split(",") if s.strip()}


def parse_seed_from_name(stem: str) -> int | None:
    # Expected pattern: <object>__seed<int>
    m = re.search(r"__seed(-?\d+)$", stem)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_point_modes(mode: str) -> list[str]:
    m = mode.strip().lower()
    if m in {"no_points", "without_points", "none"}:
        return ["no_points"]
    if m in {"with_points", "with"}:
        return ["with_points"]
    if m == "both":
        return ["no_points", "with_points"]
    raise ValueError("Unsupported --point-cloud-mode. Use one of: no_points, with_points, both.")


def maybe_subsample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def euler_zyx_to_rotm(euler_zyx: np.ndarray) -> np.ndarray:
    # Keep consistent with EMS.superquadrics: Rotation.from_euler("ZYX", euler)
    rz, ry, rx = float(euler_zyx[0]), float(euler_zyx[1]), float(euler_zyx[2])
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    return rz_m @ ry_m @ rx_m


def superquadric_surface(shape: np.ndarray, scale: np.ndarray, n_eta: int = 80, n_omega: int = 120) -> np.ndarray:
    e1 = float(shape[0])
    e2 = float(shape[1])
    a1 = float(scale[0])
    a2 = float(scale[1])
    a3 = float(scale[2])

    eta = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_eta, dtype=float)
    omg = np.linspace(-np.pi, np.pi, n_omega, dtype=float)
    eta_m, omg_m = np.meshgrid(eta, omg, indexing="xy")

    c_eta = np.sign(np.cos(eta_m)) * np.power(np.abs(np.cos(eta_m)), e1)
    s_eta = np.sign(np.sin(eta_m)) * np.power(np.abs(np.sin(eta_m)), e1)
    c_omg = np.sign(np.cos(omg_m)) * np.power(np.abs(np.cos(omg_m)), e2)
    s_omg = np.sign(np.sin(omg_m)) * np.power(np.abs(np.sin(omg_m)), e2)

    x = a1 * c_eta * c_omg
    y = a2 * c_eta * s_omg
    z = a3 * s_eta
    return np.stack([x, y, z], axis=0)


def set_axes_equal(ax, xyz: np.ndarray) -> None:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    mins = np.array([x.min(), y.min(), z.min()], dtype=float)
    maxs = np.array([x.max(), y.max(), z.max()], dtype=float)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius <= 0.0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_with_matplotlib(
    points_vis: np.ndarray,
    shape: np.ndarray,
    scale: np.ndarray,
    euler: np.ndarray,
    translation: np.ndarray,
    out_img: Path,
    with_points: bool,
    hide_axes: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surf_local = superquadric_surface(shape=shape, scale=scale, n_eta=80, n_omega=120)
    rotm = euler_zyx_to_rotm(euler)
    surf_flat = surf_local.reshape(3, -1)
    surf_world = (rotm @ surf_flat) + translation.reshape(3, 1)
    xw = surf_world[0].reshape(surf_local.shape[1], surf_local.shape[2])
    yw = surf_world[1].reshape(surf_local.shape[1], surf_local.shape[2])
    zw = surf_world[2].reshape(surf_local.shape[1], surf_local.shape[2])

    fig = plt.figure(figsize=(6.0, 6.0), dpi=180, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.plot_surface(xw, yw, zw, rstride=1, cstride=1, color="#4f77ff", alpha=0.55, linewidth=0.0)

    if with_points and points_vis.size > 0:
        ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], s=1.0, c="r", alpha=0.5)

    all_xyz = np.vstack([points_vis, surf_world.T]) if points_vis.size > 0 else surf_world.T
    set_axes_equal(ax, all_xyz)
    ax.view_init(elev=22.0, azim=38.0)
    if hide_axes:
        ax.set_axis_off()
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.tight_layout()
    out_img.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_img), facecolor="white", edgecolor="white", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild per-model E1 visualization images directly from saved JSON results."
    )
    parser.add_argument("--exp-root", type=str, default="exp1/results/e1_single")
    parser.add_argument("--output", type=str, default="exp1/results/e1_single/vis")
    parser.add_argument("--methods", type=str, default="grid,pso,cmaes")
    parser.add_argument("--seeds", type=str, default="", help="Optional filter, e.g. '0,1,2'. Empty means all.")
    parser.add_argument("--format", type=str, default="png", choices=["png", "eps"])
    parser.add_argument(
        "--visualizeMode",
        type=str,
        default="reconstruction_only",
        choices=["reconstruction_only", "with_points"],
        help="Legacy switch. Prefer --point-cloud-mode.",
    )
    parser.add_argument(
        "--point-cloud-mode",
        type=str,
        default="",
        help="Point cloud rendering mode: no_points, with_points, or both. Empty means infer from --visualizeMode.",
    )
    parser.add_argument(
        "--show-axes",
        action="store_true",
        help="Show axis frame and labels. Default is hidden (pure white background).",
    )
    parser.add_argument("--arcLength", type=float, default=0.2)
    parser.add_argument("--pointSize", type=float, default=0.001)
    parser.add_argument("--max-points", type=int, default=5000, help="0 means use all points.")
    parser.add_argument("--offscreen", action="store_true", help="Enable Mayavi offscreen rendering.")
    parser.add_argument(
        "--backend",
        type=str,
        default="matplotlib",
        choices=["auto", "matplotlib", "mayavi"],
        help="Rendering backend. 'auto' tries mayavi first, then falls back to matplotlib.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-files", type=int, default=0, help="For smoke tests. 0 means all.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from hierarchical_ems_pso_core import add_external_src_path, read_ply_xyz  # pylint: disable=import-error
    from pointcloud_normalization import normalize_points  # pylint: disable=import-error

    use_mayavi = False
    mayavi_modules = {}
    if args.backend in {"auto", "mayavi"}:
        try:
            from layer_visualization import layer_color  # pylint: disable=import-error
            from mayavi_export import save_mayavi_figure  # pylint: disable=import-error
            from mayavi import mlab  # pylint: disable=import-error

            add_external_src_path(repo_root)
            from EMS.superquadrics import superquadric  # pylint: disable=import-error
            from EMS.utilities import showPoints  # pylint: disable=import-error

            if args.offscreen:
                mlab.options.offscreen = True
            mayavi_modules = {
                "layer_color": layer_color,
                "save_mayavi_figure": save_mayavi_figure,
                "mlab": mlab,
                "superquadric": superquadric,
                "showPoints": showPoints,
            }
            use_mayavi = True
        except Exception as exc:
            if args.backend == "mayavi":
                raise RuntimeError(f"Requested --backend mayavi, but mayavi stack is unavailable: {exc}") from exc
            print(f"Mayavi unavailable, fallback to matplotlib backend: {exc}")

    exp_root = (repo_root / args.exp_root).resolve()
    out_root = (repo_root / args.output).resolve()
    methods = parse_methods(args.methods)
    seeds_filter = parse_seeds(args.seeds)
    if args.point_cloud_mode.strip():
        point_modes = parse_point_modes(args.point_cloud_mode)
    else:
        point_modes = ["with_points"] if args.visualizeMode == "with_points" else ["no_points"]

    if args.backend == "matplotlib":
        mpl_cfg = out_root / ".mplconfig"
        mpl_cfg.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))

    json_files: list[Path] = []
    for method in methods:
        method_dir = exp_root / "json" / method
        if not method_dir.exists():
            continue
        json_files.extend(sorted(method_dir.glob("*.json")))

    if args.max_files > 0:
        json_files = json_files[: args.max_files]

    total = len(json_files)
    if total == 0:
        print("No JSON files found.")
        return 0

    ok = 0
    fail = 0
    skip = 0

    for i, jp in enumerate(json_files, start=1):
        method = jp.parent.name
        stem = jp.stem
        seed = parse_seed_from_name(stem)
        if seeds_filter and (seed is None or seed not in seeds_filter):
            continue

        out_dir = out_root / method
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            sq = data.get("superquadric", {})
            if not isinstance(sq, dict):
                raise ValueError("missing superquadric dict")
            shape = np.asarray(sq["shape"], dtype=float).reshape(2)
            scale = np.asarray(sq["scale"], dtype=float).reshape(3)
            euler = np.asarray(sq["euler"], dtype=float).reshape(3)
            translation = np.asarray(sq["translation"], dtype=float).reshape(3)

            ply_path = Path(str(data.get("input_path", ""))).expanduser().resolve()
            if not ply_path.exists():
                raise FileNotFoundError(f"input_path not found: {ply_path}")

            points = np.asarray(read_ply_xyz(ply_path), dtype=float)
            norm_cfg = data.get("normalization", {})
            if (
                isinstance(norm_cfg, dict)
                and bool(norm_cfg.get("enabled", False))
                and not bool(norm_cfg.get("output_in_original_scale", True))
            ):
                norm_method = str(norm_cfg.get("method", "ems_matlab"))
                points, _ = normalize_points(points, method=norm_method)

            points_vis = maybe_subsample_points(points, args.max_points, seed=0 if seed is None else seed)
            for mode in point_modes:
                with_points = mode == "with_points"
                suffix = "_with_points" if with_points else "_no_points"
                out_img = out_dir / f"{stem}{suffix}.{args.format}" if len(point_modes) > 1 else out_dir / f"{stem}.{args.format}"

                if args.skip_existing and out_img.exists():
                    skip += 1
                    print(f"[{i:04d}/{total:04d}] SKIP {method}/{stem} ({mode})")
                    continue

                if use_mayavi:
                    mlab = mayavi_modules["mlab"]
                    layer_color = mayavi_modules["layer_color"]
                    save_mayavi_figure = mayavi_modules["save_mayavi_figure"]
                    superquadric = mayavi_modules["superquadric"]
                    showPoints = mayavi_modules["showPoints"]

                    sq_obj = superquadric(shape_vec=shape, scale_vec=scale, euler_vec=euler, translation=translation)
                    fig = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
                    sq_obj.showSuperquadric(arclength=args.arcLength, color=layer_color(0), opacity=0.5)
                    if with_points:
                        showPoints(points_vis, scale_factor=args.pointSize)
                    mlab.draw()
                    save_mayavi_figure(fig, out_img, mlab=mlab)
                    mlab.close(fig)
                else:
                    render_with_matplotlib(
                        points_vis=points_vis,
                        shape=shape,
                        scale=scale,
                        euler=euler,
                        translation=translation,
                        out_img=out_img,
                        with_points=with_points,
                        hide_axes=not args.show_axes,
                    )
                ok += 1
                print(f"[{i:04d}/{total:04d}] OK   {method}/{stem} ({mode}) -> {out_img}")

        except Exception as exc:
            fail += 1
            print(f"[{i:04d}/{total:04d}] FAIL {method}/{stem}: {exc}")

    print("\nSummary")
    print(f"- total scanned: {total}")
    print(f"- ok           : {ok}")
    print(f"- fail         : {fail}")
    print(f"- skip         : {skip}")
    print(f"- output       : {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
