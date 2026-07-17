import argparse
import json
import os
import sys
import timeit
from pathlib import Path

import numpy as np

from hierarchical_ems_pso_core import add_external_src_path, read_ply_xyz, sq_to_dict, str2bool
from layer_visualization import layer_color
from pointcloud_normalization import denormalize_superquadrics_inplace, normalize_points


def _sq_to_x(sq) -> np.ndarray:
    return np.array(
        [
            sq.shape[0],
            sq.shape[1],
            sq.scale[0],
            sq.scale[1],
            sq.scale[2],
            sq.euler[0],
            sq.euler[1],
            sq.euler[2],
            sq.translation[0],
            sq.translation[1],
            sq.translation[2],
        ],
        dtype=float,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-superquadric recovery with CMA-ES tuning on (OutlierRatio, Sigma)."
    )
    parser.add_argument("path_to_data", help="Path to input *.ply point cloud.")
    parser.add_argument("--out", type=str, default="", help="Optional JSON output path.")
    parser.add_argument("--runtime", action="store_true", help="Print runtime in ms.")
    parser.add_argument("--result", action="store_true", help="Print best hyperparameters and summary.")
    parser.add_argument("--visualize", action="store_true", help="Visualize recovered superquadric.")
    parser.add_argument(
        "--visualizeMode",
        type=str,
        choices=["reconstruction_only", "with_points"],
        default="reconstruction_only",
        help="Visualization mode: only reconstructed superquadric, or superquadric with point cloud.",
    )
    parser.add_argument("--saveImage", type=str, default="", help="Optional output image path for visualization snapshot.")
    parser.add_argument("--arcLength", type=float, default=0.2, help="Arc length for superquadric rendering.")
    parser.add_argument("--pointSize", type=float, default=0.001, help="Point size for point-cloud rendering.")

    parser.add_argument("--GlobalNormalize", type=str2bool, default=False)
    parser.add_argument(
        "--GlobalNormMethod",
        type=str,
        default="ems_matlab",
        choices=["ems_matlab", "unit_box", "unit_sphere"],
    )
    parser.add_argument("--OutputInOriginalScale", type=str2bool, default=True)

    parser.add_argument("--MaxIterationEM", type=int, default=20)
    parser.add_argument("--ToleranceEM", type=float, default=1e-3)
    parser.add_argument("--RelativeToleranceEM", type=float, default=2e-1)
    parser.add_argument("--MaxOptiIterations", type=int, default=2)
    parser.add_argument("--MaxiSwitch", type=int, default=2)
    parser.add_argument("--AdaptiveUpperBound", type=str2bool, default=True)
    parser.add_argument("--Rescale", type=str2bool, default=False)
    parser.add_argument("--pThreshold", type=float, default=0.1, help="Inlier threshold on posterior p.")
    parser.add_argument(
        "--LambdaCov",
        type=float,
        default=0.1,
        help="Coverage penalty weight in fitness: F=d_fit+LambdaCov*(1-coverage_ratio).",
    )

    parser.add_argument("--OutlierRatioInit", type=float, default=0.9)
    parser.add_argument("--SigmaInit", type=float, default=0.3)
    parser.add_argument("--OutlierRatioMin", type=float, default=0.1)
    parser.add_argument("--OutlierRatioMax", type=float, default=0.95)
    parser.add_argument("--SigmaMin", type=float, default=0.0)
    parser.add_argument("--SigmaMax", type=float, default=0.8)

    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--cmaSigma", type=float, default=0.15, help="Initial CMA-ES step size.")
    parser.add_argument("--fitnessThreshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str]) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.GlobalNormalize and args.Rescale:
        raise ValueError(
            "Scale conflict: GlobalNormalize=True and Rescale=True would cause dual rescaling. "
            "Use GlobalNormalize=True with Rescale=False."
        )
    if args.LambdaCov < 0.0:
        raise ValueError("--LambdaCov must be >= 0.")

    try:
        import cma  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "CMA-ES requires the 'cma' package. Install with: pip install cma"
        ) from exc

    repo_root = Path(__file__).resolve().parents[1]
    add_external_src_path(repo_root)
    from EMS.EMS_recovery import EMS_recovery, Distance

    input_path = Path(args.path_to_data).expanduser().resolve()
    print("----------------------------------------------------")
    print(f"Loading point cloud from: {input_path} ...")
    point_cloud = np.asarray(read_ply_xyz(input_path), dtype=float)
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError(f"Input point cloud must have shape (N, 3), got {point_cloud.shape}")
    print(f"Point cloud loaded. N={point_cloud.shape[0]}")
    print("----------------------------------------------------")

    point_working = point_cloud
    norm_meta = None
    if args.GlobalNormalize:
        point_working, norm_meta = normalize_points(point_cloud, method=args.GlobalNormMethod)

    lb = np.array([args.OutlierRatioMin, args.SigmaMin], dtype=float)
    ub = np.array([args.OutlierRatioMax, args.SigmaMax], dtype=float)
    x0 = np.array([args.OutlierRatioInit, args.SigmaInit], dtype=float)
    x0 = np.clip(x0, lb, ub)

    def evaluate_candidate(x: np.ndarray):
        outlier_ratio = float(np.clip(x[0], lb[0], ub[0]))
        sigma = float(np.clip(x[1], lb[1], ub[1]))
        try:
            sq, p_raw = EMS_recovery(
                point_working,
                OutlierRatio=outlier_ratio,
                MaxIterationEM=args.MaxIterationEM,
                ToleranceEM=args.ToleranceEM,
                RelativeToleranceEM=args.RelativeToleranceEM,
                MaxOptiIterations=args.MaxOptiIterations,
                Sigma=sigma,
                MaxiSwitch=args.MaxiSwitch,
                AdaptiveUpperBound=args.AdaptiveUpperBound,
                Rescale=args.Rescale,
            )
        except Exception:
            return float("inf"), None, None, float("inf"), 0.0

        inlier_mask = p_raw > args.pThreshold
        inlier_points = point_working[inlier_mask, :]
        coverage_ratio = float(np.mean(inlier_mask))
        if inlier_points.shape[0] == 0:
            return float("inf"), sq, p_raw, float("inf"), coverage_ratio

        dist = Distance(inlier_points, _sq_to_x(sq))
        d_fit = float(np.mean(np.abs(dist)))
        if not np.isfinite(d_fit):
            return float("inf"), sq, p_raw, float("inf"), coverage_ratio
        fitness = float(d_fit + float(args.LambdaCov) * (1.0 - coverage_ratio))
        return fitness, sq, p_raw, d_fit, coverage_ratio

    def objective(x_list):
        x = np.asarray(x_list, dtype=float)
        fval, _, _, _, _ = evaluate_candidate(x)
        return float(fval)

    options = {
        "bounds": [lb.tolist(), ub.tolist()],
        "popsize": int(args.popsize),
        "seed": int(args.seed),
        "maxiter": int(args.maxiter),
        "verb_disp": 0,
        "verbose": -9,
    }

    history = []
    start = timeit.default_timer()
    es = cma.CMAEvolutionStrategy(x0.tolist(), float(args.cmaSigma), options)
    for _ in range(int(args.maxiter)):
        solutions = es.ask()
        values = [objective(s) for s in solutions]
        es.tell(solutions, values)
        history.append(float(es.best.f))
        if es.best.f <= args.fitnessThreshold:
            break
        if es.stop():
            break
    best_x = np.asarray(es.best.x, dtype=float)
    best_f, best_sq, best_p, best_d_fit, best_cov = evaluate_candidate(best_x)
    stop = timeit.default_timer()

    if best_sq is None or best_p is None:
        raise RuntimeError("CMA-ES finished but no valid superquadric was recovered.")

    if args.GlobalNormalize and args.OutputInOriginalScale and norm_meta is not None:
        denormalize_superquadrics_inplace([best_sq], norm_meta)

    best_outlier_ratio = float(np.clip(best_x[0], lb[0], ub[0]))
    best_sigma = float(np.clip(best_x[1], lb[1], ub[1]))
    inlier_count = int(np.sum(best_p > args.pThreshold))
    outlier_count = int(point_working.shape[0] - inlier_count)
    total_runtime_ms = (stop - start) * 1000.0

    print(f"Best fitness: {best_f:.6f}")
    if args.runtime:
        print(f"Runtime: {total_runtime_ms:.3f} ms")
    print("----------------------------------------------------")

    if args.result:
        print(f"Best OutlierRatio: {best_outlier_ratio:.6f}")
        print(f"Best Sigma: {best_sigma:.6f}")
        print(f"Inlier points: {inlier_count}")
        print(f"Outlier points: {outlier_count}")
        sq_dict = sq_to_dict(best_sq)
        print(
            f"shape={sq_dict['shape']} scale={sq_dict['scale']} "
            f"euler={sq_dict['euler']} translation={sq_dict['translation']}"
        )
        print("----------------------------------------------------")

    if args.out:
        output = {
            "mode": "single_cmaes_2params",
            "input_path": str(input_path),
            "best_fitness": float(best_f),
            "fitness_details": {
                "mode": "distance_coverage",
                "lambda_cov": float(args.LambdaCov),
                "distance_fit": float(best_d_fit),
                "coverage_ratio": float(best_cov),
                "fitness": float(best_f),
            },
            "best_hyperparameters": {
                "OutlierRatio": best_outlier_ratio,
                "Sigma": best_sigma,
            },
            "search_bounds": {
                "OutlierRatio": [float(args.OutlierRatioMin), float(args.OutlierRatioMax)],
                "Sigma": [float(args.SigmaMin), float(args.SigmaMax)],
            },
            "initial_values": {
                "OutlierRatioInit": float(args.OutlierRatioInit),
                "SigmaInit": float(args.SigmaInit),
            },
            "cmaes_config": {
                "popsize": int(args.popsize),
                "maxiter": int(args.maxiter),
                "cmaSigma": float(args.cmaSigma),
                "seed": int(args.seed),
            },
            "fixed_hyperparameters": {
                "MaxIterationEM": int(args.MaxIterationEM),
                "ToleranceEM": float(args.ToleranceEM),
                "RelativeToleranceEM": float(args.RelativeToleranceEM),
                "MaxOptiIterations": int(args.MaxOptiIterations),
                "MaxiSwitch": int(args.MaxiSwitch),
                "AdaptiveUpperBound": bool(args.AdaptiveUpperBound),
                "Rescale": bool(args.Rescale),
                "pThreshold": float(args.pThreshold),
                "LambdaCov": float(args.LambdaCov),
            },
            "summary": {
                "total_points": int(point_working.shape[0]),
                "inlier_points": inlier_count,
                "outlier_points": outlier_count,
                "inlier_ratio": float(inlier_count / max(1, int(point_working.shape[0]))),
            },
            "timing": {
                "total_runtime_ms": float(total_runtime_ms),
            },
            "superquadric": sq_to_dict(best_sq),
            "cmaes_history": [float(v) for v in history],
            "normalization": {
                "enabled": bool(args.GlobalNormalize),
                "method": args.GlobalNormMethod if args.GlobalNormalize else "none",
                "output_in_original_scale": bool(args.OutputInOriginalScale),
                "meta": None if norm_meta is None else norm_meta.to_jsonable(),
            },
        }
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Saved results to: {out_path}")

    if args.visualize or args.saveImage:
        if os.environ.get("DISPLAY") is None and sys.platform != "win32" and os.environ.get("TERM_PROGRAM") is None:
            print("Visualization skipped: no GUI display detected in current session.")
            return
        try:
            from mayavi import mlab
            from EMS.utilities import showPoints

            fig = mlab.figure(size=(500, 500), bgcolor=(1, 1, 1))
            best_sq.showSuperquadric(arclength=args.arcLength, color=layer_color(0), opacity=0.5)
            if args.visualizeMode == "with_points":
                vis_points = point_cloud if (not args.GlobalNormalize or args.OutputInOriginalScale) else point_working
                showPoints(vis_points, scale_factor=args.pointSize)
            if args.saveImage:
                image_path = Path(args.saveImage).expanduser().resolve()
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mlab.savefig(str(image_path))
                print(f"Saved visualization to: {image_path}")
            if args.visualize:
                mlab.show()
            else:
                mlab.close(fig)
        except Exception as exc:
            print("Visualization skipped due to backend/display error:", exc)


if __name__ == "__main__":
    main(sys.argv[1:])
