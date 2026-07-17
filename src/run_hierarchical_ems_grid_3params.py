import argparse
import json
import os
import sys
import timeit
from dataclasses import asdict
from pathlib import Path

import numpy as np

from layer_visualization import iter_layered_quadrics, layer_color
from hierarchical_ems_pso_core import (
    GridSearchConfig,
    RecoverHyperParams,
    add_external_src_path,
    read_ply_xyz,
    run_layerwise_gridsearch,
    sq_to_dict,
    str2bool,
)


def _decode_particle_3params(x: np.ndarray, base_hp: RecoverHyperParams) -> RecoverHyperParams:
    return RecoverHyperParams(
        outlier_ratio=float(np.clip(x[0], 0.0, 0.999)),
        sigma=base_hp.sigma,
        max_layer=base_hp.max_layer,
        tau_in=base_hp.tau_in,
        tau_split=base_hp.tau_split,
        eps=max(1e-6, float(x[1])),
        min_points=max(2, int(round(x[2]))),
        adaptive_upper_bound=base_hp.adaptive_upper_bound,
        rescale=base_hp.rescale,
        max_iteration_em=base_hp.max_iteration_em,
        tolerance_em=base_hp.tolerance_em,
        relative_tolerance_em=base_hp.relative_tolerance_em,
        max_opti_iterations=base_hp.max_opti_iterations,
        maxi_switch=base_hp.maxi_switch,
        min_cluster_ratio=base_hp.min_cluster_ratio,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run layer-wise hierarchical multi-superquadric recovery with manual grid-search baseline "
                    "(OutlierRatio, Eps, MinPoints)."
    )
    parser.add_argument("path_to_data", help="Path to input *.ply point cloud.")
    parser.add_argument("--out", type=str, default="", help="Optional JSON output path.")
    parser.add_argument("--runtime", action="store_true", help="Print runtime in ms.")
    parser.add_argument("--result", action="store_true", help="Print best hyperparameters and summary.")
    parser.add_argument("--visualize", action="store_true", help="Visualize recovered quadrics and input points.")
    parser.add_argument(
        "--visualizeMode",
        type=str,
        choices=["reconstruction_only", "with_points"],
        default="reconstruction_only",
        help="Visualization mode: only reconstructed superquadrics, or superquadrics with point cloud.",
    )
    parser.add_argument("--saveImage", type=str, default="", help="Optional output image path for visualization snapshot.")
    parser.add_argument("--arcLength", type=float, default=0.2, help="Arc length for quadric rendering.")
    parser.add_argument("--pointSize", type=float, default=0.001, help="Point size for point-cloud rendering.")
    parser.add_argument(
        "--GlobalNormalize",
        type=str2bool,
        default=False,
        help="Apply global external normalization before hierarchical recovery (True/False).",
    )
    parser.add_argument(
        "--GlobalNormMethod",
        type=str,
        default="ems_matlab",
        choices=["ems_matlab", "unit_box", "unit_sphere"],
        help="Global normalization method.",
    )
    parser.add_argument(
        "--OutputInOriginalScale",
        type=str2bool,
        default=True,
        help="Map recovered superquadrics back to original scale when global normalization is enabled.",
    )

    # Fixed recovery settings
    parser.add_argument("--AdaptiveUpperBound", type=str2bool, default=True)
    parser.add_argument("--Rescale", type=str2bool, default=False)
    parser.add_argument("--MaxIterationEM", type=int, default=20)
    parser.add_argument("--ToleranceEM", type=float, default=1e-3)
    parser.add_argument("--RelativeToleranceEM", type=float, default=2e-1)
    parser.add_argument("--MaxOptiIterations", type=int, default=2)
    parser.add_argument("--MaxiSwitch", type=int, default=2)
    parser.add_argument("--Sigma", type=float, default=0.3)
    parser.add_argument("--MaxLayer", type=int, default=5)
    parser.add_argument("--TauIn", type=float, default=0.1)
    parser.add_argument("--TauSplit", type=float, default=0.8)
    parser.add_argument("--MinClusterRatio", type=float, default=8e-4)

    # 3-parameter search bounds
    parser.add_argument("--OutlierRatioMin", type=float, default=0.1)
    parser.add_argument("--OutlierRatioMax", type=float, default=0.95)
    parser.add_argument("--EpsMin", type=float, default=1.0)
    parser.add_argument("--EpsMax", type=float, default=3.0)
    parser.add_argument("--MinPointsMin", type=float, default=10.0)
    parser.add_argument("--MinPointsMax", type=float, default=120.0)
    parser.add_argument(
        "--gridSteps",
        type=int,
        default=10,
        help="Uniform divisions per dimension in manual grid search. 10 => 10*10*10 candidates per layer.",
    )
    return parser


def _fitness_to_accuracy_proxy(fitness: float) -> float:
    if not np.isfinite(fitness):
        return 0.0
    return float(1.0 / (1.0 + max(0.0, fitness)))


def main(argv: list[str]) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.GlobalNormalize and args.Rescale:
        raise ValueError(
            "Scale conflict: GlobalNormalize=True and Rescale=True would cause dual rescaling. "
            "Use GlobalNormalize=True with Rescale=False for cross-dataset scale unification."
        )

    repo_root = Path(__file__).resolve().parents[1]
    add_external_src_path(repo_root)

    input_path = Path(args.path_to_data).expanduser().resolve()
    print("----------------------------------------------------")
    print(f"Loading point cloud from: {input_path} ...")
    point_cloud = np.asarray(read_ply_xyz(input_path), dtype=float)
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError(f"Input point cloud must have shape (N, 3), got {point_cloud.shape}")
    print(f"Point cloud loaded. N={point_cloud.shape[0]}")
    print("----------------------------------------------------")

    lb = np.array([args.OutlierRatioMin, args.EpsMin, args.MinPointsMin], dtype=float)
    ub = np.array([args.OutlierRatioMax, args.EpsMax, args.MinPointsMax], dtype=float)
    if np.any(lb >= ub):
        raise ValueError("All lower bounds must be strictly smaller than upper bounds.")
    if args.gridSteps < 2:
        raise ValueError("--gridSteps must be >= 2")

    base_hp = RecoverHyperParams(
        outlier_ratio=0.9,
        sigma=args.Sigma,
        max_layer=args.MaxLayer,
        tau_in=args.TauIn,
        tau_split=args.TauSplit,
        eps=1.7,
        min_points=60,
        adaptive_upper_bound=args.AdaptiveUpperBound,
        rescale=args.Rescale,
        max_iteration_em=args.MaxIterationEM,
        tolerance_em=args.ToleranceEM,
        relative_tolerance_em=args.RelativeToleranceEM,
        max_opti_iterations=args.MaxOptiIterations,
        maxi_switch=args.MaxiSwitch,
        min_cluster_ratio=args.MinClusterRatio,
    )
    grid_cfg = GridSearchConfig(steps_per_dim=args.gridSteps)

    start = timeit.default_timer()
    result = run_layerwise_gridsearch(
        point_cloud=point_cloud,
        base_hp=base_hp,
        lb=lb,
        ub=ub,
        decode_particle=_decode_particle_3params,
        grid_cfg=grid_cfg,
        global_normalize=args.GlobalNormalize,
        global_norm_method=args.GlobalNormMethod,
        output_in_original_scale=args.OutputInOriginalScale,
    )
    stop = timeit.default_timer()

    final_fitness = float(result["fitness"])
    final_accuracy_proxy = _fitness_to_accuracy_proxy(final_fitness)
    print(f"Best fitness: {final_fitness:.6f}")
    print(f"Accuracy proxy (1/(1+fitness)): {final_accuracy_proxy:.6f}")
    print(f"Recovered {len(result['list_quadrics'])} superquadrics.")
    total_runtime_ms = (stop - start) * 1000.0
    if args.runtime:
        print(f"Runtime: {total_runtime_ms:.3f} ms")
    print("----------------------------------------------------")

    if args.result:
        payload = {
            "layer_best_hyperparameters": result["layer_best_hparams"],
            "layer_best_fitness": result["layer_best_fitness"],
            "layer_timing": result["layer_timing"],
            "summary": result["summary"],
            "final_accuracy_proxy": final_accuracy_proxy,
        }
        print(json.dumps(payload, indent=2))
        print("----------------------------------------------------")

    if args.out:
        output = {
            "mode": "layerwise_grid_3params_baseline",
            "input_path": str(input_path),
            "best_fitness": final_fitness,
            "accuracy_proxy": final_accuracy_proxy,
            "layer_best_hyperparameters": result["layer_best_hparams"],
            "layer_best_fitness": result["layer_best_fitness"],
            "layer_timing": result["layer_timing"],
            "layer_histories": result["layer_histories"],
            "summary": result["summary"],
            "superquadrics": [sq_to_dict(sq) for sq in result["list_quadrics"]],
            "point_seg_count_per_layer": {str(k): len(v) for k, v in result["point_seg"].items()},
            "point_inlier_count_per_layer": {str(k): len(v) for k, v in result["point_inlier"].items()},
            "point_outlier_count_per_layer": {str(k): len(v) for k, v in result["point_outlier"].items()},
            "grid_config": asdict(grid_cfg),
            "search_space": {
                "OutlierRatioMin": args.OutlierRatioMin,
                "OutlierRatioMax": args.OutlierRatioMax,
                "EpsMin": args.EpsMin,
                "EpsMax": args.EpsMax,
                "MinPointsMin": args.MinPointsMin,
                "MinPointsMax": args.MinPointsMax,
            },
            "timing": {
                "total_runtime_ms": float(total_runtime_ms),
                "layer_timing": result["layer_timing"],
            },
            "normalization": result["normalization"],
            "fixed_hyperparameters": {
                "sigma": base_hp.sigma,
                "max_layer": base_hp.max_layer,
                "tau_in": base_hp.tau_in,
                "tau_split": base_hp.tau_split,
                "adaptive_upper_bound": base_hp.adaptive_upper_bound,
                "rescale": base_hp.rescale,
                "max_iteration_em": base_hp.max_iteration_em,
                "tolerance_em": base_hp.tolerance_em,
                "relative_tolerance_em": base_hp.relative_tolerance_em,
                "max_opti_iterations": base_hp.max_opti_iterations,
                "maxi_switch": base_hp.maxi_switch,
                "min_cluster_ratio": base_hp.min_cluster_ratio,
            },
            "quality_metric": {
                "fitness": {
                    "name": "mean_abs_distance_on_inliers",
                    "lower_is_better": True,
                    "value": final_fitness,
                },
                "accuracy_proxy": {
                    "name": "1/(1+fitness)",
                    "higher_is_better": True,
                    "value": final_accuracy_proxy,
                },
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
            for layer, quadric in iter_layered_quadrics(
                list_quadrics=result["list_quadrics"],
                per_layer_groups=result["point_inlier"],
                max_layer=max(0, base_hp.max_layer - 1),
            ):
                quadric.showSuperquadric(
                    arclength=args.arcLength,
                    color=layer_color(layer),
                    opacity=0.5,
                )
            if args.visualizeMode == "with_points":
                vis_points = (
                    point_cloud if (not args.GlobalNormalize or args.OutputInOriginalScale)
                    else result["point_cloud_used"]
                )
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
