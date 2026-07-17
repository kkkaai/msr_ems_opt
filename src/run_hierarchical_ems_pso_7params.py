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
    PsoConfig,
    RecoverHyperParams,
    add_external_src_path,
    read_ply_xyz,
    run_layerwise_pso,
    sq_to_dict,
    str2bool,
)


def _decode_particle_6params(x: np.ndarray, base_hp: RecoverHyperParams) -> RecoverHyperParams:
    return RecoverHyperParams(
        outlier_ratio=float(np.clip(x[0], 0.0, 0.999)),
        sigma=max(0.0, float(x[1])),
        max_layer=base_hp.max_layer,
        tau_in=float(np.clip(x[2], 0.0, 0.999)),
        tau_split=float(np.clip(x[3], 0.0, 0.999)),
        eps=max(1e-6, float(x[4])),
        min_points=max(2, int(round(x[5]))),
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
        description="Run layer-wise hierarchical multi-superquadric recovery with 7-parameter search "
                    "(OutlierRatio, Sigma, MaxLayer, TauIn, TauSplit, Eps, MinPoints)."
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

    # Fixed EMS settings
    parser.add_argument("--AdaptiveUpperBound", type=str2bool, default=True)
    parser.add_argument("--Rescale", type=str2bool, default=False)
    parser.add_argument("--MaxIterationEM", type=int, default=20)
    parser.add_argument("--ToleranceEM", type=float, default=1e-3)
    parser.add_argument("--RelativeToleranceEM", type=float, default=2e-1)
    parser.add_argument("--MaxOptiIterations", type=int, default=2)
    parser.add_argument("--MaxiSwitch", type=int, default=2)
    parser.add_argument("--MinClusterRatio", type=float, default=8e-4)

    # 7-parameter search bounds
    parser.add_argument("--OutlierRatioMin", type=float, default=0.1)
    parser.add_argument("--OutlierRatioMax", type=float, default=0.95)
    parser.add_argument("--SigmaMin", type=float, default=0.0)
    parser.add_argument("--SigmaMax", type=float, default=0.8)
    parser.add_argument(
        "--MaxLayerMode",
        type=str,
        choices=["search", "fixed"],
        default="search",
        help="MaxLayer handling mode: 'search' optimizes 7 params via MaxLayer enumeration; "
             "'fixed' keeps MaxLayer fixed and optimizes the other 6 params only.",
    )
    parser.add_argument(
        "--FixedMaxLayer",
        type=int,
        default=5,
        help="Used when --MaxLayerMode fixed.",
    )
    parser.add_argument("--MaxLayerMin", type=int, default=2)
    parser.add_argument("--MaxLayerMax", type=int, default=7)
    parser.add_argument("--TauInMin", type=float, default=0.05)
    parser.add_argument("--TauInMax", type=float, default=0.5)
    parser.add_argument("--TauSplitMin", type=float, default=0.5)
    parser.add_argument("--TauSplitMax", type=float, default=0.95)
    parser.add_argument("--EpsMin", type=float, default=1.0)
    parser.add_argument("--EpsMax", type=float, default=3.0)
    parser.add_argument("--MinPointsMin", type=float, default=10.0)
    parser.add_argument("--MinPointsMax", type=float, default=120.0)

    # PSO config
    parser.add_argument("--swarmsize", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--omega", type=float, default=0.72)
    parser.add_argument("--c1", type=float, default=1.49)
    parser.add_argument("--c2", type=float, default=1.49)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--fitnessThreshold", type=float, default=1e-4)
    parser.add_argument(
        "--FitnessMode",
        type=str,
        choices=["legacy", "distance_coverage", "distance_coverage_outlier_complexity"],
        default="legacy",
        help=(
            "Objective mode: "
            "legacy=distance only; "
            "distance_coverage=distance + lambda_cov*(1-coverage_ratio); "
            "distance_coverage_outlier_complexity adds residual and complexity penalties."
        ),
    )
    parser.add_argument(
        "--LambdaCov",
        type=float,
        default=0.0,
        help="Coverage penalty weight for non-legacy objectives.",
    )
    parser.add_argument(
        "--LambdaOut",
        type=float,
        default=0.1,
        help="Residual unexplained ratio penalty weight (used in distance_coverage_outlier_complexity).",
    )
    parser.add_argument(
        "--LambdaComp",
        type=float,
        default=0.01,
        help="Complexity penalty weight on number of superquadrics (used in distance_coverage_outlier_complexity).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str]) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.GlobalNormalize and args.Rescale:
        raise ValueError(
            "Scale conflict: GlobalNormalize=True and Rescale=True would cause dual rescaling. "
            "Use GlobalNormalize=True with Rescale=False for cross-dataset scale unification."
        )
    if args.LambdaCov < 0.0:
        raise ValueError("--LambdaCov must be >= 0.")
    if args.LambdaOut < 0.0:
        raise ValueError("--LambdaOut must be >= 0.")
    if args.LambdaComp < 0.0:
        raise ValueError("--LambdaComp must be >= 0.")

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

    if args.MaxLayerMode == "fixed":
        if args.FixedMaxLayer < 1:
            raise ValueError("FixedMaxLayer must be >= 1.")
        max_layer_candidates = [int(args.FixedMaxLayer)]
    else:
        if args.MaxLayerMin > args.MaxLayerMax:
            raise ValueError("MaxLayerMin must be <= MaxLayerMax.")
        max_layer_candidates = list(range(int(args.MaxLayerMin), int(args.MaxLayerMax) + 1))

    lb = np.array(
        [
            args.OutlierRatioMin,
            args.SigmaMin,
            args.TauInMin,
            args.TauSplitMin,
            args.EpsMin,
            args.MinPointsMin,
        ],
        dtype=float,
    )
    ub = np.array(
        [
            args.OutlierRatioMax,
            args.SigmaMax,
            args.TauInMax,
            args.TauSplitMax,
            args.EpsMax,
            args.MinPointsMax,
        ],
        dtype=float,
    )
    if np.any(lb >= ub):
        raise ValueError("All lower bounds must be strictly smaller than upper bounds.")

    pso_cfg = PsoConfig(
        swarmsize=args.swarmsize,
        maxiter=args.maxiter,
        omega=args.omega,
        c1=args.c1,
        c2=args.c2,
        tol=args.tol,
        patience=args.patience,
        fitness_threshold=args.fitnessThreshold,
        seed=args.seed,
    )

    search_records = []

    start = timeit.default_timer()
    best_record = None
    best_fitness = float("inf")

    for max_layer in max_layer_candidates:
        candidate_t_start = timeit.default_timer()
        base_hp = RecoverHyperParams(
            outlier_ratio=0.9,
            sigma=0.3,
            max_layer=max_layer,
            tau_in=0.1,
            tau_split=0.8,
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
        record = run_layerwise_pso(
            point_cloud=point_cloud,
            base_hp=base_hp,
            pso_cfg=pso_cfg,
            lb=lb,
            ub=ub,
            decode_particle=_decode_particle_6params,
            global_normalize=args.GlobalNormalize,
            global_norm_method=args.GlobalNormMethod,
            output_in_original_scale=args.OutputInOriginalScale,
            fitness_mode=args.FitnessMode,
            lambda_cov=args.LambdaCov,
            lambda_out=args.LambdaOut,
            lambda_comp=args.LambdaComp,
        )
        candidate_t_end = timeit.default_timer()
        record["candidate_runtime_ms"] = float((candidate_t_end - candidate_t_start) * 1000.0)
        record["max_layer"] = max_layer
        search_records.append(record)
        if record["fitness"] < best_fitness:
            best_fitness = record["fitness"]
            best_record = record

    stop = timeit.default_timer()
    if best_record is None:
        raise RuntimeError("Layer-wise PSO search failed to produce a valid result.")

    print(f"Best fitness: {best_record['fitness']:.6f}")
    print(f"Recovered {len(best_record['list_quadrics'])} superquadrics.")
    print(f"Best max_layer: {best_record['max_layer']}")
    print(f"MaxLayer mode: {args.MaxLayerMode}")
    total_runtime_ms = (stop - start) * 1000.0
    if args.runtime:
        print(f"Runtime: {total_runtime_ms:.3f} ms")
    print("----------------------------------------------------")

    if args.result:
        result_obj = {
            "best_max_layer": best_record["max_layer"],
            "fitness_details": best_record["fitness_details"],
            "layer_best_hyperparameters": best_record["layer_best_hparams"],
            "layer_best_fitness": best_record["layer_best_fitness"],
            "layer_best_distance_fit": best_record["layer_best_distance_fit"],
            "layer_best_coverage_ratio": best_record["layer_best_coverage_ratio"],
            "layer_best_residual_unexplained_ratio": best_record["layer_best_residual_unexplained_ratio"],
            "layer_best_num_superquadrics": best_record["layer_best_num_superquadrics"],
            "layer_timing": best_record["layer_timing"],
            "summary": best_record["summary"],
        }
        print(json.dumps(result_obj, indent=2))
        print("----------------------------------------------------")

    if args.out:
        output = {
            "mode": "layerwise_pso_7params" if args.MaxLayerMode == "search" else "layerwise_pso_6params_fixed_maxlayer",
            "max_layer_mode": args.MaxLayerMode,
            "input_path": str(input_path),
            "best_fitness": float(best_record["fitness"]),
            "fitness_details": best_record["fitness_details"],
            "best_max_layer": int(best_record["max_layer"]),
            "best_record": {
                "fitness_details": best_record["fitness_details"],
                "layer_best_hyperparameters": best_record["layer_best_hparams"],
                "layer_best_fitness": best_record["layer_best_fitness"],
                "layer_best_distance_fit": best_record["layer_best_distance_fit"],
                "layer_best_coverage_ratio": best_record["layer_best_coverage_ratio"],
                "layer_best_residual_unexplained_ratio": best_record["layer_best_residual_unexplained_ratio"],
                "layer_best_num_superquadrics": best_record["layer_best_num_superquadrics"],
                "layer_timing": best_record["layer_timing"],
                "layer_histories": best_record["layer_histories"],
                "summary": best_record["summary"],
                "superquadrics": [sq_to_dict(sq) for sq in best_record["list_quadrics"]],
                "point_seg_count_per_layer": {str(k): len(v) for k, v in best_record["point_seg"].items()},
                "point_inlier_count_per_layer": {str(k): len(v) for k, v in best_record["point_inlier"].items()},
                "point_outlier_count_per_layer": {str(k): len(v) for k, v in best_record["point_outlier"].items()},
            },
            "max_layer_search_records": [
                {
                    "max_layer": int(rec["max_layer"]),
                    "fitness": float(rec["fitness"]),
                    "candidate_runtime_ms": float(rec.get("candidate_runtime_ms", 0.0)),
                    "summary": rec["summary"],
                }
                for rec in search_records
            ],
            "pso_config": asdict(pso_cfg),
            "objective_config": {
                "fitness_mode": args.FitnessMode,
                "lambda_cov": float(args.LambdaCov),
                "lambda_out": float(args.LambdaOut),
                "lambda_comp": float(args.LambdaComp),
            },
            "timing": {
                "total_runtime_ms": float(total_runtime_ms),
                "best_candidate_runtime_ms": float(best_record.get("candidate_runtime_ms", 0.0)),
                "best_layer_timing": best_record["layer_timing"],
            },
            "normalization": best_record["normalization"],
        }
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Saved results to: {out_path}")

    if args.visualize:
        if os.environ.get("DISPLAY") is None and sys.platform != "win32" and os.environ.get("TERM_PROGRAM") is None:
            print("Visualization skipped: no GUI display detected in current session.")
            return
        try:
            from mayavi import mlab
            from EMS.utilities import showPoints

            fig = mlab.figure(size=(500, 500), bgcolor=(1, 1, 1))
            for layer, quadric in iter_layered_quadrics(
                list_quadrics=best_record["list_quadrics"],
                per_layer_groups=best_record["point_inlier"],
                max_layer=max(0, int(best_record["max_layer"]) - 1),
            ):
                quadric.showSuperquadric(
                    arclength=args.arcLength,
                    color=layer_color(layer),
                    opacity=0.5,
                )
            if args.visualizeMode == "with_points":
                vis_points = (
                    point_cloud if (not args.GlobalNormalize or args.OutputInOriginalScale)
                    else best_record["point_cloud_used"]
                )
                showPoints(vis_points, scale_factor=args.pointSize)
            mlab.show()
        except Exception as exc:
            print("Visualization skipped due to backend/display error:", exc)


if __name__ == "__main__":
    main(sys.argv[1:])
