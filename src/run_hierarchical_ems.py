import argparse
import json
import os
import sys
import timeit
from pathlib import Path

import numpy as np
from layer_visualization import iter_layered_quadrics, layer_color
from pointcloud_normalization import denormalize_superquadrics_inplace, normalize_points


def _add_external_test_path(repo_root: Path) -> None:
    test_dir = repo_root / "external" / "EMS-superquadric_fitting" / "Python" / "tests"
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir))


def _str2bool(value: str) -> bool:
    value_lower = value.lower()
    if value_lower in {"true", "1", "yes", "y", "t"}:
        return True
    if value_lower in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}. Use True/False.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multi-superquadric recovery via hierarchical_ems on a PLY point cloud."
    )
    parser.add_argument("path_to_data", help="Path to input *.ply point cloud.")
    parser.add_argument("--out", type=str, default="", help="Optional path to save JSON results.")
    parser.add_argument("--visualize", action="store_true", help="Visualize input and recovered quadrics.")
    parser.add_argument(
        "--visualizeMode",
        type=str,
        choices=["reconstruction_only", "with_points"],
        default="reconstruction_only",
        help="Visualization mode: only reconstructed superquadrics, or superquadrics with point cloud.",
    )
    parser.add_argument("--runtime", action="store_true", help="Print runtime in ms.")
    parser.add_argument("--result", action="store_true", help="Print recovered quadric parameters.")
    parser.add_argument("--arcLength", type=float, default=0.2, help="Rendering arc length.")
    parser.add_argument("--pointSize", type=float, default=0.001, help="Point size for visualization.")
    parser.add_argument(
        "--GlobalNormalize",
        type=_str2bool,
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
        type=_str2bool,
        default=True,
        help="Map recovered superquadrics back to original scale when global normalization is enabled.",
    )

    # hierarchical_ems hyperparameters
    parser.add_argument("--OutlierRatio", type=float, default=0.9)
    parser.add_argument("--MaxIterationEM", type=int, default=20)
    parser.add_argument("--ToleranceEM", type=float, default=1e-3)
    parser.add_argument("--RelativeToleranceEM", type=float, default=2e-1)
    parser.add_argument("--MaxOptiIterations", type=int, default=2)
    parser.add_argument("--Sigma", type=float, default=0.3)
    parser.add_argument("--MaxiSwitch", type=int, default=2)
    parser.add_argument(
        "--AdaptiveUpperBound",
        type=_str2bool,
        default=True,
        help="Use adaptive upper bound in EMS (True/False).",
    )
    parser.add_argument(
        "--Rescale",
        type=_str2bool,
        default=False,
        help="Use point-cloud rescaling in EMS (True/False).",
    )
    parser.add_argument("--MaxLayer", type=int, default=5)
    parser.add_argument("--TauIn", "--tau_in", dest="TauIn", type=float, default=0.1)
    parser.add_argument("--TauSplit", "--tau_split", dest="TauSplit", type=float, default=0.8)
    parser.add_argument("--Eps", type=float, default=1.7)
    parser.add_argument("--MinPoints", "--MinimumClusterSize", dest="MinPoints", type=int, default=60)
    return parser


def _sq_to_dict(sq) -> dict:
    return {
        "shape": np.asarray(sq.shape).tolist(),
        "scale": np.asarray(sq.scale).tolist(),
        "euler": np.asarray(sq.euler).tolist(),
        "translation": np.asarray(sq.translation).tolist(),
    }


def main(argv: list[str]) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.GlobalNormalize and args.Rescale:
        raise ValueError(
            "Scale conflict: GlobalNormalize=True and Rescale=True would cause dual rescaling. "
            "Use GlobalNormalize=True with Rescale=False for cross-dataset scale unification."
        )

    repo_root = Path(__file__).resolve().parents[1]
    _add_external_test_path(repo_root)

    from EMS.utilities import read_ply, showPoints
    from multiquadric_test import hierarchical_ems

    input_path = Path(args.path_to_data).expanduser().resolve()
    print("----------------------------------------------------")
    print(f"Loading point cloud from: {input_path} ...")
    point = read_ply(str(input_path))
    print("Point cloud loaded.")
    print("----------------------------------------------------")

    point_working = point
    norm_meta = None
    if args.GlobalNormalize:
        point_working, norm_meta = normalize_points(point, method=args.GlobalNormMethod)

    start = timeit.default_timer()
    point_seg, point_outlier, list_quadrics = hierarchical_ems(
        point=point_working,
        OutlierRatio=args.OutlierRatio,
        MaxIterationEM=args.MaxIterationEM,
        ToleranceEM=args.ToleranceEM,
        RelativeToleranceEM=args.RelativeToleranceEM,
        MaxOptiIterations=args.MaxOptiIterations,
        Sigma=args.Sigma,
        MaxiSwitch=args.MaxiSwitch,
        AdaptiveUpperBound=args.AdaptiveUpperBound,
        Rescale=args.Rescale,
        MaxLayer=args.MaxLayer,
        TauIn=args.TauIn,
        TauSplit=args.TauSplit,
        Eps=args.Eps,
        MinPoints=args.MinPoints,
    )
    stop = timeit.default_timer()

    if args.GlobalNormalize and args.OutputInOriginalScale and norm_meta is not None:
        denormalize_superquadrics_inplace(list_quadrics, norm_meta)

    print(f"Recovered {len(list_quadrics)} superquadrics.")
    total_runtime_ms = (stop - start) * 1000.0
    if args.runtime:
        print(f"Runtime: {total_runtime_ms:.3f} ms")
    print("----------------------------------------------------")

    if args.result:
        for i, sq in enumerate(list_quadrics):
            sq_dict = _sq_to_dict(sq)
            print(f"[SQ {i}] shape={sq_dict['shape']} scale={sq_dict['scale']} "
                  f"euler={sq_dict['euler']} translation={sq_dict['translation']}")
        print("----------------------------------------------------")

    if args.out:
        output = {
            "input_path": str(input_path),
            "num_superquadrics": len(list_quadrics),
            "hyperparameters": {
                "OutlierRatio": args.OutlierRatio,
                "MaxIterationEM": args.MaxIterationEM,
                "ToleranceEM": args.ToleranceEM,
                "RelativeToleranceEM": args.RelativeToleranceEM,
                "MaxOptiIterations": args.MaxOptiIterations,
                "Sigma": args.Sigma,
                "MaxiSwitch": args.MaxiSwitch,
                "AdaptiveUpperBound": args.AdaptiveUpperBound,
                "Rescale": args.Rescale,
                "MaxLayer": args.MaxLayer,
                "TauIn": args.TauIn,
                "TauSplit": args.TauSplit,
                "Eps": args.Eps,
                "MinPoints": args.MinPoints,
            },
            "normalization": {
                "enabled": args.GlobalNormalize,
                "method": args.GlobalNormMethod if args.GlobalNormalize else "none",
                "output_in_original_scale": args.OutputInOriginalScale,
                "meta": None if norm_meta is None else norm_meta.to_jsonable(),
            },
            "superquadrics": [_sq_to_dict(sq) for sq in list_quadrics],
            "point_seg_count_per_layer": {str(k): len(v) for k, v in point_seg.items()},
            "point_outlier_count_per_layer": {str(k): len(v) for k, v in point_outlier.items()},
            "timing": {
                "total_runtime_ms": float(total_runtime_ms),
            },
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

            fig = mlab.figure(size=(500, 500), bgcolor=(1, 1, 1))
            for layer, quadric in iter_layered_quadrics(
                list_quadrics=list_quadrics,
                per_layer_groups=point_seg,
                max_layer=max(0, args.MaxLayer - 1),
            ):
                quadric.showSuperquadric(
                    arclength=args.arcLength,
                    color=layer_color(layer),
                    opacity=0.5,
                )
            if args.visualizeMode == "with_points":
                vis_points = point if (not args.GlobalNormalize or args.OutputInOriginalScale) else point_working
                showPoints(vis_points, scale_factor=args.pointSize)
            mlab.show()
        except Exception as exc:
            print("Visualization skipped due to backend/display error:", exc)


if __name__ == "__main__":
    main(sys.argv[1:])
