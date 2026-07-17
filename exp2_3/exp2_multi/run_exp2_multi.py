#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

SCRIPT_BY_METHOD = {
    "grid": "run_hierarchical_ems_grid_4params.py",
    "pso": "run_hierarchical_ems_pso_4params.py",
    "cmaes": "run_hierarchical_ems_cmaes_4params.py",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_methods(value: str) -> list[str]:
    methods = [m.strip().lower() for m in value.split(",") if m.strip()]
    for m in methods:
        if m not in SCRIPT_BY_METHOD:
            raise ValueError(f"Unsupported method '{m}'. Supported: {sorted(SCRIPT_BY_METHOD.keys())}")
    return methods


def parse_seeds(value: str) -> list[int]:
    seeds = [int(s.strip()) for s in value.split(",") if s.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def median_or_nan(values: list[float]) -> float:
    vals = [v for v in values if v == v and v not in (float("inf"), float("-inf"))]
    if not vals:
        return float("nan")
    return float(statistics.median(vals))


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v == v and v not in (float("inf"), float("-inf"))]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def build_ok_row_from_json(base: dict, out_json: Path) -> dict:
    data = json.loads(out_json.read_text(encoding="utf-8"))
    fd = data.get("fitness_details", {})
    sm = data.get("summary", {})
    tm = data.get("timing", {})
    return {
        **base,
        "status": "OK",
        "best_fitness": safe_float(data.get("best_fitness")),
        "distance_fit": safe_float(fd.get("distance_fit")),
        "coverage_ratio": safe_float(fd.get("coverage_ratio"), safe_float(sm.get("coverage_ratio"))),
        "residual_unexplained_ratio": safe_float(
            fd.get("residual_unexplained_ratio"), safe_float(sm.get("residual_unexplained_ratio"))
        ),
        "num_superquadrics": safe_float(fd.get("num_superquadrics"), safe_float(sm.get("num_superquadrics"))),
        "runtime_ms": safe_float(tm.get("total_runtime_ms")),
        "image": base.get("image", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="E2 multi-superquadric benchmark (grid/pso/cmaes, 4 params).")
    parser.add_argument("--labels", type=str, default="data/kit_superquadric_labels.csv")
    parser.add_argument("--ply-root", type=str, default="data/KIT_ObjectModels_25k_ply")
    parser.add_argument("--output", type=str, default="exp2_3/results/e2_multi")
    parser.add_argument("--methods", type=str, default="grid,pso,cmaes")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save visualization image for each run (or per selected seed).",
    )
    parser.add_argument(
        "--image-seed",
        type=int,
        default=0,
        help="Only save images for this seed (grid always uses first seed entry).",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=["png", "eps"],
        help="Output image format. 'eps' uses direct GL2PS vector export in runner scripts.",
    )
    parser.add_argument(
        "--visualizeMode",
        type=str,
        default="with_points",
        choices=["reconstruction_only", "with_points"],
        help="Visualization mode passed to runner when --save-images is enabled.",
    )
    parser.add_argument(
        "--OffscreenRender",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Whether to enable offscreen rendering when saving images. Default False.",
    )

    parser.add_argument("--GlobalNormalize", type=str, default="True")
    parser.add_argument("--GlobalNormMethod", type=str, default="ems_matlab")
    parser.add_argument("--OutputInOriginalScale", type=str, default="True")
    parser.add_argument("--Rescale", type=str, default="False")
    parser.add_argument("--AdaptiveUpperBound", type=str, default="True")

    parser.add_argument("--MaxLayer", type=int, default=5)
    parser.add_argument("--TauIn", type=float, default=0.1)
    parser.add_argument("--TauSplit", type=float, default=0.8)
    parser.add_argument("--MinClusterRatio", type=float, default=8e-4)

    parser.add_argument("--MaxIterationEM", type=int, default=20)
    parser.add_argument("--ToleranceEM", type=float, default=1e-3)
    parser.add_argument("--RelativeToleranceEM", type=float, default=2e-1)
    parser.add_argument("--MaxOptiIterations", type=int, default=2)
    parser.add_argument("--MaxiSwitch", type=int, default=2)

    parser.add_argument(
        "--FitnessMode",
        type=str,
        choices=["legacy", "distance_coverage", "distance_coverage_outlier_complexity"],
        default="distance_coverage_outlier_complexity",
    )
    parser.add_argument("--LambdaCov", type=float, default=0.03)
    parser.add_argument("--LambdaOut", type=float, default=0.03)
    parser.add_argument("--LambdaComp", type=float, default=0.03)

    # 4-parameter search bounds
    parser.add_argument("--OutlierRatioMin", type=float, default=0.1)
    parser.add_argument("--OutlierRatioMax", type=float, default=0.95)
    parser.add_argument("--SigmaMin", type=float, default=0.0)
    parser.add_argument("--SigmaMax", type=float, default=0.8)
    parser.add_argument("--EpsMin", type=float, default=1.0)
    parser.add_argument("--EpsMax", type=float, default=3.0)
    parser.add_argument("--MinPointsMin", type=float, default=10.0)
    parser.add_argument("--MinPointsMax", type=float, default=120.0)

    # method-specific budgets
    parser.add_argument("--grid-steps", type=int, default=5)

    parser.add_argument("--pso-swarmsize", type=int, default=20)
    parser.add_argument("--pso-maxiter", type=int, default=30)
    parser.add_argument("--pso-omega", type=float, default=0.72)
    parser.add_argument("--pso-c1", type=float, default=1.49)
    parser.add_argument("--pso-c2", type=float, default=1.49)
    parser.add_argument("--pso-tol", type=float, default=1e-6)
    parser.add_argument("--pso-patience", type=int, default=5)
    parser.add_argument("--pso-fitness-threshold", type=float, default=1e-4)

    parser.add_argument("--cma-popsize", type=int, default=20)
    parser.add_argument("--cma-maxiter", type=int, default=30)
    parser.add_argument("--cma-sigma", type=float, default=0.15)
    parser.add_argument("--cma-tol", type=float, default=1e-6)
    parser.add_argument("--cma-patience", type=int, default=5)
    parser.add_argument("--cma-fitness-threshold", type=float, default=1e-4)

    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    labels_path = (repo_root / args.labels).resolve()
    ply_root = (repo_root / args.ply_root).resolve()
    out_root = (repo_root / args.output).resolve()

    methods = parse_methods(args.methods)
    seeds = parse_seeds(args.seeds)

    label_rows = [r for r in read_csv(labels_path) if r.get("label", "").strip().lower() == "multi"]
    if args.max_objects > 0:
        label_rows = label_rows[: args.max_objects]

    json_dir = out_root / "json"
    log_dir = out_root / "log"
    vis_dir = out_root / "vis"
    for m in methods:
        (json_dir / m).mkdir(parents=True, exist_ok=True)
        (log_dir / m).mkdir(parents=True, exist_ok=True)
        if args.save_images:
            (vis_dir / m).mkdir(parents=True, exist_ok=True)

    runs_per_object = sum((1 if m == "grid" else len(seeds)) for m in methods)
    total_runs = len(label_rows) * runs_per_object

    print(f"Multi objects to evaluate: {len(label_rows)}")
    print(f"Methods: {methods}")
    print(f"Planned runs: {total_runs}")

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total_runs, desc="E2 runs", unit="run", dynamic_ncols=True)

    run_rows: list[dict] = []
    completed = 0

    for idx, row in enumerate(label_rows, start=1):
        rel = row["path"].strip()
        obj_name = Path(rel).stem
        ply_path = ply_root / rel

        if not ply_path.exists():
            for m in methods:
                method_seeds = [seeds[0]] if m == "grid" else seeds
                for seed in method_seeds:
                    completed += 1
                    run_rows.append(
                        {
                            "object": obj_name,
                            "method": m,
                            "seed": int(seed),
                            "status": "FAIL",
                            "reason": f"missing_ply:{ply_path}",
                        }
                    )
                    print(f"[run {completed:04d}/{total_runs:04d}] FAIL {obj_name} | {m} | seed={seed}: missing ply", flush=True)
                    if pbar is not None:
                        pbar.update(1)
            continue

        for m in methods:
            method_seeds = [seeds[0]] if m == "grid" else seeds
            for seed in method_seeds:
                planned = completed + 1
                out_json = json_dir / m / f"{obj_name}__seed{seed}.json"
                out_log = log_dir / m / f"{obj_name}__seed{seed}.log"
                image_path = vis_dir / m / f"{obj_name}__seed{seed}.{args.image_format}"
                need_image = bool(args.save_images and int(seed) == int(args.image_seed))
                base = {
                    "object": obj_name,
                    "method": m,
                    "seed": int(seed),
                    "json": str(out_json),
                    "log": str(out_log),
                    "image": str(image_path) if need_image else "",
                }

                if args.skip_existing and out_json.exists() and ((not need_image) or image_path.exists()):
                    completed += 1
                    try:
                        run_rows.append(build_ok_row_from_json(base=base, out_json=out_json))
                        print(
                            f"[run {completed:04d}/{total_runs:04d}] REUSE {obj_name} | {m} | seed={seed}",
                            flush=True,
                        )
                    except Exception as exc:
                        run_rows.append({**base, "status": "FAIL", "reason": f"existing_json_parse:{exc}"})
                        print(
                            f"[run {completed:04d}/{total_runs:04d}] FAIL  {obj_name} | {m} | seed={seed} "
                            f"(existing json parse)",
                            flush=True,
                        )
                    if pbar is not None:
                        pbar.update(1)
                    continue

                runner = repo_root / "src" / SCRIPT_BY_METHOD[m]
                print(f"[run {planned:04d}/{total_runs:04d}] START {obj_name} | {m} | seed={seed}", flush=True)
                if pbar is not None:
                    pbar.set_postfix_str(f"{obj_name} | {m} | seed={seed}")

                cmd = [
                    sys.executable,
                    str(runner),
                    str(ply_path),
                    "--out",
                    str(out_json),
                    "--runtime",
                    "--GlobalNormalize",
                    str(args.GlobalNormalize),
                    "--GlobalNormMethod",
                    str(args.GlobalNormMethod),
                    "--OutputInOriginalScale",
                    str(args.OutputInOriginalScale),
                    "--Rescale",
                    str(args.Rescale),
                    "--AdaptiveUpperBound",
                    str(args.AdaptiveUpperBound),
                    "--MaxLayer",
                    str(args.MaxLayer),
                    "--TauIn",
                    str(args.TauIn),
                    "--TauSplit",
                    str(args.TauSplit),
                    "--MinClusterRatio",
                    str(args.MinClusterRatio),
                    "--MaxIterationEM",
                    str(args.MaxIterationEM),
                    "--ToleranceEM",
                    str(args.ToleranceEM),
                    "--RelativeToleranceEM",
                    str(args.RelativeToleranceEM),
                    "--MaxOptiIterations",
                    str(args.MaxOptiIterations),
                    "--MaxiSwitch",
                    str(args.MaxiSwitch),
                    "--FitnessMode",
                    str(args.FitnessMode),
                    "--LambdaCov",
                    str(args.LambdaCov),
                    "--LambdaOut",
                    str(args.LambdaOut),
                    "--LambdaComp",
                    str(args.LambdaComp),
                    "--OutlierRatioMin",
                    str(args.OutlierRatioMin),
                    "--OutlierRatioMax",
                    str(args.OutlierRatioMax),
                    "--SigmaMin",
                    str(args.SigmaMin),
                    "--SigmaMax",
                    str(args.SigmaMax),
                    "--EpsMin",
                    str(args.EpsMin),
                    "--EpsMax",
                    str(args.EpsMax),
                    "--MinPointsMin",
                    str(args.MinPointsMin),
                    "--MinPointsMax",
                    str(args.MinPointsMax),
                ]
                if need_image:
                    cmd.extend(
                        [
                            "--saveImage",
                            str(image_path),
                            "--OffscreenRender",
                            str(args.OffscreenRender),
                            "--visualizeMode",
                            str(args.visualizeMode),
                        ]
                    )

                if m == "grid":
                    cmd.extend(["--gridSteps", str(args.grid_steps)])
                elif m == "pso":
                    cmd.extend(
                        [
                            "--swarmsize",
                            str(args.pso_swarmsize),
                            "--maxiter",
                            str(args.pso_maxiter),
                            "--omega",
                            str(args.pso_omega),
                            "--c1",
                            str(args.pso_c1),
                            "--c2",
                            str(args.pso_c2),
                            "--tol",
                            str(args.pso_tol),
                            "--patience",
                            str(args.pso_patience),
                            "--fitnessThreshold",
                            str(args.pso_fitness_threshold),
                            "--seed",
                            str(seed),
                        ]
                    )
                elif m == "cmaes":
                    cmd.extend(
                        [
                            "--popsize",
                            str(args.cma_popsize),
                            "--maxiter",
                            str(args.cma_maxiter),
                            "--cmaSigma",
                            str(args.cma_sigma),
                            "--tol",
                            str(args.cma_tol),
                            "--patience",
                            str(args.cma_patience),
                            "--fitnessThreshold",
                            str(args.cma_fitness_threshold),
                            "--seed",
                            str(seed),
                        ]
                    )

                proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
                out_log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
                completed += 1

                if proc.returncode != 0 or (not out_json.exists()):
                    run_rows.append({**base, "status": "FAIL", "reason": f"returncode:{proc.returncode}"})
                    print(f"[run {completed:04d}/{total_runs:04d}] FAIL {obj_name} | {m} | seed={seed} (return={proc.returncode})", flush=True)
                    if pbar is not None:
                        pbar.update(1)
                    continue

                try:
                    run_rows.append(build_ok_row_from_json(base=base, out_json=out_json))
                except Exception as exc:
                    run_rows.append({**base, "status": "FAIL", "reason": f"json_parse:{exc}"})
                    print(f"[run {completed:04d}/{total_runs:04d}] FAIL {obj_name} | {m} | seed={seed} (json parse)", flush=True)
                    if pbar is not None:
                        pbar.update(1)
                    continue
                print(f"[run {completed:04d}/{total_runs:04d}] OK   {obj_name} | {m} | seed={seed}", flush=True)
                if pbar is not None:
                    pbar.update(1)

        print(f"[{idx:03d}/{len(label_rows):03d}] DONE {obj_name}", flush=True)

    if pbar is not None:
        pbar.close()

    out_root.mkdir(parents=True, exist_ok=True)
    runs_csv = out_root / "runs_raw.csv"
    existing_rows: list[dict] = []
    if runs_csv.exists():
        try:
            existing_rows = read_csv(runs_csv)
        except Exception:
            existing_rows = []

    merged_map: dict[tuple[str, str, str], dict] = {}
    for r in existing_rows + run_rows:
        obj = str(r.get("object", "")).strip()
        method = str(r.get("method", "")).strip().lower()
        seed_raw = str(r.get("seed", "")).strip()
        try:
            seed_key = str(int(float(seed_raw)))
        except Exception:
            seed_key = seed_raw
        merged_map[(obj, method, seed_key)] = r
    run_rows_merged = [
        merged_map[k] for k in sorted(merged_map.keys(), key=lambda x: (x[0], x[1], x[2]))
    ]

    run_fields = sorted({k for r in run_rows_merged for k in r.keys()}) if run_rows_merged else []
    with runs_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(run_rows_merged)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in run_rows_merged:
        if r.get("status") != "OK":
            continue
        grouped.setdefault((str(r["object"]), str(r["method"])), []).append(r)

    objects = sorted({Path(r["path"]).stem for r in label_rows})
    methods_all = sorted({str(r.get("method", "")).strip().lower() for r in run_rows_merged if str(r.get("method", "")).strip()})
    canonical_order = ["grid", "pso", "cmaes"]
    methods_eval = [m for m in canonical_order if m in methods_all] + [m for m in methods_all if m not in canonical_order]

    per_rows: list[dict] = []
    for obj in objects:
        for m in methods_eval:
            ok_rows = grouped.get((obj, m), [])
            if not ok_rows:
                per_rows.append(
                    {
                        "object": obj,
                        "method": m,
                        "status": "FAIL",
                        "n_success_runs": 0,
                        "fitness_med": float("nan"),
                        "distance_fit_med": float("nan"),
                        "coverage_ratio_med": float("nan"),
                        "residual_unexplained_ratio_med": float("nan"),
                        "num_superquadrics_med": float("nan"),
                        "runtime_ms_med": float("nan"),
                    }
                )
                continue

            per_rows.append(
                {
                    "object": obj,
                    "method": m,
                    "status": "OK",
                    "n_success_runs": len(ok_rows),
                    "fitness_med": median_or_nan([safe_float(x.get("best_fitness")) for x in ok_rows]),
                    "distance_fit_med": median_or_nan([safe_float(x.get("distance_fit")) for x in ok_rows]),
                    "coverage_ratio_med": median_or_nan([safe_float(x.get("coverage_ratio")) for x in ok_rows]),
                    "residual_unexplained_ratio_med": median_or_nan([safe_float(x.get("residual_unexplained_ratio")) for x in ok_rows]),
                    "num_superquadrics_med": median_or_nan([safe_float(x.get("num_superquadrics")) for x in ok_rows]),
                    "runtime_ms_med": median_or_nan([safe_float(x.get("runtime_ms")) for x in ok_rows]),
                }
            )

    per_csv = out_root / "per_object.csv"
    per_fields = sorted({k for r in per_rows for k in r.keys()}) if per_rows else []
    with per_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_fields)
        writer.writeheader()
        writer.writerows(per_rows)

    summary_rows: list[dict] = []
    num_objects = len(label_rows)
    for m in methods_eval:
        rows_m = [r for r in per_rows if r["method"] == m]
        ok_m = [r for r in rows_m if r["status"] == "OK"]

        fit_mean, fit_std = mean_std([safe_float(r.get("fitness_med")) for r in ok_m])
        d_mean, d_std = mean_std([safe_float(r.get("distance_fit_med")) for r in ok_m])
        cov_mean, cov_std = mean_std([safe_float(r.get("coverage_ratio_med")) for r in ok_m])
        res_mean, res_std = mean_std([safe_float(r.get("residual_unexplained_ratio_med")) for r in ok_m])
        nsq_mean, nsq_std = mean_std([safe_float(r.get("num_superquadrics_med")) for r in ok_m])
        rt_mean, rt_std = mean_std([safe_float(r.get("runtime_ms_med")) for r in ok_m])
        failure_rate = 1.0 - (len(ok_m) / num_objects if num_objects > 0 else 0.0)

        summary_rows.append(
            {
                "method": m,
                "num_objects": int(num_objects),
                "num_success": int(len(ok_m)),
                "failure_rate": float(failure_rate),
                "fitness_mean": fit_mean,
                "fitness_std": fit_std,
                "distance_fit_mean": d_mean,
                "distance_fit_std": d_std,
                "coverage_ratio_mean": cov_mean,
                "coverage_ratio_std": cov_std,
                "residual_unexplained_ratio_mean": res_mean,
                "residual_unexplained_ratio_std": res_std,
                "num_superquadrics_mean": nsq_mean,
                "num_superquadrics_std": nsq_std,
                "runtime_ms_mean": rt_mean,
                "runtime_ms_std": rt_std,
            }
        )

    table2_csv = out_root / "table2_summary.csv"
    table_fields = [
        "method",
        "num_objects",
        "num_success",
        "failure_rate",
        "fitness_mean",
        "fitness_std",
        "distance_fit_mean",
        "distance_fit_std",
        "coverage_ratio_mean",
        "coverage_ratio_std",
        "residual_unexplained_ratio_mean",
        "residual_unexplained_ratio_std",
        "num_superquadrics_mean",
        "num_superquadrics_std",
        "runtime_ms_mean",
        "runtime_ms_std",
    ]
    with table2_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    table2_md = out_root / "table2_summary.md"
    lines = [
        "| Method | Success/Total | Failure Rate | F (mean±std) | d_fit (mean±std) | Coverage (mean±std) | Residual (mean±std) | #SQ (mean±std) | Runtime ms (mean±std) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['method']} | {r['num_success']}/{r['num_objects']} | {r['failure_rate']:.3f} | "
            f"{r['fitness_mean']:.6f} +- {r['fitness_std']:.6f} | "
            f"{r['distance_fit_mean']:.6f} +- {r['distance_fit_std']:.6f} | "
            f"{r['coverage_ratio_mean']:.6f} +- {r['coverage_ratio_std']:.6f} | "
            f"{r['residual_unexplained_ratio_mean']:.6f} +- {r['residual_unexplained_ratio_std']:.6f} | "
            f"{r['num_superquadrics_mean']:.3f} +- {r['num_superquadrics_std']:.3f} | "
            f"{r['runtime_ms_mean']:.3f} +- {r['runtime_ms_std']:.3f} |"
        )
    table2_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    config_json = out_root / "run_config.json"
    config_json.write_text(
        json.dumps(
            {
                "methods_this_run": methods,
                "methods_aggregated": methods_eval,
                "seeds": seeds,
                "num_objects": len(label_rows),
                "labels": str(labels_path),
                "ply_root": str(ply_root),
                "objective": {
                    "fitness_mode": args.FitnessMode,
                    "lambda_cov": args.LambdaCov,
                    "lambda_out": args.LambdaOut,
                    "lambda_comp": args.LambdaComp,
                },
                "render": {
                    "save_images": bool(args.save_images),
                    "offscreen_render": str(args.OffscreenRender),
                    "visualize_mode": str(args.visualizeMode),
                    "image_format": str(args.image_format),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ok = sum(1 for r in run_rows_merged if r.get("status") == "OK")
    fail = sum(1 for r in run_rows_merged if r.get("status") == "FAIL")
    skip = sum(1 for r in run_rows_merged if r.get("status") == "SKIP")
    print("\nSummary")
    print(f"- objects: {num_objects}")
    print(f"- runs   : {len(run_rows_merged)}")
    print(f"- ok     : {ok}")
    print(f"- fail   : {fail}")
    print(f"- skip   : {skip}")
    print(f"- out    : {out_root}")
    print(f"- table2 : {table2_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
