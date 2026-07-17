#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

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
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(statistics.median(vals))


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def derive_budget_config(args, mode: str) -> dict:
    if mode == "best":
        return {
            "grid_steps": int(args.grid_best_steps),
            "pso_swarmsize": int(args.pso_best_swarmsize),
            "pso_maxiter": int(args.pso_best_maxiter),
            "cma_popsize": int(args.cma_best_popsize),
            "cma_maxiter": int(args.cma_best_maxiter),
            "budget_evals_per_layer": None,
        }

    # fair mode
    budget = int(args.budget_evals_per_layer)
    if budget < 16:
        raise ValueError("--budget-evals-per-layer must be >= 16 for 4D search.")

    if int(args.grid_steps) > 0:
        grid_steps = int(args.grid_steps)
    else:
        grid_steps = int(np.floor(budget ** 0.25))
        grid_steps = max(2, grid_steps)

    pso_swarmsize = int(args.pso_swarmsize)
    cma_popsize = int(args.cma_popsize)
    pso_maxiter = max(1, budget // max(1, pso_swarmsize))
    cma_maxiter = max(1, budget // max(1, cma_popsize))

    return {
        "grid_steps": int(grid_steps),
        "pso_swarmsize": int(pso_swarmsize),
        "pso_maxiter": int(pso_maxiter),
        "cma_popsize": int(cma_popsize),
        "cma_maxiter": int(cma_maxiter),
        "budget_evals_per_layer": int(budget),
    }


def estimate_effective_evals(method: str, data: dict, cfg: dict) -> int:
    layer_timing = data.get("layer_timing", [])
    if not isinstance(layer_timing, list):
        return 0

    if method == "grid":
        return int(sum(int(x.get("grid_evaluations_effective", 0)) for x in layer_timing if isinstance(x, dict)))

    if method == "pso":
        swarm = int(cfg["pso_swarmsize"])
        iters = sum(int(x.get("pso_iterations", 0)) for x in layer_timing if isinstance(x, dict))
        return int(iters * swarm)

    if method == "cmaes":
        popsize = int(cfg["cma_popsize"])
        iters = sum(int(x.get("cma_iterations", 0)) for x in layer_timing if isinstance(x, dict))
        return int(iters * popsize)

    return 0


def build_curve_points(method: str, data: dict, cfg: dict) -> list[tuple[int, float]]:
    layer_histories = data.get("layer_histories", [])
    if not isinstance(layer_histories, list):
        return []

    if method == "grid":
        step = 1
    elif method == "pso":
        step = int(cfg["pso_swarmsize"])
    else:
        step = int(cfg["cma_popsize"])

    points: list[tuple[int, float]] = []
    eval_count = 0
    global_best = float("inf")
    points.append((0, global_best))

    for hist in layer_histories:
        if not isinstance(hist, list):
            continue
        for v in hist:
            fv = safe_float(v, float("inf"))
            eval_count += step
            if fv < global_best:
                global_best = fv
            points.append((int(eval_count), float(global_best)))

    if len(points) <= 1 and np.isfinite(safe_float(data.get("best_fitness"), float("nan"))):
        points.append((step, safe_float(data.get("best_fitness"))))

    return points


def main() -> int:
    parser = argparse.ArgumentParser(description="E3 optimizer overhead benchmark (fair budget vs best mode).")
    parser.add_argument("--labels", type=str, default="data/kit_superquadric_labels.csv")
    parser.add_argument("--ply-root", type=str, default="data/KIT_ObjectModels_25k_ply")
    parser.add_argument("--output", type=str, default="exp2_3/results/e3_budget")
    parser.add_argument("--methods", type=str, default="grid,pso,cmaes")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--mode", type=str, default="fair", choices=["fair", "best"])

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

    # fair-budget config
    parser.add_argument("--budget-evals-per-layer", type=int, default=625)
    parser.add_argument("--grid-steps", type=int, default=0, help="Override fair-mode auto grid steps if > 0.")
    parser.add_argument("--pso-swarmsize", type=int, default=10)
    parser.add_argument("--cma-popsize", type=int, default=10)

    # best-mode config
    parser.add_argument("--grid-best-steps", type=int, default=10)
    parser.add_argument("--pso-best-swarmsize", type=int, default=20)
    parser.add_argument("--pso-best-maxiter", type=int, default=30)
    parser.add_argument("--cma-best-popsize", type=int, default=20)
    parser.add_argument("--cma-best-maxiter", type=int, default=30)

    # shared optimizer settings
    parser.add_argument("--pso-omega", type=float, default=0.72)
    parser.add_argument("--pso-c1", type=float, default=1.49)
    parser.add_argument("--pso-c2", type=float, default=1.49)
    parser.add_argument("--pso-tol", type=float, default=1e-12)
    parser.add_argument("--pso-patience", type=int, default=10**9)
    parser.add_argument("--pso-fitness-threshold", type=float, default=-1.0)

    parser.add_argument("--cma-sigma", type=float, default=0.15)
    parser.add_argument("--cma-tol", type=float, default=1e-12)
    parser.add_argument("--cma-patience", type=int, default=10**9)
    parser.add_argument("--cma-fitness-threshold", type=float, default=-1.0)

    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    labels_path = (repo_root / args.labels).resolve()
    ply_root = (repo_root / args.ply_root).resolve()
    out_root = (repo_root / args.output).resolve()

    methods = parse_methods(args.methods)
    seeds = parse_seeds(args.seeds)
    budget_cfg = derive_budget_config(args, mode=args.mode)

    label_rows = [r for r in read_csv(labels_path) if r.get("label", "").strip().lower() == "multi"]
    if args.max_objects > 0:
        label_rows = label_rows[: args.max_objects]

    json_dir = out_root / "json"
    log_dir = out_root / "log"
    for m in methods:
        (json_dir / m).mkdir(parents=True, exist_ok=True)
        (log_dir / m).mkdir(parents=True, exist_ok=True)

    runs_per_object = sum((1 if m == "grid" else len(seeds)) for m in methods)
    total_runs = len(label_rows) * runs_per_object

    print(f"E3 mode: {args.mode}")
    print(f"Objects: {len(label_rows)}")
    print(f"Methods: {methods}")
    print(f"Budget cfg: {budget_cfg}")
    print(f"Planned runs: {total_runs}")

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total_runs, desc=f"E3-{args.mode} runs", unit="run", dynamic_ncols=True)

    run_rows: list[dict] = []
    curve_rows: list[dict] = []
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

                if args.skip_existing and out_json.exists():
                    completed += 1
                    run_rows.append(
                        {
                            "object": obj_name,
                            "method": m,
                            "seed": int(seed),
                            "status": "SKIP",
                            "reason": "existing_json",
                            "json": str(out_json),
                            "log": str(out_log),
                        }
                    )
                    print(f"[run {completed:04d}/{total_runs:04d}] SKIP {obj_name} | {m} | seed={seed}", flush=True)
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

                if m == "grid":
                    cmd.extend(["--gridSteps", str(budget_cfg["grid_steps"])])
                elif m == "pso":
                    cmd.extend(
                        [
                            "--swarmsize",
                            str(budget_cfg["pso_swarmsize"]),
                            "--maxiter",
                            str(budget_cfg["pso_maxiter"]),
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
                else:
                    cmd.extend(
                        [
                            "--popsize",
                            str(budget_cfg["cma_popsize"]),
                            "--maxiter",
                            str(budget_cfg["cma_maxiter"]),
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

                base = {
                    "object": obj_name,
                    "method": m,
                    "seed": int(seed),
                    "json": str(out_json),
                    "log": str(out_log),
                    "mode": args.mode,
                }

                if proc.returncode != 0 or (not out_json.exists()):
                    run_rows.append({**base, "status": "FAIL", "reason": f"returncode:{proc.returncode}"})
                    print(f"[run {completed:04d}/{total_runs:04d}] FAIL {obj_name} | {m} | seed={seed} (return={proc.returncode})", flush=True)
                    if pbar is not None:
                        pbar.update(1)
                    continue

                try:
                    data = json.loads(out_json.read_text(encoding="utf-8"))
                except Exception as exc:
                    run_rows.append({**base, "status": "FAIL", "reason": f"json_parse:{exc}"})
                    print(f"[run {completed:04d}/{total_runs:04d}] FAIL {obj_name} | {m} | seed={seed} (json parse)", flush=True)
                    if pbar is not None:
                        pbar.update(1)
                    continue

                fd = data.get("fitness_details", {})
                tm = data.get("timing", {})
                eff_evals = estimate_effective_evals(method=m, data=data, cfg=budget_cfg)

                run_rows.append(
                    {
                        **base,
                        "status": "OK",
                        "best_fitness": safe_float(data.get("best_fitness")),
                        "distance_fit": safe_float(fd.get("distance_fit")),
                        "coverage_ratio": safe_float(fd.get("coverage_ratio")),
                        "residual_unexplained_ratio": safe_float(fd.get("residual_unexplained_ratio")),
                        "num_superquadrics": safe_float(fd.get("num_superquadrics")),
                        "runtime_ms": safe_float(tm.get("total_runtime_ms")),
                        "effective_evals": int(eff_evals),
                    }
                )

                curve_points = build_curve_points(method=m, data=data, cfg=budget_cfg)
                run_id = f"{obj_name}::{m}::seed{seed}"
                for x, y in curve_points:
                    curve_rows.append(
                        {
                            "run_id": run_id,
                            "object": obj_name,
                            "method": m,
                            "seed": int(seed),
                            "eval_count": int(x),
                            "best_fitness": float(y),
                        }
                    )

                print(f"[run {completed:04d}/{total_runs:04d}] OK   {obj_name} | {m} | seed={seed}", flush=True)
                if pbar is not None:
                    pbar.update(1)

        print(f"[{idx:03d}/{len(label_rows):03d}] DONE {obj_name}", flush=True)

    if pbar is not None:
        pbar.close()

    out_root.mkdir(parents=True, exist_ok=True)

    runs_csv = out_root / "runs_raw.csv"
    existing_runs: list[dict] = []
    if runs_csv.exists():
        try:
            existing_runs = read_csv(runs_csv)
        except Exception:
            existing_runs = []

    merged_run_map: dict[tuple[str, str, str, str], dict] = {}
    for r in existing_runs + run_rows:
        obj = str(r.get("object", "")).strip()
        method = str(r.get("method", "")).strip().lower()
        mode_key = str(r.get("mode", args.mode)).strip().lower()
        seed_raw = str(r.get("seed", "")).strip()
        try:
            seed_key = str(int(float(seed_raw)))
        except Exception:
            seed_key = seed_raw
        merged_run_map[(obj, method, mode_key, seed_key)] = r
    run_rows_merged = [
        merged_run_map[k] for k in sorted(merged_run_map.keys(), key=lambda x: (x[2], x[0], x[1], x[3]))
    ]

    run_fields = sorted({k for r in run_rows_merged for k in r.keys()}) if run_rows_merged else []
    with runs_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(run_rows_merged)

    curves_csv = out_root / "curves.csv"
    existing_curves: list[dict] = []
    if curves_csv.exists():
        try:
            existing_curves = read_csv(curves_csv)
        except Exception:
            existing_curves = []
    merged_curve_map: dict[tuple[str, str], dict] = {}
    for r in existing_curves + curve_rows:
        run_id = str(r.get("run_id", "")).strip()
        eval_raw = str(r.get("eval_count", "")).strip()
        merged_curve_map[(run_id, eval_raw)] = r
    curve_rows_merged = [
        merged_curve_map[k] for k in sorted(merged_curve_map.keys(), key=lambda x: (x[0], safe_float(x[1], float("inf"))))
    ]

    curve_fields = ["run_id", "object", "method", "seed", "eval_count", "best_fitness"]
    with curves_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=curve_fields)
        writer.writeheader()
        writer.writerows(curve_rows_merged)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in run_rows_merged:
        if str(r.get("mode", args.mode)).strip().lower() != str(args.mode).lower():
            continue
        if r.get("status") != "OK":
            continue
        grouped.setdefault((str(r["object"]), str(r["method"])), []).append(r)

    objects = sorted({Path(r["path"]).stem for r in label_rows})
    methods_all = sorted(
        {
            str(r.get("method", "")).strip().lower()
            for r in run_rows_merged
            if str(r.get("mode", args.mode)).strip().lower() == str(args.mode).lower()
            and str(r.get("method", "")).strip()
        }
    )
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
                        "runtime_ms_med": float("nan"),
                        "effective_evals_med": float("nan"),
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
                    "runtime_ms_med": median_or_nan([safe_float(x.get("runtime_ms")) for x in ok_rows]),
                    "effective_evals_med": median_or_nan([safe_float(x.get("effective_evals")) for x in ok_rows]),
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
        ok_m = [r for r in per_rows if r.get("method") == m and r.get("status") == "OK"]
        fit_mean, fit_std = mean_std([safe_float(r.get("fitness_med")) for r in ok_m])
        d_mean, d_std = mean_std([safe_float(r.get("distance_fit_med")) for r in ok_m])
        rt_mean, rt_std = mean_std([safe_float(r.get("runtime_ms_med")) for r in ok_m])
        ev_mean, ev_std = mean_std([safe_float(r.get("effective_evals_med")) for r in ok_m])
        failure_rate = 1.0 - (len(ok_m) / num_objects if num_objects > 0 else 0.0)

        summary_rows.append(
            {
                "mode": args.mode,
                "method": m,
                "num_objects": int(num_objects),
                "num_success": int(len(ok_m)),
                "failure_rate": float(failure_rate),
                "fitness_mean": fit_mean,
                "fitness_std": fit_std,
                "distance_fit_mean": d_mean,
                "distance_fit_std": d_std,
                "runtime_ms_mean": rt_mean,
                "runtime_ms_std": rt_std,
                "effective_evals_mean": ev_mean,
                "effective_evals_std": ev_std,
            }
        )

    table3_csv = out_root / "table3_summary.csv"
    table_fields = [
        "mode",
        "method",
        "num_objects",
        "num_success",
        "failure_rate",
        "fitness_mean",
        "fitness_std",
        "distance_fit_mean",
        "distance_fit_std",
        "runtime_ms_mean",
        "runtime_ms_std",
        "effective_evals_mean",
        "effective_evals_std",
    ]
    with table3_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    table3_md = out_root / "table3_summary.md"
    lines = [
        "| Mode | Method | Success/Total | Failure Rate | F (mean+-std) | d_fit (mean+-std) | Runtime ms (mean+-std) | Effective evals (mean+-std) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['mode']} | {r['method']} | {r['num_success']}/{r['num_objects']} | {r['failure_rate']:.3f} | "
            f"{r['fitness_mean']:.6f} +- {r['fitness_std']:.6f} | "
            f"{r['distance_fit_mean']:.6f} +- {r['distance_fit_std']:.6f} | "
            f"{r['runtime_ms_mean']:.3f} +- {r['runtime_ms_std']:.3f} | "
            f"{r['effective_evals_mean']:.1f} +- {r['effective_evals_std']:.1f} |"
        )
    table3_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    config_json = out_root / "run_config.json"
    config_json.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "methods_this_run": methods,
                "methods_aggregated": methods_eval,
                "seeds": seeds,
                "num_objects": len(label_rows),
                "labels": str(labels_path),
                "ply_root": str(ply_root),
                "budget_cfg": budget_cfg,
                "objective": {
                    "fitness_mode": args.FitnessMode,
                    "lambda_cov": args.LambdaCov,
                    "lambda_out": args.LambdaOut,
                    "lambda_comp": args.LambdaComp,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ok = sum(
        1
        for r in run_rows_merged
        if str(r.get("mode", args.mode)).strip().lower() == str(args.mode).lower() and r.get("status") == "OK"
    )
    fail = sum(
        1
        for r in run_rows_merged
        if str(r.get("mode", args.mode)).strip().lower() == str(args.mode).lower() and r.get("status") == "FAIL"
    )
    skip = sum(
        1
        for r in run_rows_merged
        if str(r.get("mode", args.mode)).strip().lower() == str(args.mode).lower() and r.get("status") == "SKIP"
    )
    print("\nSummary")
    print(f"- mode  : {args.mode}")
    print(f"- runs  : {sum(1 for r in run_rows_merged if str(r.get('mode', args.mode)).strip().lower() == str(args.mode).lower())}")
    print(f"- ok    : {ok}")
    print(f"- fail  : {fail}")
    print(f"- skip  : {skip}")
    print(f"- out   : {out_root}")
    print(f"- table3: {table3_csv}")
    print(f"- curves: {curves_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
