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
    "grid": "run_single_ems_grid_2params.py",
    "pso": "run_single_ems_pso_2params.py",
    "cmaes": "run_single_ems_cmaes_2params.py",
}


def parse_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_methods(methods_str: str) -> list[str]:
    methods = [m.strip().lower() for m in methods_str.split(",") if m.strip()]
    for m in methods:
        if m not in SCRIPT_BY_METHOD:
            raise ValueError(f"Unsupported method '{m}'. Supported: {sorted(SCRIPT_BY_METHOD.keys())}")
    return methods


def parse_seeds(seeds_str: str) -> list[int]:
    seeds = [int(s.strip()) for s in seeds_str.split(",") if s.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def median_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.median(values))


def safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E1 single-superquadric benchmark on KIT single objects (grid/pso/cma-es)."
    )
    parser.add_argument("--labels", type=str, default="data/kit_superquadric_labels.csv")
    parser.add_argument("--ply-root", type=str, default="data/KIT_ObjectModels_25k_ply")
    parser.add_argument("--output", type=str, default="exp1/results/e1_single")
    parser.add_argument("--methods", type=str, default="grid,pso,cmaes")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max-objects", type=int, default=0, help="For smoke test. 0 means all.")

    parser.add_argument("--GlobalNormalize", type=str, default="True")
    parser.add_argument("--GlobalNormMethod", type=str, default="ems_matlab")
    parser.add_argument("--OutputInOriginalScale", type=str, default="True")
    parser.add_argument("--Rescale", type=str, default="False")
    parser.add_argument("--AdaptiveUpperBound", type=str, default="True")
    parser.add_argument("--pThreshold", type=str, default="0.1")

    # Unified lightweight budget (about 100 evaluations each)
    parser.add_argument("--grid-outlier-steps", type=int, default=10)
    parser.add_argument("--grid-sigma-steps", type=int, default=10)
    parser.add_argument("--pso-swarmsize", type=int, default=10)
    parser.add_argument("--pso-maxiter", type=int, default=10)
    parser.add_argument("--cma-popsize", type=int, default=10)
    parser.add_argument("--cma-maxiter", type=int, default=10)
    parser.add_argument("--LambdaCov", type=float, default=0.0)

    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    labels_path = (repo_root / args.labels).resolve()
    ply_root = (repo_root / args.ply_root).resolve()
    out_root = (repo_root / args.output).resolve()
    methods = parse_methods(args.methods)
    seeds = parse_seeds(args.seeds)

    rows = [r for r in parse_csv(labels_path) if r.get("label", "").strip().lower() == "single"]
    if args.max_objects > 0:
        rows = rows[: args.max_objects]

    json_dir = out_root / "json"
    log_dir = out_root / "log"
    for method in methods:
        (json_dir / method).mkdir(parents=True, exist_ok=True)
        (log_dir / method).mkdir(parents=True, exist_ok=True)

    run_rows: list[dict] = []
    runs_per_object = 0
    for method in methods:
        runs_per_object += 1 if method == "grid" else len(seeds)
    total_runs = len(rows) * runs_per_object
    completed_runs = 0
    print(f"Single objects to evaluate: {len(rows)}")
    print(f"Methods: {methods}")
    print(f"Planned runs: {total_runs}")
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total_runs, desc="Exp1 runs", unit="run", dynamic_ncols=True)
    else:
        print("tqdm not installed, fallback to plain text progress logs.")

    for idx, r in enumerate(rows, start=1):
        rel = r["path"].strip()
        obj_name = Path(rel).stem
        ply_path = ply_root / rel
        if not ply_path.exists():
            for method in methods:
                method_seeds = [seeds[0]] if method == "grid" else seeds
                for seed in method_seeds:
                    completed_runs += 1
                    run_rows.append(
                        {
                            "object": obj_name,
                            "method": method,
                            "seed": int(seed),
                            "status": "FAIL",
                            "reason": f"missing_ply:{ply_path}",
                        }
                    )
                    print(
                        f"[run {completed_runs:04d}/{total_runs:04d}] "
                        f"FAIL {obj_name} | {method} | seed={seed}: missing ply",
                        flush=True,
                    )
                    if pbar is not None:
                        pbar.update(1)
            continue

        for method in methods:
            method_seeds = [seeds[0]] if method == "grid" else seeds
            for seed in method_seeds:
                planned_idx = completed_runs + 1
                out_json = json_dir / method / f"{obj_name}__seed{seed}.json"
                out_log = log_dir / method / f"{obj_name}__seed{seed}.log"
                if args.skip_existing and out_json.exists():
                    completed_runs += 1
                    run_rows.append(
                        {
                            "object": obj_name,
                            "method": method,
                            "seed": int(seed),
                            "status": "SKIP",
                            "reason": "existing_json",
                            "json": str(out_json),
                            "log": str(out_log),
                        }
                    )
                    print(
                        f"[run {completed_runs:04d}/{total_runs:04d}] "
                        f"SKIP {obj_name} | {method} | seed={seed}",
                        flush=True,
                    )
                    if pbar is not None:
                        pbar.update(1)
                    continue

                runner = repo_root / "src" / SCRIPT_BY_METHOD[method]
                print(
                    f"[run {planned_idx:04d}/{total_runs:04d}] "
                    f"START {obj_name} | {method} | seed={seed}",
                    flush=True,
                )
                if pbar is not None:
                    pbar.set_postfix_str(f"{obj_name} | {method} | seed={seed}")
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
                    "--pThreshold",
                    str(args.pThreshold),
                    "--LambdaCov",
                    str(args.LambdaCov),
                ]

                if method == "grid":
                    cmd.extend(
                        [
                            "--OutlierRatioSteps",
                            str(args.grid_outlier_steps),
                            "--SigmaSteps",
                            str(args.grid_sigma_steps),
                        ]
                    )
                elif method == "pso":
                    cmd.extend(
                        [
                            "--swarmsize",
                            str(args.pso_swarmsize),
                            "--maxiter",
                            str(args.pso_maxiter),
                            "--seed",
                            str(seed),
                        ]
                    )
                elif method == "cmaes":
                    cmd.extend(
                        [
                            "--popsize",
                            str(args.cma_popsize),
                            "--maxiter",
                            str(args.cma_maxiter),
                            "--seed",
                            str(seed),
                        ]
                    )

                proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
                out_log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
                completed_runs += 1

                base_row = {
                    "object": obj_name,
                    "method": method,
                    "seed": int(seed),
                    "json": str(out_json),
                    "log": str(out_log),
                }
                if proc.returncode != 0 or (not out_json.exists()):
                    run_rows.append(
                        {
                            **base_row,
                            "status": "FAIL",
                            "reason": f"returncode:{proc.returncode}",
                        }
                    )
                    print(
                        f"[run {completed_runs:04d}/{total_runs:04d}] "
                        f"FAIL {obj_name} | {method} | seed={seed} (return={proc.returncode})",
                        flush=True,
                    )
                    if pbar is not None:
                        pbar.update(1)
                    continue

                try:
                    d = json.loads(out_json.read_text(encoding="utf-8"))
                except Exception as exc:
                    run_rows.append(
                        {
                            **base_row,
                            "status": "FAIL",
                            "reason": f"json_parse:{exc}",
                        }
                    )
                    print(
                        f"[run {completed_runs:04d}/{total_runs:04d}] "
                        f"FAIL {obj_name} | {method} | seed={seed} (json parse)",
                        flush=True,
                    )
                    if pbar is not None:
                        pbar.update(1)
                    continue

                total_points = safe_float(d.get("summary", {}).get("total_points"), float("nan"))
                inlier_points = safe_float(d.get("summary", {}).get("inlier_points"), float("nan"))
                inlier_ratio = safe_float(
                    d.get("summary", {}).get("inlier_ratio"),
                    inlier_points / total_points if total_points > 0 else float("nan"),
                )
                run_rows.append(
                    {
                        **base_row,
                        "status": "OK",
                        "best_fitness": safe_float(d.get("best_fitness")),
                        "inlier_ratio": inlier_ratio,
                        "runtime_ms": safe_float(d.get("timing", {}).get("total_runtime_ms")),
                    }
                )
                print(
                    f"[run {completed_runs:04d}/{total_runs:04d}] "
                    f"OK   {obj_name} | {method} | seed={seed}",
                    flush=True,
                )
                if pbar is not None:
                    pbar.update(1)
        print(f"[{idx:03d}/{len(rows):03d}] DONE {obj_name}", flush=True)

    if pbar is not None:
        pbar.close()

    out_root.mkdir(parents=True, exist_ok=True)
    runs_csv = out_root / "runs_raw.csv"
    run_fields = sorted({k for row in run_rows for k in row.keys()}) if run_rows else []
    with runs_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(run_rows)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in run_rows:
        if row.get("status") != "OK":
            continue
        grouped.setdefault((str(row["object"]), str(row["method"])), []).append(row)

    per_object_rows: list[dict] = []
    objects = sorted({str(r["object"]) for r in run_rows})
    for obj in objects:
        for method in methods:
            key = (obj, method)
            ok_rows = grouped.get(key, [])
            if not ok_rows:
                per_object_rows.append(
                    {
                        "object": obj,
                        "method": method,
                        "status": "FAIL",
                        "n_success_runs": 0,
                        "best_fitness_med": float("nan"),
                        "inlier_ratio_med": float("nan"),
                        "runtime_ms_med": float("nan"),
                    }
                )
                continue
            per_object_rows.append(
                {
                    "object": obj,
                    "method": method,
                    "status": "OK",
                    "n_success_runs": len(ok_rows),
                    "best_fitness_med": median_or_nan([safe_float(r.get("best_fitness")) for r in ok_rows]),
                    "inlier_ratio_med": median_or_nan([safe_float(r.get("inlier_ratio")) for r in ok_rows]),
                    "runtime_ms_med": median_or_nan([safe_float(r.get("runtime_ms")) for r in ok_rows]),
                }
            )

    per_object_csv = out_root / "per_object.csv"
    per_fields = sorted({k for row in per_object_rows for k in row.keys()}) if per_object_rows else []
    with per_object_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_fields)
        writer.writeheader()
        writer.writerows(per_object_rows)

    summary_rows: list[dict] = []
    num_objects = len(objects)
    for method in methods:
        rows_m = [r for r in per_object_rows if r["method"] == method]
        ok_m = [r for r in rows_m if r["status"] == "OK"]

        def stat_mean_std(key: str):
            vals = [safe_float(r[key]) for r in ok_m]
            vals = [v for v in vals if np_isfinite(v)]
            if not vals:
                return float("nan"), float("nan")
            if len(vals) == 1:
                return float(vals[0]), 0.0
            return float(statistics.mean(vals)), float(statistics.stdev(vals))

        fit_mean, fit_std = stat_mean_std("best_fitness_med")
        in_mean, in_std = stat_mean_std("inlier_ratio_med")
        rt_mean, rt_std = stat_mean_std("runtime_ms_med")

        fail_rate = 1.0 - (len(ok_m) / num_objects if num_objects > 0 else 0.0)
        summary_rows.append(
            {
                "method": method,
                "num_objects": num_objects,
                "num_success": len(ok_m),
                "failure_rate": float(fail_rate),
                "fit_error_mean": fit_mean,
                "fit_error_std": fit_std,
                "inlier_ratio_mean": in_mean,
                "inlier_ratio_std": in_std,
                "runtime_ms_mean": rt_mean,
                "runtime_ms_std": rt_std,
            }
        )

    table1_csv = out_root / "table1_summary.csv"
    table_fields = [
        "method",
        "num_objects",
        "num_success",
        "failure_rate",
        "fit_error_mean",
        "fit_error_std",
        "inlier_ratio_mean",
        "inlier_ratio_std",
        "runtime_ms_mean",
        "runtime_ms_std",
    ]
    with table1_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    table1_md = out_root / "table1_summary.md"
    lines = [
        "| Method | Success/Total | Failure Rate | Fit Error (mean±std) | Inlier Ratio (mean±std) | Runtime ms (mean±std) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['method']} | {r['num_success']}/{r['num_objects']} | {r['failure_rate']:.3f} | "
            f"{r['fit_error_mean']:.6f} +- {r['fit_error_std']:.6f} | "
            f"{r['inlier_ratio_mean']:.6f} +- {r['inlier_ratio_std']:.6f} | "
            f"{r['runtime_ms_mean']:.3f} +- {r['runtime_ms_std']:.3f} |"
        )
    table1_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok = sum(1 for r in run_rows if r.get("status") == "OK")
    fail = sum(1 for r in run_rows if r.get("status") == "FAIL")
    skip = sum(1 for r in run_rows if r.get("status") == "SKIP")
    print("\nSummary")
    print(f"- objects: {num_objects}")
    print(f"- runs   : {len(run_rows)}")
    print(f"- ok     : {ok}")
    print(f"- fail   : {fail}")
    print(f"- skip   : {skip}")
    print(f"- out    : {out_root}")
    print(f"- table1 : {table1_csv}")
    return 0


def np_isfinite(v: float) -> bool:
    try:
        return bool(abs(v) != float("inf") and v == v)
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
