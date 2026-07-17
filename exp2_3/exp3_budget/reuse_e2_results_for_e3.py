#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_methods(value: str) -> list[str]:
    methods = [m.strip().lower() for m in value.split(",") if m.strip()]
    valid = {"grid", "pso", "cmaes"}
    for m in methods:
        if m not in valid:
            raise ValueError(f"Unsupported method '{m}'. Supported: {sorted(valid)}")
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


def estimate_effective_evals(method: str, data: dict) -> int:
    layer_timing = data.get("layer_timing", [])
    if not isinstance(layer_timing, list):
        return 0

    if method == "grid":
        return int(sum(int(x.get("grid_evaluations_effective", 0)) for x in layer_timing if isinstance(x, dict)))
    if method == "pso":
        swarm = int(data.get("pso_config", {}).get("swarmsize", 1))
        iters = sum(int(x.get("pso_iterations", 0)) for x in layer_timing if isinstance(x, dict))
        return int(iters * max(1, swarm))
    if method == "cmaes":
        popsize = int(data.get("cmaes_config", {}).get("popsize", 1))
        iters = sum(int(x.get("cma_iterations", 0)) for x in layer_timing if isinstance(x, dict))
        return int(iters * max(1, popsize))
    return 0


def build_curve_points(method: str, data: dict) -> list[tuple[int, float]]:
    layer_histories = data.get("layer_histories", [])
    if not isinstance(layer_histories, list):
        return []

    if method == "grid":
        step = 1
    elif method == "pso":
        step = int(data.get("pso_config", {}).get("swarmsize", 1))
    else:
        step = int(data.get("cmaes_config", {}).get("popsize", 1))
    step = max(1, step)

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
    parser = argparse.ArgumentParser(
        description="Reuse E2 results and derive an E3 supplementary analysis without rerunning reconstruction."
    )
    parser.add_argument("--e2-root", type=str, default="exp2_3/results/e2_multi_full")
    parser.add_argument("--output", type=str, default="exp2_3/results/e3_from_e2_supp")
    parser.add_argument("--methods", type=str, default="grid,pso,cmaes")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--labels", type=str, default="data/kit_superquadric_labels.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    e2_root = (repo_root / args.e2_root).resolve()
    out_root = (repo_root / args.output).resolve()
    labels_path = (repo_root / args.labels).resolve()

    methods = parse_methods(args.methods)
    seeds = parse_seeds(args.seeds)
    seeds_set = {int(s) for s in seeds}

    e2_runs_csv = e2_root / "runs_raw.csv"
    if not e2_runs_csv.exists():
        raise FileNotFoundError(f"E2 runs file not found: {e2_runs_csv}")
    source_rows = read_csv(e2_runs_csv)

    multi_objects = [r["path"] for r in read_csv(labels_path) if r.get("label", "").strip().lower() == "multi"]
    object_names = sorted({Path(p).stem for p in multi_objects})

    filtered_rows: list[dict] = []
    for r in source_rows:
        method = str(r.get("method", "")).strip().lower()
        if method not in methods:
            continue
        seed_raw = str(r.get("seed", "")).strip()
        try:
            seed = int(float(seed_raw))
        except Exception:
            continue
        if method != "grid" and seed not in seeds_set:
            continue
        if method == "grid" and (len(seeds_set) > 0 and seed not in seeds_set):
            continue
        filtered_rows.append(r)

    if not filtered_rows:
        raise RuntimeError("No rows selected from E2 results. Check --methods/--seeds.")

    run_rows: list[dict] = []
    curve_rows: list[dict] = []

    for r in filtered_rows:
        obj = str(r.get("object", "")).strip()
        method = str(r.get("method", "")).strip().lower()
        seed = int(float(r.get("seed", 0)))
        status = str(r.get("status", "")).strip().upper()
        base = {
            "object": obj,
            "method": method,
            "seed": seed,
            "mode": "supplement_from_e2",
            "json": str(r.get("json", "")),
            "log": str(r.get("log", "")),
        }

        if status != "OK":
            run_rows.append({**base, "status": status if status else "FAIL", "reason": str(r.get("reason", ""))})
            continue

        json_path = Path(str(r.get("json", ""))).expanduser()
        if not json_path.is_absolute():
            json_path = (repo_root / json_path).resolve()
        if not json_path.exists():
            run_rows.append({**base, "status": "FAIL", "reason": f"missing_json:{json_path}"})
            continue

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            run_rows.append({**base, "status": "FAIL", "reason": f"json_parse:{exc}"})
            continue

        fd = data.get("fitness_details", {})
        tm = data.get("timing", {})
        eff_evals = estimate_effective_evals(method=method, data=data)

        run_rows.append(
            {
                **base,
                "status": "OK",
                "best_fitness": safe_float(data.get("best_fitness", r.get("best_fitness"))),
                "distance_fit": safe_float(fd.get("distance_fit", r.get("distance_fit"))),
                "coverage_ratio": safe_float(fd.get("coverage_ratio", r.get("coverage_ratio"))),
                "residual_unexplained_ratio": safe_float(
                    fd.get("residual_unexplained_ratio", r.get("residual_unexplained_ratio"))
                ),
                "num_superquadrics": safe_float(fd.get("num_superquadrics", r.get("num_superquadrics"))),
                "runtime_ms": safe_float(tm.get("total_runtime_ms", r.get("runtime_ms"))),
                "effective_evals": int(eff_evals),
            }
        )

        run_id = f"{obj}::{method}::seed{seed}"
        for x, y in build_curve_points(method=method, data=data):
            curve_rows.append(
                {
                    "run_id": run_id,
                    "object": obj,
                    "method": method,
                    "seed": seed,
                    "eval_count": int(x),
                    "best_fitness": float(y),
                }
            )

    out_root.mkdir(parents=True, exist_ok=True)

    runs_csv = out_root / "runs_raw.csv"
    run_fields = sorted({k for row in run_rows for k in row.keys()}) if run_rows else []
    with runs_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(run_rows)

    curves_csv = out_root / "curves.csv"
    curve_fields = ["run_id", "object", "method", "seed", "eval_count", "best_fitness"]
    with curves_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=curve_fields)
        writer.writeheader()
        writer.writerows(curve_rows)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in run_rows:
        if row.get("status") != "OK":
            continue
        grouped.setdefault((str(row["object"]), str(row["method"])), []).append(row)

    methods_all = sorted({str(r.get("method", "")).strip().lower() for r in run_rows if str(r.get("method", "")).strip()})
    canonical_order = ["grid", "pso", "cmaes"]
    methods_eval = [m for m in canonical_order if m in methods_all] + [m for m in methods_all if m not in canonical_order]

    per_rows: list[dict] = []
    for obj in object_names:
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
    per_fields = sorted({k for row in per_rows for k in row.keys()}) if per_rows else []
    with per_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_fields)
        writer.writeheader()
        writer.writerows(per_rows)

    summary_rows: list[dict] = []
    num_objects = len(object_names)
    for m in methods_eval:
        ok_m = [r for r in per_rows if r.get("method") == m and r.get("status") == "OK"]
        fit_mean, fit_std = mean_std([safe_float(r.get("fitness_med")) for r in ok_m])
        d_mean, d_std = mean_std([safe_float(r.get("distance_fit_med")) for r in ok_m])
        rt_mean, rt_std = mean_std([safe_float(r.get("runtime_ms_med")) for r in ok_m])
        ev_mean, ev_std = mean_std([safe_float(r.get("effective_evals_med")) for r in ok_m])
        failure_rate = 1.0 - (len(ok_m) / num_objects if num_objects > 0 else 0.0)
        summary_rows.append(
            {
                "mode": "supplement_from_e2",
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

    source_config = {}
    src_cfg_path = e2_root / "run_config.json"
    if src_cfg_path.exists():
        try:
            source_config = json.loads(src_cfg_path.read_text(encoding="utf-8"))
        except Exception:
            source_config = {}

    config_json = out_root / "run_config.json"
    config_json.write_text(
        json.dumps(
            {
                "mode": "supplement_from_e2",
                "source_e2_root": str(e2_root),
                "methods_selected": methods,
                "methods_aggregated": methods_eval,
                "seeds_selected": seeds,
                "num_objects": num_objects,
                "source_run_config": source_config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ok = sum(1 for r in run_rows if r.get("status") == "OK")
    fail = sum(1 for r in run_rows if r.get("status") == "FAIL")
    print("Summary")
    print(f"- source: {e2_root}")
    print(f"- out   : {out_root}")
    print(f"- runs  : {len(run_rows)}")
    print(f"- ok    : {ok}")
    print(f"- fail  : {fail}")
    print(f"- table3: {table3_csv}")
    print(f"- curves: {curves_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
