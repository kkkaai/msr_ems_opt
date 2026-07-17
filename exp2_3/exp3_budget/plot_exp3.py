#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOT_FONT_SIZE = 24
plt.rcParams.update(
    {
        "font.size": PLOT_FONT_SIZE,
        "axes.titlesize": PLOT_FONT_SIZE,
        "axes.labelsize": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_FONT_SIZE,
        "ytick.labelsize": PLOT_FONT_SIZE,
        "legend.fontsize": PLOT_FONT_SIZE,
        "mathtext.fontset": "stix",
        "font.family": "Times New Roman",
    }
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v, default=np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def build_run_curves(curve_rows: list[dict], methods: list[str]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in curve_rows:
        run_id = str(r.get("run_id", "")).strip()
        if not run_id:
            continue
        grouped[run_id].append(r)

    curves_by_method: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {m: [] for m in methods}

    for run_id, rows in grouped.items():
        method = str(rows[0].get("method", "")).strip()
        if method not in curves_by_method:
            continue

        rows_sorted = sorted(rows, key=lambda x: to_float(x.get("eval_count"), np.inf))
        xs = np.array([to_float(r.get("eval_count")) for r in rows_sorted], dtype=float)
        ys = np.array([to_float(r.get("best_fitness"), np.inf) for r in rows_sorted], dtype=float)

        finite_mask = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[finite_mask]
        ys = ys[finite_mask]
        if xs.size == 0:
            continue

        # Enforce non-increasing best fitness curve
        best = np.inf
        ys_best = []
        for y in ys:
            if y < best:
                best = y
            ys_best.append(best)
        ys = np.array(ys_best, dtype=float)

        x_max = float(np.max(xs))
        if x_max <= 0:
            continue
        x_norm = xs / x_max

        curves_by_method[method].append((x_norm, ys))

    return curves_by_method


def plot_fig5_convergence(curves_by_method: dict[str, list[tuple[np.ndarray, np.ndarray]]], out_path: Path) -> None:
    grid = np.linspace(0.0, 1.0, 120)
    fig, ax = plt.subplots(figsize=(10.8, 6.4), dpi=180)

    colors = {
        "grid": "#d62728",
        "pso": "#1f77b4",
        "cmaes": "#2ca02c",
    }

    any_curve = False
    for method, curves in curves_by_method.items():
        if not curves:
            continue
        samples = []
        for x, y in curves:
            if x.size == 0:
                continue
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]

            # Remove duplicate x for interpolation
            keep = np.ones_like(x_sorted, dtype=bool)
            keep[1:] = np.diff(x_sorted) > 1e-12
            x_unique = x_sorted[keep]
            y_unique = y_sorted[keep]
            if x_unique.size < 2:
                continue
            y_interp = np.interp(grid, x_unique, y_unique)
            samples.append(y_interp)

        if not samples:
            continue

        any_curve = True
        arr = np.vstack(samples)
        med = np.median(arr, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)

        ax.plot(grid, med, color=colors.get(method, None), linewidth=2.0, label=f"{method} median")
        ax.fill_between(grid, q25, q75, color=colors.get(method, None), alpha=0.22, linewidth=0)

    if not any_curve:
        raise RuntimeError("No valid convergence curves found in curves.csv")

    ax.set_xlabel("Consumed evaluation budget ratio")
    ax.set_ylabel("Best fitness so far (lower is better)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot E3 figure (Fig5 convergence) from run_exp3_budget outputs.")
    parser.add_argument("--exp-root", type=str, default="exp2_3/results/e3_budget")
    parser.add_argument("--methods", type=str, default="grid,pso,cmaes")
    parser.add_argument("--format", type=str, default="png", choices=["png", "svg"])
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    exp_root = Path(args.exp_root).expanduser().resolve()
    curves_csv = exp_root / "curves.csv"

    curve_rows = read_csv(curves_csv)
    curves_by_method = build_run_curves(curve_rows=curve_rows, methods=methods)

    fig_dir = exp_root / "figures"
    fig5 = fig_dir / f"fig5_convergence.{args.format}"
    plot_fig5_convergence(curves_by_method=curves_by_method, out_path=fig5)

    print(f"Saved Fig5: {fig5}")
    print(f"Figures dir: {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
