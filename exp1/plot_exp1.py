#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v, default=np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def plot_boxplots(per_object_rows: list[dict], out_path: Path, methods: list[str]) -> None:
    rows_ok = [r for r in per_object_rows if r.get("status") == "OK"]
    fit_data = []
    rt_data = []
    for m in methods:
        rows_m = [r for r in rows_ok if r.get("method") == m]
        fit_data.append([to_float(r.get("best_fitness_med")) for r in rows_m])
        rt_data.append([to_float(r.get("runtime_ms_med")) for r in rows_m])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=180)
    axes[0].boxplot(fit_data, labels=methods, showfliers=False)
    axes[0].set_title("Fit Error by Method")
    axes[0].set_ylabel("fit error (lower better)")
    axes[0].grid(alpha=0.25)

    axes[1].boxplot(rt_data, labels=methods, showfliers=False)
    axes[1].set_title("Runtime by Method")
    axes[1].set_ylabel("runtime (ms)")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def choose_heatmap_json(
    runs_rows: list[dict],
    heatmap_object: str | None,
    method: str = "grid",
) -> Path | None:
    candidates = [r for r in runs_rows if r.get("method") == method and r.get("status") == "OK"]
    if heatmap_object:
        candidates = [r for r in candidates if r.get("object") == heatmap_object]
    if not candidates:
        return None
    return Path(candidates[0]["json"]).expanduser().resolve()


def plot_heatmap(grid_json: Path, out_path: Path) -> None:
    data = json.loads(grid_json.read_text(encoding="utf-8"))
    grid_surface = data.get("grid_surface", {})
    x = np.asarray(grid_surface.get("outlier_ratio_axis", []), dtype=float)
    y = np.asarray(grid_surface.get("sigma_axis", []), dtype=float)
    z = np.asarray(grid_surface.get("fitness", []), dtype=float)
    if x.size == 0 or y.size == 0 or z.size == 0:
        raise RuntimeError(f"Grid surface missing in {grid_json}")

    # z shape expected as [len(x), len(y)], transpose for imshow with y vertical
    z_plot = z.T
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=180)
    im = ax.imshow(
        z_plot,
        origin="lower",
        aspect="auto",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="viridis",
    )
    ax.set_xlabel("OutlierRatio")
    ax.set_ylabel("Sigma")
    ax.set_title(f"Parameter Sensitivity Heatmap ({grid_json.stem})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("fit error")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def select_heatmap_objects(
    runs_rows: list[dict],
    method: str,
    num_objects: int,
    seed: int,
) -> list[tuple[str, Path]]:
    candidates = [r for r in runs_rows if r.get("method") == method and r.get("status") == "OK"]
    obj_to_json: dict[str, Path] = {}
    for r in candidates:
        obj = str(r.get("object", "")).strip()
        if not obj or obj in obj_to_json:
            continue
        obj_to_json[obj] = Path(str(r["json"])).expanduser().resolve()

    objects = sorted(obj_to_json.keys())
    if not objects:
        return []

    k = min(int(num_objects), len(objects))
    rng = np.random.default_rng(int(seed))
    selected = list(rng.choice(objects, size=k, replace=False))
    selected_pairs = [(obj, obj_to_json[obj]) for obj in selected]
    return selected_pairs


def plot_heatmap_multiples(
    selected_pairs: list[tuple[str, Path]],
    out_path: Path,
    rows: int = 3,
    cols: int = 3,
    normalize_mode: str = "none",
) -> None:
    if not selected_pairs:
        raise RuntimeError("No selected heatmap objects to plot.")

    heatmaps: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    global_min = np.inf
    global_max = -np.inf

    for obj_name, grid_json in selected_pairs:
        data = json.loads(grid_json.read_text(encoding="utf-8"))
        grid_surface = data.get("grid_surface", {})
        x = np.asarray(grid_surface.get("outlier_ratio_axis", []), dtype=float)
        y = np.asarray(grid_surface.get("sigma_axis", []), dtype=float)
        z = np.asarray(grid_surface.get("fitness", []), dtype=float)
        if x.size == 0 or y.size == 0 or z.size == 0:
            continue
        finite = z[np.isfinite(z)]
        if finite.size == 0:
            continue
        z_plot = z.copy()
        if normalize_mode == "per_object":
            local_min = float(np.min(finite))
            local_max = float(np.max(finite))
            denom = local_max - local_min
            if denom <= 1e-12:
                z_plot = np.zeros_like(z_plot, dtype=float)
            else:
                z_plot = (z_plot - local_min) / denom
            global_min = 0.0
            global_max = 1.0
        else:
            local_min = float(np.min(finite))
            local_max = float(np.max(finite))
            global_min = min(global_min, local_min)
            global_max = max(global_max, local_max)
        heatmaps.append((obj_name, x, y, z_plot))

    if not heatmaps:
        raise RuntimeError("Selected objects do not contain valid grid_surface heatmaps.")

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.9 * cols, 4.1 * rows),
        dpi=180,
        constrained_layout=True,
    )
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    mappable = None

    for idx, ax in enumerate(axes_arr.flat):
        if idx >= len(heatmaps):
            ax.axis("off")
            continue
        obj_name, x, y, z = heatmaps[idx]
        z_plot = z.T
        im = ax.imshow(
            z_plot,
            origin="lower",
            aspect="auto",
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap="viridis",
            vmin=global_min,
            vmax=global_max,
        )
        mappable = im
        ax.set_box_aspect(1)
        ax.set_title(obj_name, fontsize=9)
        ax.set_xlabel("OutlierRatio", fontsize=8)
        ax.set_ylabel("Sigma", fontsize=8)
        ax.tick_params(labelsize=7)

    if mappable is not None:
        cbar = fig.colorbar(
            mappable,
            ax=axes_arr.ravel().tolist(),
            location="right",
            fraction=0.03,
            pad=0.02,
            shrink=0.96,
        )
        if normalize_mode == "per_object":
            cbar.set_label("normalized fit error [0,1]", fontsize=9)
        else:
            cbar.set_label("fit error", fontsize=9)

    if normalize_mode == "per_object":
        title = "Parameter Sensitivity Heatmaps (Multi-object, per-object normalized)"
    else:
        title = "Parameter Sensitivity Heatmaps (Multi-object, absolute scale)"
    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate E1 figures (Fig1 boxplots, Fig2 heatmap).")
    parser.add_argument("--exp-root", type=str, default="exp1/results/e1_single")
    parser.add_argument("--methods", type=str, default="grid,pso,cmaes")
    parser.add_argument("--heatmap-object", type=str, default="")
    parser.add_argument("--format", type=str, default="png", choices=["png", "svg"])
    parser.add_argument("--heatmap-method", type=str, default="grid")
    parser.add_argument("--multi-heatmap-num", type=int, default=9)
    parser.add_argument("--multi-heatmap-seed", type=int, default=42)
    parser.add_argument("--multi-heatmap-rows", type=int, default=3)
    parser.add_argument("--multi-heatmap-cols", type=int, default=3)
    parser.add_argument(
        "--multi-heatmap-normalize",
        type=str,
        default="none",
        choices=["none", "per_object"],
        help="Normalization for Fig3 heatmap wall.",
    )
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    exp_root = Path(args.exp_root).expanduser().resolve()
    per_object_csv = exp_root / "per_object.csv"
    runs_csv = exp_root / "runs_raw.csv"
    fig_dir = exp_root / "figures"

    per_rows = read_csv(per_object_csv)
    runs_rows = read_csv(runs_csv)

    fig1 = fig_dir / f"fig1_boxplots.{args.format}"
    plot_boxplots(per_rows, fig1, methods)

    heatmap_object = args.heatmap_object.strip() or None
    grid_json = choose_heatmap_json(runs_rows, heatmap_object=heatmap_object, method=args.heatmap_method)
    if grid_json is not None:
        fig2 = fig_dir / f"fig2_heatmap.{args.format}"
        plot_heatmap(grid_json, fig2)
        print(f"Saved Fig2: {fig2}")
    else:
        print("Skip Fig2: no successful grid JSON found.")

    num_slots = int(args.multi_heatmap_rows) * int(args.multi_heatmap_cols)
    num_to_select = min(int(args.multi_heatmap_num), num_slots)
    selected_pairs = select_heatmap_objects(
        runs_rows=runs_rows,
        method=args.heatmap_method,
        num_objects=num_to_select,
        seed=args.multi_heatmap_seed,
    )
    if selected_pairs:
        suffix = "_norm_per_object" if args.multi_heatmap_normalize == "per_object" else "_abs"
        fig3 = fig_dir / (
            f"fig3_heatmap_wall_{args.multi_heatmap_rows}x{args.multi_heatmap_cols}{suffix}.{args.format}"
        )
        plot_heatmap_multiples(
            selected_pairs=selected_pairs,
            out_path=fig3,
            rows=args.multi_heatmap_rows,
            cols=args.multi_heatmap_cols,
            normalize_mode=args.multi_heatmap_normalize,
        )
        selected_txt = fig_dir / (
            f"fig3_selected_objects_seed{args.multi_heatmap_seed}.txt"
        )
        selected_json = fig_dir / (
            f"fig3_selected_objects_seed{args.multi_heatmap_seed}.json"
        )
        selected_txt.write_text(
            "\n".join([obj for obj, _ in selected_pairs]) + "\n",
            encoding="utf-8",
        )
        selected_json.write_text(
            json.dumps(
                {
                    "seed": int(args.multi_heatmap_seed),
                    "method": args.heatmap_method,
                    "num_requested": int(args.multi_heatmap_num),
                    "num_slots": int(num_slots),
                    "num_effective_requested": int(num_to_select),
                    "num_selected": int(len(selected_pairs)),
                    "objects": [
                        {"object": obj, "json": str(path)}
                        for obj, path in selected_pairs
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved Fig3: {fig3}")
        print(f"Saved selected objects: {selected_txt}")
        print(f"Saved selected objects json: {selected_json}")
    else:
        print("Skip Fig3: no successful heatmap candidates found.")

    print(f"Saved Fig1: {fig1}")
    print(f"Figures dir: {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
