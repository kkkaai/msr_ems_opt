#!/usr/bin/env python3
import argparse
import csv
import json
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


def _ok_rows(per_rows: list[dict], method: str) -> list[dict]:
    return [r for r in per_rows if r.get("method") == method and r.get("status") == "OK"]


def _accuracy_proxy_from_fitness(f: float) -> float:
    if not np.isfinite(f):
        return 0.0
    return float(1.0 / (1.0 + max(0.0, f)))


def plot_fig4_pareto(
    per_rows: list[dict],
    out_path: Path,
    methods: list[str],
    x_max_s: float | None = None,
    x_tick_end_s: float | None = None,
    x_tick_step_s: float = 100.0,
) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 6.4), dpi=180)
    colors = {
        "grid": "#d62728",
        "pso": "#1f77b4",
        "cmaes": "#2ca02c",
    }

    all_x: list[float] = []
    all_y: list[float] = []

    for method in methods:
        rows = _ok_rows(per_rows, method)
        if not rows:
            continue
        runtimes_ms = np.array([to_float(r.get("runtime_ms_med")) for r in rows], dtype=float)
        fitness = np.array([to_float(r.get("fitness_med")) for r in rows], dtype=float)
        acc = np.array([_accuracy_proxy_from_fitness(v) for v in fitness], dtype=float)
        runtimes_s = runtimes_ms / 1000.0

        finite_mask = np.isfinite(runtimes_s) & np.isfinite(acc)
        runtimes_s = runtimes_s[finite_mask]
        acc = acc[finite_mask]
        if runtimes_s.size == 0:
            continue

        mean_x = float(np.mean(runtimes_s))
        mean_y = float(np.mean(acc))
        all_x.extend(runtimes_s.tolist())
        all_x.append(mean_x)
        all_y.extend(acc.tolist())
        all_y.append(mean_y)

        ax.scatter(
            runtimes_s,
            acc,
            s=36,
            alpha=0.9,
            color=colors.get(method, None),
            label=f"{method} (per object)",
        )
        ax.scatter(
            [mean_x],
            [mean_y],
            s=360,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
            color=colors.get(method, None),
            label=f"{method} mean",
        )

    ax.set_xlabel("Runtime per object (s), lower is better")
    ax.set_ylabel("Accuracy proxy = 1/(1+F), higher is better")
    ax.grid(alpha=0.25)
    if all_x:
        if x_max_s is not None and np.isfinite(float(x_max_s)) and float(x_max_s) > 0.0:
            ax.set_xlim(0.0, float(x_max_s))
        else:
            x_min = float(np.min(all_x))
            x_max = float(np.max(all_x))
            x_span = max(1e-6, x_max - x_min)
            x_pad_left = 0.08 * x_span
            # Reserve in-axes whitespace on the right for the legend.
            x_pad_right = 0.35 * x_span
            ax.set_xlim(max(0.0, x_min - x_pad_left), x_max + x_pad_right)
    if all_y:
        y_min = float(np.min(all_y))
        y_max = float(np.max(all_y))
        y_span = max(1e-4, y_max - y_min)
        y_pad = max(0.01, 0.10 * y_span)
        y_lo = max(0.0, y_min - y_pad)
        y_hi = min(1.02, y_max + y_pad)
        if y_hi - y_lo < 0.05:
            mid = 0.5 * (y_lo + y_hi)
            y_lo = max(0.0, mid - 0.025)
            y_hi = min(1.0, mid + 0.025)
        ax.set_ylim(y_lo, y_hi)
    if x_tick_end_s is not None and np.isfinite(float(x_tick_end_s)) and float(x_tick_step_s) > 0.0:
        ax.set_xticks(np.arange(0.0, float(x_tick_end_s) + 1e-9, float(x_tick_step_s)))
    ax.ticklabel_format(style="plain", axis="x")
    ax.legend(
        ncols=1,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        framealpha=0.92,
        fontsize=22,
        borderpad=0.45,
        labelspacing=0.35,
        handletextpad=0.45,
        columnspacing=0.6,
        markerscale=1.0,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.93)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def choose_representative_objects(
    per_rows: list[dict],
    method: str,
) -> list[tuple[str, str, float]]:
    rows = _ok_rows(per_rows, method)
    if not rows:
        return []

    rows_sorted = sorted(rows, key=lambda r: to_float(r.get("fitness_med"), np.inf))
    n = len(rows_sorted)
    best = rows_sorted[0]
    median = rows_sorted[n // 2]
    worst = rows_sorted[-1]

    return [
        ("best", str(best.get("object")), to_float(best.get("fitness_med"), np.nan)),
        ("median", str(median.get("object")), to_float(median.get("fitness_med"), np.nan)),
        ("worst", str(worst.get("object")), to_float(worst.get("fitness_med"), np.nan)),
    ]


def _resolve_image_path(img: str) -> Path | None:
    p = Path(img).expanduser().resolve()
    if p.exists():
        return p

    # Backward-compatible fallback:
    # some historical runs saved image path as .svg in runs_raw.csv,
    # while actual rendered files were later regenerated as .png.
    for ext in [".png", ".jpg", ".jpeg", ".eps", ".svg"]:
        alt = p.with_suffix(ext)
        if alt.exists():
            return alt
    return None


def build_image_index(runs_rows: list[dict], methods: list[str], image_seed: int) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for r in runs_rows:
        method = str(r.get("method", ""))
        obj = str(r.get("object", ""))
        if method not in methods:
            continue
        if r.get("status") != "OK":
            continue
        seed = int(float(r.get("seed", -1))) if str(r.get("seed", "")).strip() else -1
        if seed != int(image_seed):
            continue
        img = str(r.get("image", "")).strip()
        if not img:
            continue
        p = _resolve_image_path(img)
        if p is not None:
            out[(method, obj)] = p
    return out


def _metric_text(per_rows: list[dict], method: str, obj: str) -> str:
    rows = [r for r in per_rows if r.get("method") == method and r.get("object") == obj and r.get("status") == "OK"]
    if not rows:
        return f"{obj}\n{method}\nno metrics"
    r = rows[0]
    return (
        f"{obj}\n{method}\n"
        f"F={to_float(r.get('fitness_med')):.4f}\n"
        f"d_fit={to_float(r.get('distance_fit_med')):.4f}\n"
        f"cov={to_float(r.get('coverage_ratio_med')):.3f}\n"
        f"rt={to_float(r.get('runtime_ms_med')):.1f}ms"
    )


def plot_fig3_representative(
    per_rows: list[dict],
    runs_rows: list[dict],
    methods: list[str],
    reference_method: str,
    image_seed: int,
    out_path: Path,
    selected_out_txt: Path,
    selected_out_json: Path,
) -> None:
    reps = choose_representative_objects(per_rows, reference_method)
    if not reps:
        raise RuntimeError(f"No successful rows for reference method: {reference_method}")

    image_idx = build_image_index(runs_rows, methods, image_seed=image_seed)

    n_rows = len(reps)
    n_cols = len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), dpi=180)
    axes_arr = np.array(axes, dtype=object)
    if axes_arr.ndim == 0:
        axes_arr = axes_arr.reshape(1, 1)
    elif axes_arr.ndim == 1:
        if n_rows == 1:
            axes_arr = axes_arr.reshape(1, n_cols)
        elif n_cols == 1:
            axes_arr = axes_arr.reshape(n_rows, 1)

    selected_payload = []
    for r_idx, (tag, obj, fit) in enumerate(reps):
        selected_payload.append({"tag": tag, "object": obj, "reference_fitness": float(fit)})
        for c_idx, method in enumerate(methods):
            ax = axes_arr[r_idx, c_idx]
            img_path = image_idx.get((method, obj))
            if img_path is not None and img_path.exists():
                try:
                    img = plt.imread(str(img_path))
                    ax.imshow(img)
                    ax.axis("off")
                except Exception:
                    ax.text(0.5, 0.5, _metric_text(per_rows, method, obj), ha="center", va="center", fontsize=9)
                    ax.set_axis_off()
            else:
                ax.text(0.5, 0.5, _metric_text(per_rows, method, obj), ha="center", va="center", fontsize=9)
                ax.set_axis_off()

            if r_idx == 0:
                ax.set_title(method, fontsize=11)
            if c_idx == 0:
                ax.text(-0.02, 0.5, tag, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=11)

    fig.suptitle("Fig3: Representative Objects (best / median / worst)", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    selected_out_txt.parent.mkdir(parents=True, exist_ok=True)
    selected_out_txt.write_text("\n".join([f"{x['tag']}: {x['object']}" for x in selected_payload]) + "\n", encoding="utf-8")
    selected_out_json.write_text(
        json.dumps(
            {
                "reference_method": reference_method,
                "image_seed": int(image_seed),
                "selected": selected_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot E2 figures (Fig3/Fig4) from run_exp2_multi outputs.")
    parser.add_argument("--exp-root", type=str, default="exp2_3/results/e2_multi")
    parser.add_argument("--methods", type=str, default="grid,pso,cmaes")
    parser.add_argument("--format", type=str, default="png", choices=["png", "svg"])
    parser.add_argument("--reference-method", type=str, default="pso")
    parser.add_argument("--image-seed", type=int, default=0)
    parser.add_argument(
        "--fig4-x-max-s",
        type=float,
        default=-1.0,
        help="If >0, force Fig4 x-axis max in seconds; otherwise auto-scale from data.",
    )
    parser.add_argument(
        "--fig4-x-tick-end-s",
        type=float,
        default=-1.0,
        help="If >0, force Fig4 x ticks to stop at this value in seconds.",
    )
    parser.add_argument(
        "--fig4-x-tick-step-s",
        type=float,
        default=100.0,
        help="Fig4 x-axis tick spacing in seconds when --fig4-x-tick-end-s is set.",
    )
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    exp_root = Path(args.exp_root).expanduser().resolve()
    per_csv = exp_root / "per_object.csv"
    runs_csv = exp_root / "runs_raw.csv"

    per_rows = read_csv(per_csv)
    runs_rows = read_csv(runs_csv)

    fig_dir = exp_root / "figures"
    fig4 = fig_dir / f"fig4_pareto.{args.format}"
    plot_fig4_pareto(
        per_rows=per_rows,
        out_path=fig4,
        methods=methods,
        x_max_s=(float(args.fig4_x_max_s) if float(args.fig4_x_max_s) > 0.0 else None),
        x_tick_end_s=(float(args.fig4_x_tick_end_s) if float(args.fig4_x_tick_end_s) > 0.0 else None),
        x_tick_step_s=float(args.fig4_x_tick_step_s),
    )

    fig3 = fig_dir / f"fig3_representative.{args.format}"
    sel_txt = fig_dir / "fig3_selected_objects.txt"
    sel_json = fig_dir / "fig3_selected_objects.json"
    try:
        plot_fig3_representative(
            per_rows=per_rows,
            runs_rows=runs_rows,
            methods=methods,
            reference_method=args.reference_method,
            image_seed=args.image_seed,
            out_path=fig3,
            selected_out_txt=sel_txt,
            selected_out_json=sel_json,
        )
        print(f"Saved Fig3: {fig3}")
        print(f"Saved Fig3 selected list: {sel_txt}")
    except Exception as exc:
        print(f"Skip Fig3: {exc}")

    print(f"Saved Fig4: {fig4}")
    print(f"Figures dir: {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
