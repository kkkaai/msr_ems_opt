#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FONT_SIZE = 26
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
    }
)


X_TICKS = [20, 40, 60, 80, 100]
Y_TICKS = [2.0, 1.7, 1.5, 1.3, 1.0]  # top -> bottom

# Recorded directly from the two PNG heatmaps.
FIT_ERROR = np.array(
    [
        [0.1118, 0.1108, 0.1125, 0.1154, 0.1182],
        [0.1095, 0.1113, 0.1127, 0.1148, 0.1176],
        [0.1095, 0.1114, 0.1110, 0.1252, 0.1329],
        [0.1097, 0.1108, 0.1305, 0.1186, 0.1324],
        [0.1092, 0.1254, 0.1323, 0.1341, 0.1341],
    ],
    dtype=float,
)

INLIER_COUNT = np.array(
    [
        [5976, 5793, 5757, 5606, 5329],
        [5893, 5786, 5707, 5378, 5116],
        [5880, 5756, 5606, 4942, 3437],
        [5849, 5734, 5205, 2736, 2351],
        [5820, 4680, 2321, 2276, 2276],
    ],
    dtype=float,
)


def save_matrix_csv(matrix: np.ndarray, out_csv: Path, value_name: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma"] + [str(x) for x in X_TICKS])
        for y, row in zip(Y_TICKS, matrix):
            writer.writerow([y] + [f"{v:.4f}" if value_name == "fit_error" else f"{int(v)}" for v in row])


def _text_color(value: float, vmin: float, vmax: float, white_when: str = "low") -> str:
    if vmax <= vmin:
        return "white"
    norm = (value - vmin) / (vmax - vmin)
    if white_when == "high":
        return "white" if norm > 0.45 else "#222222"
    return "white" if norm < 0.62 else "#222222"


def plot_heatmap(
    matrix: np.ndarray,
    out_svg: Path,
    number_format: str,
    cmap: str,
    cbar_ticks: np.ndarray | None = None,
    text_white_when: str = "low",
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.8), dpi=200)
    n_rows, n_cols = matrix.shape
    x_edges = np.arange(0, n_cols + 1, dtype=float)
    y_edges = np.arange(0, n_rows + 1, dtype=float)

    # Use pcolormesh so SVG stays vector (no embedded raster image).
    mappable = ax.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap=cmap,
        shading="flat",
        edgecolors=(1, 1, 1, 0.18),
        linewidth=0.6,
    )
    cbar = fig.colorbar(mappable, ax=ax)
    if cbar_ticks is not None and cbar_ticks.size > 0:
        cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    ax.set_xticks(np.arange(len(X_TICKS)) + 0.5)
    ax.set_xticklabels([str(x) for x in X_TICKS])
    ax.set_yticks(np.arange(len(Y_TICKS)) + 0.5)
    ax.set_yticklabels([f"{y:g}" for y in Y_TICKS])
    ax.set_xlim(0.0, float(n_cols))
    ax.set_ylim(0.0, float(n_rows))
    ax.invert_yaxis()

    vmin = float(np.min(matrix))
    vmax = float(np.max(matrix))
    for i in range(n_rows):
        for j in range(n_cols):
            val = float(matrix[i, j])
            txt = format(val, number_format)
            ax.text(
                j + 0.5,
                i + 0.5,
                txt,
                ha="center",
                va="center",
                color=_text_color(val, vmin, vmax, white_when=text_white_when),
                fontsize=FONT_SIZE - 2,
            )

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)


def main() -> None:
    out_dir = Path("exp1/results/recovered_heatmaps_from_png").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_matrix_csv(FIT_ERROR, out_dir / "fit_error_heatmap_values.csv", "fit_error")
    save_matrix_csv(INLIER_COUNT, out_dir / "inlier_count_heatmap_values.csv", "inlier_count")

    plot_heatmap(
        FIT_ERROR,
        out_dir / "fit_error_heatmap.svg",
        ".4f",
        cmap="YlGnBu_r",
        cbar_ticks=np.arange(0.110, 0.131, 0.005),
        text_white_when="low",
    )
    plot_heatmap(
        INLIER_COUNT,
        out_dir / "inlier_count_heatmap.svg",
        ".0f",
        cmap="YlGnBu",
        cbar_ticks=np.arange(2500, 5501, 500),
        text_white_when="high",
    )

    print("Saved:")
    print(f"  - {out_dir / 'fit_error_heatmap_values.csv'}")
    print(f"  - {out_dir / 'inlier_count_heatmap_values.csv'}")
    print(f"  - {out_dir / 'fit_error_heatmap.svg'}")
    print(f"  - {out_dir / 'inlier_count_heatmap.svg'}")


if __name__ == "__main__":
    main()
