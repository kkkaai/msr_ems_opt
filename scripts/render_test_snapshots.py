#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_ply_xyz(path: Path) -> np.ndarray:
    try:
        from plyfile import PlyData
    except Exception as exc:
        raise RuntimeError("plyfile is required: pip install plyfile") from exc
    with path.open("rb") as f:
        ply = PlyData.read(f)
    v = ply["vertex"]
    return np.column_stack((v["x"], v["y"], v["z"]))


def main() -> int:
    p = argparse.ArgumentParser(description="Render static result snapshots for test JSON outputs.")
    p.add_argument("--test-dir", type=str, default="test")
    p.add_argument("--ply-dir", type=str, default="data/KIT_ObjectModels_25k_ply")
    p.add_argument("--max-points", type=int, default=12000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    repo_root = Path(__file__).resolve().parents[1]
    test_dir = (repo_root / args.test_dir).resolve()
    ply_dir = (repo_root / args.ply_dir).resolve()

    json_files = sorted(test_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {test_dir}")

    for jf in json_files:
        with jf.open("r", encoding="utf-8") as f:
            data = json.load(f)

        input_path = data.get("input_path", "")
        ply_path = Path(input_path) if input_path else (ply_dir / (jf.stem.split("_", 1)[1] + ".ply"))
        if not ply_path.exists():
            print(f"[SKIP] PLY not found for {jf.name}: {ply_path}")
            continue

        points = read_ply_xyz(ply_path)
        n = len(points)
        if n > args.max_points:
            idx = rng.choice(n, size=args.max_points, replace=False)
            pts = points[idx]
        else:
            pts = points

        fitness = data.get("best_fitness", float("nan"))
        num_sq = data.get("summary", {}).get("num_superquadrics", "?")

        fig = plt.figure(figsize=(6, 6), dpi=150)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.2, c="#1f77b4", alpha=0.85)
        ax.set_title(f"{ply_path.stem}\\nfitness={fitness:.4f}, SQ={num_sq}", fontsize=9)
        ax.set_axis_off()
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        out_png = jf.with_suffix(".png")
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
        print(f"[OK] {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
