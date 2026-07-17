#!/usr/bin/env python3
import argparse
import random
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Randomly sample KIT 25k PLY models and run run_hierarchical_ems_pso_3params.py in batch."
    )
    p.add_argument("--input-dir", type=str, default="data/KIT_ObjectModels_25k_ply")
    p.add_argument("--output-dir", type=str, default="test")
    p.add_argument("--num-models", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # Runtime/quality trade-off
    p.add_argument("--swarmsize", type=int, default=6)
    p.add_argument("--maxiter", type=int, default=8)
    p.add_argument("--MaxLayer", type=int, default=3)

    # Forwarded hyperparameters
    p.add_argument("--OutlierRatioMin", type=float, default=0.1)
    p.add_argument("--OutlierRatioMax", type=float, default=0.95)
    p.add_argument("--EpsMin", type=float, default=1.0)
    p.add_argument("--EpsMax", type=float, default=3.0)
    p.add_argument("--MinPointsMin", type=float, default=10.0)
    p.add_argument("--MinPointsMax", type=float, default=120.0)

    p.add_argument("--AdaptiveUpperBound", type=str, default="True")
    p.add_argument("--Rescale", type=str, default="False")
    p.add_argument("--Sigma", type=float, default=0.3)
    p.add_argument("--TauIn", type=float, default=0.1)
    p.add_argument("--TauSplit", type=float, default=0.8)

    p.add_argument("--arcLength", type=float, default=0.2)
    p.add_argument("--pointSize", type=float, default=0.001)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = (repo_root / args.input_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    ply_files = sorted(input_dir.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in {input_dir}")

    if args.num_models > len(ply_files):
        raise ValueError(f"num-models={args.num_models} exceeds available files={len(ply_files)}")

    random.seed(args.seed)
    sampled = random.sample(ply_files, args.num_models)

    output_dir.mkdir(parents=True, exist_ok=True)

    runner = (repo_root / "src" / "run_hierarchical_ems_pso_3params.py").resolve()
    summary_lines = []

    for idx, ply in enumerate(sampled, start=1):
        stem = ply.stem
        json_out = output_dir / f"{idx:02d}_{stem}.json"
        img_out = output_dir / f"{idx:02d}_{stem}.png"
        log_out = output_dir / f"{idx:02d}_{stem}.log"

        cmd = [
            sys.executable,
            str(runner),
            str(ply),
            "--out", str(json_out),
            "--saveImage", str(img_out),
            "--runtime",
            "--result",
            "--swarmsize", str(args.swarmsize),
            "--maxiter", str(args.maxiter),
            "--MaxLayer", str(args.MaxLayer),
            "--OutlierRatioMin", str(args.OutlierRatioMin),
            "--OutlierRatioMax", str(args.OutlierRatioMax),
            "--EpsMin", str(args.EpsMin),
            "--EpsMax", str(args.EpsMax),
            "--MinPointsMin", str(args.MinPointsMin),
            "--MinPointsMax", str(args.MinPointsMax),
            "--AdaptiveUpperBound", str(args.AdaptiveUpperBound),
            "--Rescale", str(args.Rescale),
            "--Sigma", str(args.Sigma),
            "--TauIn", str(args.TauIn),
            "--TauSplit", str(args.TauSplit),
            "--arcLength", str(args.arcLength),
            "--pointSize", str(args.pointSize),
        ]

        print(f"[{idx:02d}/{len(sampled):02d}] Running: {ply.name}")
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        log_out.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

        if proc.returncode == 0:
            status = "OK"
        else:
            status = f"FAIL({proc.returncode})"

        summary_lines.append(f"[{idx:02d}] {status} {ply.name} | json={json_out.name} | png={img_out.name}")
        print(summary_lines[-1])

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nBatch finished. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
