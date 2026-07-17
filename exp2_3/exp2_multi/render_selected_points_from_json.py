#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import render_images_from_json as rij


def parse_objects_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render point-cloud-only images for selected objects from E2 JSON results."
    )
    parser.add_argument("--results-root", type=str, required=True, help="e.g. exp2_3/results/e2_multi_8x4_f005_c001")
    parser.add_argument("--method", type=str, default="grid", choices=["grid", "pso", "cmaes"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objects", type=str, required=True, help="Comma-separated object basenames, e.g. CokePlasticLarge,Deodorant")
    parser.add_argument("--output", type=str, required=True, help="Output directory for rendered point cloud PNGs")
    parser.add_argument("--point-size", type=float, default=0.0, help="<=0 enables auto point size based on point-cloud scale.")
    parser.add_argument("--offscreen", type=rij.str2bool, default=False)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_root = (repo_root / args.results_root).resolve()
    json_dir = results_root / "json" / args.method
    out_dir = (repo_root / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    objects = parse_objects_csv(args.objects)
    if not objects:
        raise ValueError("No objects parsed from --objects.")

    ok = 0
    fail = 0
    for obj in objects:
        stem = f"{obj}_25k__seed{args.seed}"
        json_path = json_dir / f"{stem}.json"
        out_path = out_dir / f"{stem}.png"
        if not json_path.exists():
            print(f"MISSING JSON: {json_path}")
            fail += 1
            continue
        if (not args.overwrite) and out_path.exists():
            print(f"SKIP existing: {out_path}")
            continue

        # Validate JSON has input_path early for cleaner errors.
        data = json.loads(json_path.read_text(encoding="utf-8"))
        input_path = Path(str(data.get("input_path", "")))
        if not input_path.exists():
            print(f"MISSING input_path in JSON: {json_path} -> {input_path}")
            fail += 1
            continue

        success, err = rij._render_one_isolated(
            json_path=json_path,
            image_path=out_path,
            visualize_mode="points_only",
            arc_length=0.2,
            point_size=float(args.point_size),
            offscreen=bool(args.offscreen),
            timeout_sec=int(args.timeout_sec),
        )
        if success:
            ok += 1
            print(f"OK   {obj} -> {out_path}")
        else:
            fail += 1
            print(f"FAIL {obj}: {err}")

    print("Done")
    print(f"- ok  : {ok}")
    print(f"- fail: {fail}")
    print(f"- out : {out_dir}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
