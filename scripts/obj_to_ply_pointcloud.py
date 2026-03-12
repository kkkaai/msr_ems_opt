#!/usr/bin/env python3
"""
Batch-convert OBJ meshes to sampled point clouds (PLY) with Open3D.

Default input:
  data/KIT_ObjectModels_25k_obj

Default output:
  data/KIT_ObjectModels_25k_ply

Examples:
  python scripts/obj_to_ply_pointcloud.py

  python scripts/obj_to_ply_pointcloud.py \
      --input-dir data/KIT_ObjectModels_25k_obj \
      --output-dir data/KIT_ObjectModels_25k_ply \
      --num-points 5000 \
      --method poisson

  python scripts/obj_to_ply_pointcloud.py \
      --method uniform \
      --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch sample point clouds from OBJ meshes and save as PLY."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/KIT_ObjectModels_25k_obj"),
        help="Directory containing OBJ files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/KIT_ObjectModels_25k_ply"),
        help="Directory to save generated PLY files.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=5000,
        help="Number of sampled points per mesh.",
    )
    parser.add_argument(
        "--method",
        choices=["poisson", "uniform"],
        default="poisson",
        help="Sampling method on mesh surface.",
    )
    parser.add_argument(
        "--init-factor",
        type=int,
        default=5,
        help="Open3D Poisson sampling init factor (only for --method poisson).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Optional voxel downsample size after sampling (0 disables).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PLY files.",
    )
    return parser.parse_args()


def discover_obj_files(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.rglob("*.obj") if p.is_file())


def sample_one_mesh(
    obj_path: Path,
    out_path: Path,
    num_points: int,
    method: str,
    init_factor: int,
    voxel_size: float,
) -> Tuple[bool, str]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(obj_path))
    if mesh.is_empty():
        return False, f"{obj_path}: empty mesh"
    if len(mesh.triangles) == 0:
        return False, f"{obj_path}: no triangles"

    if method == "poisson":
        pcd = mesh.sample_points_poisson_disk(
            number_of_points=num_points,
            init_factor=init_factor,
        )
    else:
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(out_path), pcd)
    if not ok:
        return False, f"{obj_path}: failed to write {out_path}"
    return True, f"{obj_path} -> {out_path}"


def main() -> int:
    args = parse_args()

    if args.num_points <= 0:
        print("ERROR: --num-points must be > 0", file=sys.stderr)
        return 2
    if args.init_factor <= 0:
        print("ERROR: --init-factor must be > 0", file=sys.stderr)
        return 2
    if args.voxel_size < 0:
        print("ERROR: --voxel-size must be >= 0", file=sys.stderr)
        return 2
    if not args.input_dir.exists():
        print(f"ERROR: input directory not found: {args.input_dir}", file=sys.stderr)
        return 2

    obj_files = discover_obj_files(args.input_dir)
    print(f"Input dir : {args.input_dir.resolve()}")
    print(f"Output dir: {args.output_dir.resolve()}")
    print(f"OBJ files : {len(obj_files)}")
    print(f"Method    : {args.method}")
    print(f"Points    : {args.num_points}")
    if args.method == "poisson":
        print(f"Init fac. : {args.init_factor}")
    if args.voxel_size > 0:
        print(f"Voxel size: {args.voxel_size}")

    if not obj_files:
        print("No OBJ files found.")
        return 0

    ok_count = 0
    skip_count = 0
    fail_count = 0

    for idx, obj_path in enumerate(obj_files, start=1):
        rel = obj_path.relative_to(args.input_dir)
        out_path = args.output_dir / rel.with_suffix(".ply")

        if out_path.exists() and not args.overwrite:
            skip_count += 1
            print(f"[{idx:4d}/{len(obj_files)}] SKIP {out_path}")
            continue

        ok, msg = sample_one_mesh(
            obj_path=obj_path,
            out_path=out_path,
            num_points=args.num_points,
            method=args.method,
            init_factor=args.init_factor,
            voxel_size=args.voxel_size,
        )
        if ok:
            ok_count += 1
            print(f"[{idx:4d}/{len(obj_files)}] OK   {msg}")
        else:
            fail_count += 1
            print(f"[{idx:4d}/{len(obj_files)}] FAIL {msg}")

    print("\nSummary")
    print(f"- ok   : {ok_count}")
    print(f"- skip : {skip_count}")
    print(f"- fail : {fail_count}")
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

