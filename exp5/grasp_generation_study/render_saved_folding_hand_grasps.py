#!/usr/bin/env python3
"""Render saved folding-hand grasp JSON without re-running optimization."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "msr_matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import numpy as np

from grasp_generator.folding_hand import FoldingHandPose
from grasp_generator.superquadric_geometry import SuperquadricModel
from run_generate_folding_hand_grasps import DEFAULT_INPUT_ROOT, DEFAULT_OUTPUT_ROOT, render
from grasp_generator.folding_hand import FoldingHandRightKinematics


def pose_from_dict(raw: dict) -> FoldingHandPose:
    controls = np.asarray(list(raw["controls"].values()), dtype=float)
    return FoldingHandPose(
        translation=np.asarray(raw["translation_m"], dtype=float),
        rotation_matrix=np.asarray(raw["rotation_matrix"], dtype=float),
        controls=controls,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--result-json", type=Path, default=DEFAULT_OUTPUT_ROOT / "selected_folding_hand_grasps.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "visualizations")
    args = parser.parse_args()

    rows = json.loads(args.result_json.read_text(encoding="utf-8"))
    hand = FoldingHandRightKinematics.default()
    by_object: dict[str, list[dict]] = {}
    for raw in rows:
        by_object.setdefault(raw["object"], []).append(raw)

    for object_name, object_rows in sorted(by_object.items()):
        model = SuperquadricModel.from_json(args.input_root / f"{object_name}_20cm.json")
        object_dir = args.output_root / object_name
        object_dir.mkdir(parents=True, exist_ok=True)
        for old_png in object_dir.glob("*.png"):
            old_png.unlink()
        for index, raw in enumerate(object_rows, start=1):
            row = {
                "object": raw["object"],
                "target_sq": int(raw["target_sq"].replace("SQ", "")),
                "pregrasp": pose_from_dict(raw["pregrasp"]),
                "grasp": pose_from_dict(raw["grasp"]),
            }
            render(model, hand, row, object_dir / f"grasp_{index:02d}_{raw['target_sq']}.png")
        print(f"{object_name}: rendered {len(object_rows)}")


if __name__ == "__main__":
    main()
