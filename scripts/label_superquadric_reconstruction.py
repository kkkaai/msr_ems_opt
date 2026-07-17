#!/usr/bin/env python3
"""
Interactive labeling for KIT point clouds:
- Press '1': single-superquadric reconstruction
- Press '2': multi-superquadric reconstruction
- Press '3': special

Writes one CSV file under data/ by default:
- data/kit_superquadric_labels.csv

CSV format:
  index,path,label
  0,Example_25k.ply,single
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Set


LABEL_SINGLE = "single"
LABEL_MULTI = "multi"
LABEL_SPECIAL = "special"


class KeyDecision:
    def __init__(self) -> None:
        self.choice: str | None = None

    def on_single(self, vis) -> bool:
        self.choice = LABEL_SINGLE
        vis.close()
        return False

    def on_multi(self, vis) -> bool:
        self.choice = LABEL_MULTI
        vis.close()
        return False

    def on_special(self, vis) -> bool:
        self.choice = LABEL_SPECIAL
        vis.close()
        return False

    def on_quit(self, vis) -> bool:
        self.choice = "quit"
        vis.close()
        return False

    def on_undo(self, vis) -> bool:
        self.choice = "undo"
        vis.close()
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually label KIT point clouds as single or multi superquadric."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/KIT_ObjectModels_25k_ply"),
        help="Directory containing .ply files (searched recursively).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/kit_superquadric_labels.csv"),
        help="CSV output for labels.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in the unlabeled list (default: 0).",
    )
    parser.add_argument(
        "--mode",
        choices=["continue", "restart"],
        default="continue",
        help="continue: resume from existing CSV; restart: overwrite CSV and relabel from scratch.",
    )
    return parser.parse_args()


def discover_ply_files(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.rglob("*.ply") if p.is_file())


def normalize_to_input_relative(raw_path: str, input_dir: Path, repo_root: Path) -> str:
    raw_path = raw_path.strip()
    if not raw_path:
        return ""

    raw = Path(raw_path)
    if raw.is_absolute():
        try:
            return raw.resolve().relative_to(input_dir.resolve()).as_posix()
        except ValueError:
            return raw.resolve().as_posix()

    # Already relative to input dir.
    if raw_path == ".":
        return raw_path
    if raw_path.startswith("./"):
        return raw_path[2:]
    if raw_path.startswith(input_dir.as_posix().rstrip("/") + "/"):
        return raw_path[len(input_dir.as_posix().rstrip("/")) + 1 :]

    # Try resolving as repo-relative path.
    repo_candidate = (repo_root / raw).resolve()
    try:
        return repo_candidate.relative_to(input_dir.resolve()).as_posix()
    except ValueError:
        return raw.as_posix()


def read_labeled_paths(csv_path: Path, input_dir: Path, repo_root: Path) -> Set[str]:
    labeled: Set[str] = set()
    if not csv_path.exists():
        return labeled

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Support old/new formats; path is always the last or second column.
            value = row[-1].strip()
            if value in ("single", "multi", "special") and len(row) >= 2:
                value = row[1].strip()
            if not value or value == "path":
                continue
            labeled.add(normalize_to_input_relative(value, input_dir, repo_root))
    return labeled


def ensure_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "path", "label"])


def overwrite_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "path", "label"])


def append_path(csv_path: Path, index: int, rel_path: str, label: str) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([index, rel_path, label])


def remove_last_label_row(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return False
    rows = rows[:-1]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return True


def to_input_relative(path: Path, input_dir: Path) -> str:
    return path.resolve().relative_to(input_dir.resolve()).as_posix()


def show_point_cloud_and_wait_choice(ply_path: Path) -> str | None:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        return None

    decision = KeyDecision()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Label: {ply_path.name}", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.0

    vis.register_key_callback(ord("1"), decision.on_single)
    vis.register_key_callback(ord("2"), decision.on_multi)
    vis.register_key_callback(ord("3"), decision.on_special)
    vis.register_key_callback(ord("Q"), decision.on_quit)
    vis.register_key_callback(ord("~"), decision.on_undo)
    vis.register_key_callback(ord("`"), decision.on_undo)

    # Optional convenience for some keyboards/IMEs.
    vis.register_key_callback(ord("q"), decision.on_quit)

    vis.run()
    vis.destroy_window()
    return decision.choice


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    input_dir = args.input_dir.resolve()

    if not args.input_dir.exists():
        print(f"ERROR: input directory not found: {args.input_dir}", file=sys.stderr)
        return 2

    all_ply_files = discover_ply_files(args.input_dir)
    if not all_ply_files:
        print(f"No .ply files found under: {args.input_dir}")
        return 0

    if args.mode == "restart":
        overwrite_csv_header(args.csv)
        already_labeled: Set[str] = set()
    else:
        ensure_csv_header(args.csv)
        already_labeled = read_labeled_paths(args.csv, input_dir, repo_root)

    target_files: List[Path] = []
    all_index: dict[str, int] = {}
    for p in all_ply_files:
        rel = to_input_relative(p, input_dir)
        all_index[rel] = len(all_index)
        if args.mode == "restart" or rel not in already_labeled:
            target_files.append(p)

    if args.start_index < 0 or args.start_index >= max(1, len(target_files)):
        print(
            f"ERROR: --start-index out of range. got {args.start_index}, "
            f"valid range is [0, {max(0, len(target_files)-1)}]",
            file=sys.stderr,
        )
        return 2

    print("Interactive labeling controls:")
    print("- In Open3D window, press 1 => single-superquadric")
    print("- In Open3D window, press 2 => multi-superquadric")
    print("- In Open3D window, press 3 => special")
    print("- In Open3D window, press ~ (or `) => undo previous label")
    print("- In Open3D window, press Q => quit")
    print()
    print(f"Total .ply files      : {len(all_ply_files)}")
    print(f"Mode                  : {args.mode}")
    print(f"Already labeled       : {len(already_labeled)}")
    print(f"Target in this run    : {len(target_files)}")
    print(f"Start index           : {args.start_index}")

    if not target_files:
        print("All files are already labeled. Nothing to do.")
        return 0

    labeled_this_run = 0
    history: List[tuple[int, str, str]] = []
    i = args.start_index

    while i < len(target_files):
        ply_path = target_files[i]
        rel = to_input_relative(ply_path, input_dir)
        rel_idx = all_index[rel]
        print(f"\n[{i + 1}/{len(target_files)}] {rel}")

        choice = show_point_cloud_and_wait_choice(ply_path)

        if choice == LABEL_SINGLE:
            append_path(args.csv, rel_idx, rel, LABEL_SINGLE)
            history.append((i, rel, LABEL_SINGLE))
            labeled_this_run += 1
            print("  -> labeled as SINGLE")
            i += 1
        elif choice == LABEL_MULTI:
            append_path(args.csv, rel_idx, rel, LABEL_MULTI)
            history.append((i, rel, LABEL_MULTI))
            labeled_this_run += 1
            print("  -> labeled as MULTI")
            i += 1
        elif choice == LABEL_SPECIAL:
            append_path(args.csv, rel_idx, rel, LABEL_SPECIAL)
            history.append((i, rel, LABEL_SPECIAL))
            labeled_this_run += 1
            print("  -> labeled as SPECIAL")
            i += 1
        elif choice == "undo":
            if not history:
                print("  -> no previous label to undo")
                continue
            prev_i, prev_rel, prev_label = history.pop()
            removed = remove_last_label_row(args.csv)
            if removed:
                labeled_this_run = max(0, labeled_this_run - 1)
                i = prev_i
                print(f"  -> undo last label: {prev_rel} ({prev_label}), back to it")
            else:
                print("  -> undo failed: CSV has no data rows")
        elif choice == "quit":
            print("User requested quit. Progress has been saved.")
            break
        else:
            print("  -> skip (empty point cloud or no valid key pressed)")

    print("\nDone.")
    print(f"Labeled in this run: {labeled_this_run}")
    print(f"CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
