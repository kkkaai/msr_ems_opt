#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

PLY_DTYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
}


def _parse_header(path: Path):
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            header_lines.append(line.decode("ascii", errors="strict").strip())
            if header_lines[-1] == "end_header":
                break
        data_offset = f.tell()

    if not header_lines or header_lines[0] != "ply":
        raise ValueError(f"Not a PLY file: {path}")

    fmt = None
    vertex_count = None
    in_vertex = False
    prop_defs = []

    for line in header_lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise ValueError(f"Vertex list property not supported: {path}")
            prop_defs.append((parts[2], parts[1]))

    if fmt is None or vertex_count is None or not prop_defs:
        raise ValueError(f"Missing required header fields in {path}")

    return fmt, vertex_count, prop_defs, data_offset


def read_ply_xyz(path: Path) -> np.ndarray:
    fmt, vertex_count, prop_defs, data_offset = _parse_header(path)

    needed = ["x", "y", "z"]
    prop_names = [p[0] for p in prop_defs]
    if not all(n in prop_names for n in needed):
        raise ValueError(f"No xyz fields in {path}")

    if fmt == "binary_little_endian":
        np_dtype = []
        for prop_name, prop_type in prop_defs:
            if prop_type not in PLY_DTYPE_MAP:
                raise ValueError(f"Unsupported type {prop_type} in {path}")
            np_dtype.append((prop_name, "<" + PLY_DTYPE_MAP[prop_type]))

        with path.open("rb") as f:
            f.seek(data_offset)
            arr = np.fromfile(f, dtype=np.dtype(np_dtype), count=vertex_count)
        if arr.shape[0] != vertex_count:
            raise ValueError(f"Unexpected EOF in {path}")
        pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
        return pts

    if fmt == "ascii":
        usecols = [prop_names.index("x"), prop_names.index("y"), prop_names.index("z")]
        pts = np.loadtxt(path, dtype=np.float64, skiprows=len(open(path, "r", encoding="ascii", errors="strict").read().split("end_header\n", 1)[0].splitlines()) + 1, usecols=usecols, max_rows=vertex_count)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.shape[0] != vertex_count or pts.shape[1] != 3:
            raise ValueError(f"ASCII vertex parse failed in {path}")
        return pts

    raise ValueError(f"Unsupported PLY format {fmt} in {path}")


def compute_one(path: Path, rng: np.random.Generator, pair_samples: int):
    pts = read_ply_xyz(path)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ext = maxs - mins
    diag = float(np.linalg.norm(ext))

    if cKDTree is not None:
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        nn = dists[:, 1]
    else:
        diff = pts[:, None, :] - pts[None, :, :]
        dmat = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dmat, np.inf)
        nn = dmat.min(axis=1)

    n = pts.shape[0]
    s = min(pair_samples, n * (n - 1) // 2)
    i = rng.integers(0, n, size=s)
    j = rng.integers(0, n, size=s)
    mask = i != j
    i = i[mask]
    j = j[mask]
    pd = np.linalg.norm(pts[i] - pts[j], axis=1)

    return {
        "file": path.name,
        "num_points": int(n),
        "bbox_min": mins.tolist(),
        "bbox_max": maxs.tolist(),
        "bbox_extent": ext.tolist(),
        "bbox_diagonal": diag,
        "bbox_volume": float(ext[0] * ext[1] * ext[2]),
        "nn_dist_mean": float(nn.mean()),
        "nn_dist_median": float(np.median(nn)),
        "nn_dist_p05": float(np.percentile(nn, 5)),
        "nn_dist_p95": float(np.percentile(nn, 95)),
        "nn_dist_min": float(nn.min()),
        "nn_dist_max": float(nn.max()),
        "pair_dist_mean": float(pd.mean()) if len(pd) else 0.0,
        "pair_dist_median": float(np.median(pd)) if len(pd) else 0.0,
        "pair_dist_p05": float(np.percentile(pd, 5)) if len(pd) else 0.0,
        "pair_dist_p95": float(np.percentile(pd, 95)) if len(pd) else 0.0,
    }


def summarize(all_stats):
    arr_ext = np.array([s["bbox_extent"] for s in all_stats])
    arr_diag = np.array([s["bbox_diagonal"] for s in all_stats])
    arr_vol = np.array([s["bbox_volume"] for s in all_stats])
    arr_nn_mean = np.array([s["nn_dist_mean"] for s in all_stats])
    arr_nn_med = np.array([s["nn_dist_median"] for s in all_stats])
    arr_pair_mean = np.array([s["pair_dist_mean"] for s in all_stats])

    return {
        "num_files": len(all_stats),
        "points_per_file": {
            "min": int(min(s["num_points"] for s in all_stats)),
            "max": int(max(s["num_points"] for s in all_stats)),
            "mean": float(np.mean([s["num_points"] for s in all_stats])),
        },
        "bbox_extent_xyz": {
            "min": arr_ext.min(axis=0).tolist(),
            "max": arr_ext.max(axis=0).tolist(),
            "mean": arr_ext.mean(axis=0).tolist(),
            "median": np.median(arr_ext, axis=0).tolist(),
        },
        "bbox_diagonal": {
            "min": float(arr_diag.min()),
            "max": float(arr_diag.max()),
            "mean": float(arr_diag.mean()),
            "median": float(np.median(arr_diag)),
        },
        "bbox_volume": {
            "min": float(arr_vol.min()),
            "max": float(arr_vol.max()),
            "mean": float(arr_vol.mean()),
            "median": float(np.median(arr_vol)),
        },
        "nn_distance": {
            "mean_of_means": float(arr_nn_mean.mean()),
            "mean_of_medians": float(arr_nn_med.mean()),
            "min_mean": float(arr_nn_mean.min()),
            "max_mean": float(arr_nn_mean.max()),
        },
        "pair_distance": {
            "mean_of_means": float(arr_pair_mean.mean()),
            "min_mean": float(arr_pair_mean.min()),
            "max_mean": float(arr_pair_mean.max()),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pair-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_dir = Path(args.input)
    files = sorted(in_dir.glob("*.ply"))
    if not files:
        raise SystemExit(f"No .ply files found in {in_dir}")

    rng = np.random.default_rng(args.seed)
    all_stats = [compute_one(p, rng, args.pair_samples) for p in files]
    summary = summarize(all_stats)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "files": all_stats}, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_path} with {len(all_stats)} files")


if __name__ == "__main__":
    main()
