import argparse
import sys
import timeit
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.cluster import DBSCAN

from pointcloud_normalization import (
    NormalizationMeta,
    denormalize_points,
    denormalize_superquadrics_inplace,
    normalize_points,
)


def add_external_src_path(repo_root: Path) -> None:
    ems_src_dir = repo_root / "external" / "EMS-superquadric_fitting" / "Python" / "src"
    if str(ems_src_dir) not in sys.path:
        sys.path.insert(0, str(ems_src_dir))


def str2bool(value: str) -> bool:
    value_lower = value.lower()
    if value_lower in {"true", "1", "yes", "y", "t"}:
        return True
    if value_lower in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}. Use True/False.")


def read_ply_xyz(path: Path) -> np.ndarray:
    try:
        from plyfile import PlyData  # type: ignore

        ply = PlyData.read(str(path))
        vertex = ply["vertex"]
        points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(float)
        return points
    except Exception:
        pass

    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY file: unexpected EOF before end_header")
            line_str = line.decode("utf-8", errors="ignore").strip()
            header_lines.append(line_str)
            if line_str == "end_header":
                break

        format_line = next((ln for ln in header_lines if ln.startswith("format ")), "")
        if "ascii" not in format_line:
            raise RuntimeError("Binary PLY requires the 'plyfile' package.")

        vertex_line = next((ln for ln in header_lines if ln.startswith("element vertex ")), "")
        if not vertex_line:
            raise ValueError("Invalid PLY file: missing 'element vertex' in header")
        vertex_count = int(vertex_line.split()[-1])

        points = np.loadtxt(f, dtype=float, usecols=(0, 1, 2), max_rows=vertex_count)
        if points.ndim == 1:
            points = points.reshape(1, 3)
        return points


@dataclass
class RecoverHyperParams:
    outlier_ratio: float
    sigma: float
    max_layer: int
    tau_in: float
    tau_split: float
    eps: float
    min_points: int
    adaptive_upper_bound: bool
    rescale: bool
    max_iteration_em: int
    tolerance_em: float
    relative_tolerance_em: float
    max_opti_iterations: int
    maxi_switch: int
    min_cluster_ratio: float


@dataclass
class PsoConfig:
    swarmsize: int
    maxiter: int
    omega: float
    c1: float
    c2: float
    tol: float
    patience: int
    fitness_threshold: float
    seed: int | None


@dataclass
class CmaesConfig:
    popsize: int
    maxiter: int
    sigma: float
    tol: float
    patience: int
    fitness_threshold: float
    seed: int | None


@dataclass
class GridSearchConfig:
    steps_per_dim: int


@dataclass
class LayerEval:
    child_segments: list[np.ndarray]
    child_segment_indices: list[np.ndarray]
    inlier_groups: list[np.ndarray]
    inlier_index_groups: list[np.ndarray]
    outlier_groups: list[np.ndarray]
    outlier_index_groups: list[np.ndarray]
    quadrics: list
    primitive_assignments: list[dict]
    fitness: float
    inlier_count: int
    distance_fit: float
    coverage_ratio: float
    residual_unexplained_ratio: float
    num_superquadrics: int


def compose_fitness(
    distance_fit: float,
    coverage_ratio: float,
    residual_unexplained_ratio: float = 0.0,
    num_superquadrics: int = 0,
    fitness_mode: str = "legacy",
    lambda_cov: float = 0.0,
    lambda_out: float = 0.0,
    lambda_comp: float = 0.0,
) -> float:
    if not np.isfinite(distance_fit):
        return float("inf")
    if fitness_mode == "legacy":
        return float(distance_fit)
    if fitness_mode == "distance_coverage":
        cov_term = max(0.0, min(1.0, float(coverage_ratio)))
        return float(distance_fit + float(lambda_cov) * (1.0 - cov_term))
    if fitness_mode == "distance_coverage_outlier_complexity":
        cov_term = max(0.0, min(1.0, float(coverage_ratio)))
        out_term = max(0.0, min(1.0, float(residual_unexplained_ratio)))
        comp_term = max(0.0, float(num_superquadrics))
        return float(
            distance_fit
            + float(lambda_cov) * (1.0 - cov_term)
            + float(lambda_out) * out_term
            + float(lambda_comp) * comp_term
        )
    raise ValueError(
        f"Unsupported fitness_mode={fitness_mode}. "
        "Expected one of: legacy, distance_coverage, distance_coverage_outlier_complexity."
    )


def sq_to_dict(sq) -> dict:
    return {
        "shape": np.asarray(sq.shape).tolist(),
        "scale": np.asarray(sq.scale).tolist(),
        "euler": np.asarray(sq.euler).tolist(),
        "translation": np.asarray(sq.translation).tolist(),
    }


def primitive_assignment_to_dict(assignment: dict) -> dict:
    return {
        "primitive_id": int(assignment["primitive_id"]),
        "layer": int(assignment["layer"]),
        "segment_index": int(assignment["segment_index"]),
        "input_point_count": int(assignment["input_point_count"]),
        "inlier_count": int(assignment["inlier_count"]),
        "outlier_count": int(assignment["outlier_count"]),
        "inlier_indices": np.asarray(assignment["inlier_indices"], dtype=np.int64).tolist(),
    }


def _sq_to_x(sq) -> np.ndarray:
    return np.array(
        [
            sq.shape[0],
            sq.shape[1],
            sq.scale[0],
            sq.scale[1],
            sq.scale[2],
            sq.euler[0],
            sq.euler[1],
            sq.euler[2],
            sq.translation[0],
            sq.translation[1],
            sq.translation[2],
        ],
        dtype=float,
    )


def run_pso(
    objective: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    config: PsoConfig,
    init_particle: np.ndarray | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    rng = np.random.default_rng(config.seed)
    dim = lb.shape[0]

    particles = rng.uniform(lb, ub, size=(config.swarmsize, dim))
    if init_particle is not None:
        init_particle = np.asarray(init_particle, dtype=float).reshape(-1)
        if init_particle.shape[0] != dim:
            raise ValueError(
                f"init_particle dimension mismatch: expected {dim}, got {init_particle.shape[0]}"
            )
        particles[0] = np.clip(init_particle, lb, ub)
    vel_scale = 0.2 * (ub - lb)
    velocities = rng.uniform(-vel_scale, vel_scale, size=(config.swarmsize, dim))

    pbest = particles.copy()
    pbest_scores = np.array([float("inf")] * config.swarmsize, dtype=float)
    gbest = particles[0].copy()
    gbest_score = float("inf")

    history: list[float] = []
    stale_iters = 0

    for _ in range(config.maxiter):
        improved = False
        for i in range(config.swarmsize):
            score = float(objective(particles[i]))
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest[i] = particles[i].copy()
            if score < gbest_score:
                gbest_score = score
                gbest = particles[i].copy()
                improved = True

        history.append(gbest_score)
        if gbest_score <= config.fitness_threshold:
            break

        if improved:
            stale_iters = 0
        else:
            stale_iters += 1
            if stale_iters >= config.patience:
                break

        if len(history) >= 2 and abs(history[-1] - history[-2]) < config.tol:
            stale_iters += 1
            if stale_iters >= config.patience:
                break

        r1 = rng.random((config.swarmsize, dim))
        r2 = rng.random((config.swarmsize, dim))
        velocities = (
            config.omega * velocities
            + config.c1 * r1 * (pbest - particles)
            + config.c2 * r2 * (gbest - particles)
        )
        particles = np.clip(particles + velocities, lb, ub)

    return gbest, gbest_score, history


def run_cmaes(
    objective: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    config: CmaesConfig,
    init_particle: np.ndarray | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    try:
        import cma  # type: ignore
    except Exception as exc:
        raise RuntimeError("CMA-ES requires package 'cma'. Install with: pip install cma") from exc

    dim = lb.shape[0]
    if init_particle is None:
        x0 = (lb + ub) / 2.0
    else:
        x0 = np.asarray(init_particle, dtype=float).reshape(-1)
        if x0.shape[0] != dim:
            raise ValueError(f"init_particle dimension mismatch: expected {dim}, got {x0.shape[0]}")
        x0 = np.clip(x0, lb, ub)

    options = {
        "bounds": [lb.tolist(), ub.tolist()],
        "popsize": int(config.popsize),
        "seed": None if config.seed is None else int(config.seed),
        "maxiter": int(config.maxiter),
        "verb_disp": 0,
        "verbose": -9,
    }
    es = cma.CMAEvolutionStrategy(x0.tolist(), float(max(config.sigma, 1e-9)), options)

    history: list[float] = []
    stale_iters = 0
    best_seen = float("inf")
    best_solution = np.asarray(x0, dtype=float)
    best_value = float("inf")
    for _ in range(int(config.maxiter)):
        solutions = es.ask()
        values = [float(objective(np.asarray(s, dtype=float))) for s in solutions]
        for solution, value in zip(solutions, values):
            if np.isfinite(value) and value < best_value:
                best_solution = np.asarray(solution, dtype=float)
                best_value = float(value)
        es.tell(solutions, values)
        curr_best = float(best_value)
        history.append(curr_best)
        if curr_best <= config.fitness_threshold:
            break

        improved = curr_best + 1e-12 < best_seen
        if improved:
            best_seen = curr_best
            stale_iters = 0
        else:
            stale_iters += 1
            if stale_iters >= int(config.patience):
                break

        if len(history) >= 2 and abs(history[-1] - history[-2]) < float(config.tol):
            stale_iters += 1
            if stale_iters >= int(config.patience):
                break

        if es.stop():
            break

    return best_solution, best_value, history


def evaluate_layer(
    segments: list[np.ndarray],
    hp: RecoverHyperParams,
    layer_idx: int,
    total_points: int,
    segment_indices: list[np.ndarray] | None = None,
    fitness_mode: str = "legacy",
    lambda_cov: float = 0.0,
    lambda_out: float = 0.0,
    lambda_comp: float = 0.0,
    min_coverage: float = 0.0,
    lambda_min_coverage: float = 0.0,
) -> LayerEval:
    from EMS.EMS_recovery import EMS_recovery, Distance

    child_segments: list[np.ndarray] = []
    child_segment_indices: list[np.ndarray] = []
    inlier_groups: list[np.ndarray] = []
    inlier_index_groups: list[np.ndarray] = []
    outlier_groups: list[np.ndarray] = []
    outlier_index_groups: list[np.ndarray] = []
    quadrics = []
    primitive_assignments: list[dict] = []

    if segment_indices is None:
        segment_indices = [
            np.arange(segment.shape[0], dtype=np.int64) for segment in segments
        ]
    if len(segment_indices) != len(segments):
        raise ValueError("segment_indices must be aligned one-to-one with segments.")

    min_cluster_size_abs = max(5, int(round(hp.min_cluster_ratio * total_points)))
    weighted_dist_sum = 0.0
    inlier_count_sum = 0
    residual_unexplained_points_sum = 0

    for segment_index, (segment, original_indices) in enumerate(
        zip(segments, segment_indices)
    ):
        original_indices = np.asarray(original_indices, dtype=np.int64).reshape(-1)
        if original_indices.shape[0] != segment.shape[0]:
            raise ValueError("Each index array must have the same length as its segment.")
        if segment.shape[0] < 11:
            outlier_groups.append(segment)
            outlier_index_groups.append(original_indices)
            residual_unexplained_points_sum += int(segment.shape[0])
            continue

        sq, p_raw = EMS_recovery(
            segment,
            OutlierRatio=hp.outlier_ratio,
            MaxIterationEM=hp.max_iteration_em,
            ToleranceEM=hp.tolerance_em,
            RelativeToleranceEM=hp.relative_tolerance_em,
            MaxOptiIterations=hp.max_opti_iterations,
            Sigma=hp.sigma,
            MaxiSwitch=hp.maxi_switch,
            AdaptiveUpperBound=hp.adaptive_upper_bound,
            Rescale=hp.rescale,
        )
        quadrics.append(sq)

        inlier_mask = p_raw > hp.tau_in
        inlier_points = segment[inlier_mask, :]
        outlier_points = segment[~inlier_mask, :]
        inlier_indices = original_indices[inlier_mask]
        outlier_indices = original_indices[~inlier_mask]
        inlier_groups.append(inlier_points)
        inlier_index_groups.append(inlier_indices)
        primitive_assignments.append(
            {
                "layer": int(layer_idx),
                "segment_index": int(segment_index),
                "input_point_count": int(segment.shape[0]),
                "inlier_count": int(inlier_indices.shape[0]),
                "outlier_count": int(outlier_indices.shape[0]),
                "inlier_indices": inlier_indices,
            }
        )

        if inlier_points.shape[0] > 0:
            dist = Distance(inlier_points, _sq_to_x(sq))
            score = float(np.mean(np.abs(dist)))
            if np.isfinite(score):
                weighted_dist_sum += score * inlier_points.shape[0]
                inlier_count_sum += inlier_points.shape[0]

        can_split = (
            layer_idx < hp.max_layer - 1
            and np.sum(p_raw) < (hp.tau_split * segment.shape[0])
            and outlier_points.shape[0] >= hp.min_points
        )
        if can_split:
            clustering = DBSCAN(eps=hp.eps, min_samples=hp.min_points).fit(outlier_points)
            labels = clustering.labels_
            unique_labels = [int(item) for item in set(labels) if item >= 0]
            for label in unique_labels:
                cluster = outlier_points[labels == label, :]
                if cluster.shape[0] >= min_cluster_size_abs:
                    child_segments.append(cluster)
                    child_segment_indices.append(outlier_indices[labels == label])
            residual_points = outlier_points[labels == -1, :]
            residual_indices = outlier_indices[labels == -1]
            outlier_groups.append(residual_points)
            outlier_index_groups.append(residual_indices)
            residual_unexplained_points_sum += int(residual_points.shape[0])
        else:
            outlier_groups.append(outlier_points)
            outlier_index_groups.append(outlier_indices)
            residual_unexplained_points_sum += int(outlier_points.shape[0])

    layer_points_total = int(sum(int(seg.shape[0]) for seg in segments))
    distance_fit = float("inf") if inlier_count_sum == 0 else weighted_dist_sum / inlier_count_sum
    coverage_ratio = 0.0 if layer_points_total == 0 else float(inlier_count_sum / layer_points_total)
    residual_unexplained_ratio = (
        0.0 if layer_points_total == 0 else float(residual_unexplained_points_sum / layer_points_total)
    )
    num_superquadrics = int(len(quadrics))
    fitness = compose_fitness(
        distance_fit=distance_fit,
        coverage_ratio=coverage_ratio,
        residual_unexplained_ratio=residual_unexplained_ratio,
        num_superquadrics=num_superquadrics,
        fitness_mode=fitness_mode,
        lambda_cov=lambda_cov,
        lambda_out=lambda_out,
        lambda_comp=lambda_comp,
    )
    coverage_deficit = max(0.0, float(min_coverage) - coverage_ratio)
    fitness += float(lambda_min_coverage) * coverage_deficit**2
    return LayerEval(
        child_segments=child_segments,
        child_segment_indices=child_segment_indices,
        inlier_groups=inlier_groups,
        inlier_index_groups=inlier_index_groups,
        outlier_groups=outlier_groups,
        outlier_index_groups=outlier_index_groups,
        quadrics=quadrics,
        primitive_assignments=primitive_assignments,
        fitness=float(fitness),
        inlier_count=inlier_count_sum,
        distance_fit=float(distance_fit),
        coverage_ratio=float(coverage_ratio),
        residual_unexplained_ratio=float(residual_unexplained_ratio),
        num_superquadrics=int(num_superquadrics),
    )


def run_layerwise_pso(
    point_cloud: np.ndarray,
    base_hp: RecoverHyperParams,
    pso_cfg: PsoConfig,
    lb: np.ndarray,
    ub: np.ndarray,
    decode_particle: Callable[[np.ndarray, RecoverHyperParams], RecoverHyperParams],
    global_normalize: bool = False,
    global_norm_method: str = "ems_matlab",
    output_in_original_scale: bool = True,
    fitness_mode: str = "legacy",
    lambda_cov: float = 0.0,
    lambda_out: float = 0.0,
    lambda_comp: float = 0.0,
    min_coverage: float = 0.0,
    lambda_min_coverage: float = 0.0,
) -> dict:
    def _denormalize_nested_point_dict(point_dict: dict, meta: NormalizationMeta) -> None:
        for key, groups in point_dict.items():
            if not isinstance(groups, list):
                continue
            converted = []
            for arr in groups:
                arr_np = np.asarray(arr, dtype=float)
                if arr_np.size == 0:
                    converted.append(arr_np.reshape((-1, 3)))
                else:
                    converted.append(denormalize_points(arr_np, meta))
            point_dict[key] = converted

    norm_meta: NormalizationMeta | None = None
    if global_normalize:
        point_cloud_working, norm_meta = normalize_points(point_cloud, method=global_norm_method)
    else:
        point_cloud_working = np.asarray(point_cloud, dtype=float)

    point_seg = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_inlier = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_outlier = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_seg_indices = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_inlier_indices = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_outlier_indices = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_seg[0] = [point_cloud_working]
    point_seg_indices[0] = [np.arange(point_cloud_working.shape[0], dtype=np.int64)]

    list_quadrics = []
    primitive_assignments: list[dict] = []
    total_weighted_dist = 0.0
    total_inliers = 0

    layer_histories: list[list[float]] = []
    layer_best_hparams: list[dict] = []
    layer_best_fitness: list[float] = []
    layer_best_distance_fit: list[float] = []
    layer_best_coverage_ratio: list[float] = []
    layer_best_residual_unexplained_ratio: list[float] = []
    layer_best_num_superquadrics: list[int] = []
    layer_timing: list[dict] = []

    for layer_idx in range(base_hp.max_layer):
        layer_t_start = timeit.default_timer()
        segments = point_seg[layer_idx]
        segment_indices = point_seg_indices[layer_idx]
        if len(segments) == 0:
            layer_histories.append([])
            layer_best_hparams.append({})
            layer_best_fitness.append(float("inf"))
            layer_best_distance_fit.append(float("inf"))
            layer_best_coverage_ratio.append(0.0)
            layer_best_residual_unexplained_ratio.append(0.0)
            layer_best_num_superquadrics.append(0)
            layer_timing.append(
                {
                    "layer": int(layer_idx),
                    "status": "skipped_empty",
                    "num_input_segments": 0,
                    "pso_iterations": 0,
                    "pso_time_ms": 0.0,
                    "best_eval_time_ms": 0.0,
                    "layer_total_time_ms": 0.0,
                    "best_fitness": float("inf"),
                    "best_distance_fit": float("inf"),
                    "best_coverage_ratio": 0.0,
                    "best_residual_unexplained_ratio": 0.0,
                    "best_num_superquadrics": 0,
                    "num_recovered_quadrics": 0,
                    "num_child_segments": 0,
                    "inlier_points": 0,
                }
            )
            continue

        cache: dict[tuple, tuple[RecoverHyperParams, LayerEval]] = {}

        def objective(x: np.ndarray) -> float:
            hp = decode_particle(x, base_hp)
            key = tuple(np.round(x, 6).tolist())
            if key in cache:
                return cache[key][1].fitness
            try:
                eval_result = evaluate_layer(
                    segments=segments,
                    hp=hp,
                    layer_idx=layer_idx,
                    total_points=point_cloud_working.shape[0],
                    segment_indices=segment_indices,
                    fitness_mode=fitness_mode,
                    lambda_cov=lambda_cov,
                    lambda_out=lambda_out,
                    lambda_comp=lambda_comp,
                    min_coverage=min_coverage,
                    lambda_min_coverage=lambda_min_coverage,
                )
            except Exception:
                return float("inf")
            cache[key] = (hp, eval_result)
            return eval_result.fitness

        pso_t_start = timeit.default_timer()
        best_x, best_f, history = run_pso(objective, lb, ub, pso_cfg)
        pso_t_end = timeit.default_timer()
        best_hp_layer = decode_particle(best_x, base_hp)
        best_key = tuple(np.round(best_x, 6).tolist())
        best_eval_t_start = timeit.default_timer()
        best_eval: LayerEval | None = None
        if best_key in cache and np.isfinite(cache[best_key][1].fitness):
            _, best_eval = cache[best_key]
        else:
            try:
                best_eval = evaluate_layer(
                    segments=segments,
                    hp=best_hp_layer,
                    layer_idx=layer_idx,
                    total_points=point_cloud_working.shape[0],
                    segment_indices=segment_indices,
                    fitness_mode=fitness_mode,
                    lambda_cov=lambda_cov,
                    lambda_out=lambda_out,
                    lambda_comp=lambda_comp,
                    min_coverage=min_coverage,
                    lambda_min_coverage=lambda_min_coverage,
                )
            except Exception:
                best_eval = None

        if best_eval is None or not np.isfinite(best_eval.fitness):
            finite_candidates = [
                eval_res for _, eval_res in cache.values() if np.isfinite(eval_res.fitness)
            ]
            if finite_candidates:
                best_eval = min(finite_candidates, key=lambda e: float(e.fitness))
                best_f = float(best_eval.fitness)
            else:
                layer_histories.append([float(v) for v in history])
                layer_best_hparams.append(
                    {
                        "outlier_ratio": best_hp_layer.outlier_ratio,
                        "sigma": best_hp_layer.sigma,
                        "max_layer": best_hp_layer.max_layer,
                        "tau_in": best_hp_layer.tau_in,
                        "tau_split": best_hp_layer.tau_split,
                        "eps": best_hp_layer.eps,
                        "min_points": best_hp_layer.min_points,
                    }
                )
                layer_best_fitness.append(float("inf"))
                layer_best_distance_fit.append(float("inf"))
                layer_best_coverage_ratio.append(0.0)
                layer_best_residual_unexplained_ratio.append(0.0)
                layer_best_num_superquadrics.append(0)
                point_seg[layer_idx + 1] = []
                point_seg_indices[layer_idx + 1] = []
                point_inlier[layer_idx] = []
                point_inlier_indices[layer_idx] = []
                point_outlier[layer_idx] = segments
                point_outlier_indices[layer_idx] = segment_indices
                best_eval_t_end = timeit.default_timer()
                layer_t_end = timeit.default_timer()
                layer_timing.append(
                    {
                        "layer": int(layer_idx),
                        "status": "failed_no_valid_candidate",
                        "num_input_segments": int(len(segments)),
                        "pso_iterations": int(len(history)),
                        "pso_time_ms": float((pso_t_end - pso_t_start) * 1000.0),
                        "best_eval_time_ms": float((best_eval_t_end - best_eval_t_start) * 1000.0),
                        "layer_total_time_ms": float((layer_t_end - layer_t_start) * 1000.0),
                        "best_fitness": float("inf"),
                        "best_distance_fit": float("inf"),
                        "best_coverage_ratio": 0.0,
                        "best_residual_unexplained_ratio": 0.0,
                        "best_num_superquadrics": 0,
                        "num_recovered_quadrics": 0,
                        "num_child_segments": 0,
                        "inlier_points": 0,
                    }
                )
                continue
        best_eval_t_end = timeit.default_timer()

        point_seg[layer_idx + 1] = best_eval.child_segments
        point_seg_indices[layer_idx + 1] = best_eval.child_segment_indices
        point_inlier[layer_idx] = best_eval.inlier_groups
        point_inlier_indices[layer_idx] = best_eval.inlier_index_groups
        point_outlier[layer_idx] = best_eval.outlier_groups
        point_outlier_indices[layer_idx] = best_eval.outlier_index_groups
        list_quadrics.extend(best_eval.quadrics)
        for assignment in best_eval.primitive_assignments:
            assignment_with_id = dict(assignment)
            assignment_with_id["primitive_id"] = len(primitive_assignments)
            primitive_assignments.append(assignment_with_id)

        if best_eval.inlier_count > 0 and np.isfinite(best_eval.fitness):
            total_weighted_dist += best_eval.distance_fit * best_eval.inlier_count
            total_inliers += best_eval.inlier_count

        layer_histories.append([float(v) for v in history])
        layer_best_hparams.append(
            {
                "outlier_ratio": best_hp_layer.outlier_ratio,
                "sigma": best_hp_layer.sigma,
                "max_layer": best_hp_layer.max_layer,
                "tau_in": best_hp_layer.tau_in,
                "tau_split": best_hp_layer.tau_split,
                "eps": best_hp_layer.eps,
                "min_points": best_hp_layer.min_points,
            }
        )
        layer_best_fitness.append(float(best_f))
        layer_best_distance_fit.append(float(best_eval.distance_fit))
        layer_best_coverage_ratio.append(float(best_eval.coverage_ratio))
        layer_best_residual_unexplained_ratio.append(float(best_eval.residual_unexplained_ratio))
        layer_best_num_superquadrics.append(int(best_eval.num_superquadrics))
        layer_t_end = timeit.default_timer()
        layer_timing.append(
            {
                "layer": int(layer_idx),
                "status": "ok",
                "num_input_segments": int(len(segments)),
                "pso_iterations": int(len(history)),
                "pso_time_ms": float((pso_t_end - pso_t_start) * 1000.0),
                "best_eval_time_ms": float((best_eval_t_end - best_eval_t_start) * 1000.0),
                "layer_total_time_ms": float((layer_t_end - layer_t_start) * 1000.0),
                "best_fitness": float(best_f),
                "best_distance_fit": float(best_eval.distance_fit),
                "best_coverage_ratio": float(best_eval.coverage_ratio),
                "best_residual_unexplained_ratio": float(best_eval.residual_unexplained_ratio),
                "best_num_superquadrics": int(best_eval.num_superquadrics),
                "num_recovered_quadrics": int(len(best_eval.quadrics)),
                "num_child_segments": int(len(best_eval.child_segments)),
                "inlier_points": int(best_eval.inlier_count),
            }
        )

    final_distance_fit = float("inf") if total_inliers == 0 else total_weighted_dist / total_inliers
    global_coverage_ratio = (
        0.0 if point_cloud_working.shape[0] == 0 else float(total_inliers / point_cloud_working.shape[0])
    )
    total_residual_points = 0
    for _, groups in point_outlier.items():
        for arr in groups:
            arr_np = np.asarray(arr)
            if arr_np.ndim >= 1:
                total_residual_points += int(arr_np.shape[0])
    global_residual_unexplained_ratio = (
        0.0 if point_cloud_working.shape[0] == 0 else float(total_residual_points / point_cloud_working.shape[0])
    )
    total_num_superquadrics = int(len(list_quadrics))
    final_fitness = compose_fitness(
        distance_fit=final_distance_fit,
        coverage_ratio=global_coverage_ratio,
        residual_unexplained_ratio=global_residual_unexplained_ratio,
        num_superquadrics=total_num_superquadrics,
        fitness_mode=fitness_mode,
        lambda_cov=lambda_cov,
        lambda_out=lambda_out,
        lambda_comp=lambda_comp,
    )
    summary = {
        "num_superquadrics": len(list_quadrics),
        "inlier_points_total": int(total_inliers),
        "coverage_ratio": float(global_coverage_ratio),
        "residual_unexplained_ratio": float(global_residual_unexplained_ratio),
        "distance_fit": float(final_distance_fit),
        "point_seg_count_per_layer": {str(k): len(v) for k, v in point_seg.items()},
        "point_inlier_count_per_layer": {str(k): len(v) for k, v in point_inlier.items()},
        "point_outlier_count_per_layer": {str(k): len(v) for k, v in point_outlier.items()},
    }

    if norm_meta is not None and output_in_original_scale:
        denormalize_superquadrics_inplace(list_quadrics, norm_meta)
        _denormalize_nested_point_dict(point_seg, norm_meta)
        _denormalize_nested_point_dict(point_inlier, norm_meta)
        _denormalize_nested_point_dict(point_outlier, norm_meta)
        point_cloud_used = denormalize_points(point_cloud_working, norm_meta)
    else:
        point_cloud_used = point_cloud_working

    return {
        "fitness": float(final_fitness),
        "point_seg": point_seg,
        "point_inlier": point_inlier,
        "point_outlier": point_outlier,
        "point_seg_indices": point_seg_indices,
        "point_inlier_indices": point_inlier_indices,
        "point_outlier_indices": point_outlier_indices,
        "list_quadrics": list_quadrics,
        "primitive_assignments": primitive_assignments,
        "point_cloud_used": point_cloud_used,
        "summary": summary,
        "layer_histories": layer_histories,
        "layer_best_hparams": layer_best_hparams,
        "layer_best_fitness": layer_best_fitness,
        "layer_best_distance_fit": layer_best_distance_fit,
        "layer_best_coverage_ratio": layer_best_coverage_ratio,
        "layer_best_residual_unexplained_ratio": layer_best_residual_unexplained_ratio,
        "layer_best_num_superquadrics": layer_best_num_superquadrics,
        "layer_timing": layer_timing,
        "fitness_details": {
            "mode": fitness_mode,
            "lambda_cov": float(lambda_cov),
            "lambda_out": float(lambda_out),
            "lambda_comp": float(lambda_comp),
            "min_coverage": float(min_coverage),
            "lambda_min_coverage": float(lambda_min_coverage),
            "distance_fit": float(final_distance_fit),
            "coverage_ratio": float(global_coverage_ratio),
            "residual_unexplained_ratio": float(global_residual_unexplained_ratio),
            "num_superquadrics": int(total_num_superquadrics),
            "fitness": float(final_fitness),
        },
        "normalization": {
            "enabled": bool(global_normalize),
            "method": global_norm_method if global_normalize else "none",
            "output_in_original_scale": bool(output_in_original_scale),
            "meta": None if norm_meta is None else norm_meta.to_jsonable(),
        },
    }


def run_layerwise_cmaes(
    point_cloud: np.ndarray,
    base_hp: RecoverHyperParams,
    cma_cfg: CmaesConfig,
    lb: np.ndarray,
    ub: np.ndarray,
    decode_particle: Callable[[np.ndarray, RecoverHyperParams], RecoverHyperParams],
    global_normalize: bool = False,
    global_norm_method: str = "ems_matlab",
    output_in_original_scale: bool = True,
    fitness_mode: str = "legacy",
    lambda_cov: float = 0.0,
    lambda_out: float = 0.0,
    lambda_comp: float = 0.0,
    init_particle: np.ndarray | None = None,
    min_coverage: float = 0.0,
    lambda_min_coverage: float = 0.0,
) -> dict:
    def _denormalize_nested_point_dict(point_dict: dict, meta: NormalizationMeta) -> None:
        for key, groups in point_dict.items():
            if not isinstance(groups, list):
                continue
            converted = []
            for arr in groups:
                arr_np = np.asarray(arr, dtype=float)
                if arr_np.size == 0:
                    converted.append(arr_np.reshape((-1, 3)))
                else:
                    converted.append(denormalize_points(arr_np, meta))
            point_dict[key] = converted

    norm_meta: NormalizationMeta | None = None
    if global_normalize:
        point_cloud_working, norm_meta = normalize_points(point_cloud, method=global_norm_method)
    else:
        point_cloud_working = np.asarray(point_cloud, dtype=float)

    point_seg = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_inlier = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_outlier = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_seg_indices = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_inlier_indices = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_outlier_indices = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_seg[0] = [point_cloud_working]
    point_seg_indices[0] = [np.arange(point_cloud_working.shape[0], dtype=np.int64)]

    list_quadrics = []
    primitive_assignments: list[dict] = []
    total_weighted_dist = 0.0
    total_inliers = 0

    layer_histories: list[list[float]] = []
    layer_best_hparams: list[dict] = []
    layer_best_fitness: list[float] = []
    layer_best_distance_fit: list[float] = []
    layer_best_coverage_ratio: list[float] = []
    layer_best_residual_unexplained_ratio: list[float] = []
    layer_best_num_superquadrics: list[int] = []
    layer_timing: list[dict] = []

    for layer_idx in range(base_hp.max_layer):
        layer_t_start = timeit.default_timer()
        segments = point_seg[layer_idx]
        segment_indices = point_seg_indices[layer_idx]
        if len(segments) == 0:
            layer_histories.append([])
            layer_best_hparams.append({})
            layer_best_fitness.append(float("inf"))
            layer_best_distance_fit.append(float("inf"))
            layer_best_coverage_ratio.append(0.0)
            layer_best_residual_unexplained_ratio.append(0.0)
            layer_best_num_superquadrics.append(0)
            layer_timing.append(
                {
                    "layer": int(layer_idx),
                    "status": "skipped_empty",
                    "num_input_segments": 0,
                    "cma_iterations": 0,
                    "cma_time_ms": 0.0,
                    "best_eval_time_ms": 0.0,
                    "layer_total_time_ms": 0.0,
                    "best_fitness": float("inf"),
                    "best_distance_fit": float("inf"),
                    "best_coverage_ratio": 0.0,
                    "best_residual_unexplained_ratio": 0.0,
                    "best_num_superquadrics": 0,
                    "num_recovered_quadrics": 0,
                    "num_child_segments": 0,
                    "inlier_points": 0,
                }
            )
            continue

        cache: dict[tuple, tuple[RecoverHyperParams, LayerEval]] = {}

        def objective(x: np.ndarray) -> float:
            hp = decode_particle(x, base_hp)
            key = tuple(np.round(x, 6).tolist())
            if key in cache:
                return cache[key][1].fitness
            try:
                eval_result = evaluate_layer(
                    segments=segments,
                    hp=hp,
                    layer_idx=layer_idx,
                    total_points=point_cloud_working.shape[0],
                    segment_indices=segment_indices,
                    fitness_mode=fitness_mode,
                    lambda_cov=lambda_cov,
                    lambda_out=lambda_out,
                    lambda_comp=lambda_comp,
                    min_coverage=min_coverage,
                    lambda_min_coverage=lambda_min_coverage,
                )
            except Exception:
                return float("inf")
            cache[key] = (hp, eval_result)
            return eval_result.fitness

        cma_t_start = timeit.default_timer()
        best_x, best_f, history = run_cmaes(
            objective=objective,
            lb=lb,
            ub=ub,
            config=cma_cfg,
            init_particle=init_particle,
        )
        cma_t_end = timeit.default_timer()
        best_hp_layer = decode_particle(best_x, base_hp)
        best_key = tuple(np.round(best_x, 6).tolist())
        best_eval_t_start = timeit.default_timer()
        best_eval: LayerEval | None = None
        if best_key in cache and np.isfinite(cache[best_key][1].fitness):
            _, best_eval = cache[best_key]
        else:
            try:
                best_eval = evaluate_layer(
                    segments=segments,
                    hp=best_hp_layer,
                    layer_idx=layer_idx,
                    total_points=point_cloud_working.shape[0],
                    segment_indices=segment_indices,
                    fitness_mode=fitness_mode,
                    lambda_cov=lambda_cov,
                    lambda_out=lambda_out,
                    lambda_comp=lambda_comp,
                    min_coverage=min_coverage,
                    lambda_min_coverage=lambda_min_coverage,
                )
            except Exception:
                best_eval = None

        if best_eval is None or not np.isfinite(best_eval.fitness):
            finite_candidates = [eval_res for _, eval_res in cache.values() if np.isfinite(eval_res.fitness)]
            if finite_candidates:
                best_eval = min(finite_candidates, key=lambda e: float(e.fitness))
                best_f = float(best_eval.fitness)
            else:
                layer_histories.append([float(v) for v in history])
                layer_best_hparams.append(
                    {
                        "outlier_ratio": best_hp_layer.outlier_ratio,
                        "sigma": best_hp_layer.sigma,
                        "max_layer": best_hp_layer.max_layer,
                        "tau_in": best_hp_layer.tau_in,
                        "tau_split": best_hp_layer.tau_split,
                        "eps": best_hp_layer.eps,
                        "min_points": best_hp_layer.min_points,
                    }
                )
                layer_best_fitness.append(float("inf"))
                layer_best_distance_fit.append(float("inf"))
                layer_best_coverage_ratio.append(0.0)
                layer_best_residual_unexplained_ratio.append(0.0)
                layer_best_num_superquadrics.append(0)
                point_seg[layer_idx + 1] = []
                point_seg_indices[layer_idx + 1] = []
                point_inlier[layer_idx] = []
                point_inlier_indices[layer_idx] = []
                point_outlier[layer_idx] = segments
                point_outlier_indices[layer_idx] = segment_indices
                best_eval_t_end = timeit.default_timer()
                layer_t_end = timeit.default_timer()
                layer_timing.append(
                    {
                        "layer": int(layer_idx),
                        "status": "failed_no_valid_candidate",
                        "num_input_segments": int(len(segments)),
                        "cma_iterations": int(len(history)),
                        "cma_time_ms": float((cma_t_end - cma_t_start) * 1000.0),
                        "best_eval_time_ms": float((best_eval_t_end - best_eval_t_start) * 1000.0),
                        "layer_total_time_ms": float((layer_t_end - layer_t_start) * 1000.0),
                        "best_fitness": float("inf"),
                        "best_distance_fit": float("inf"),
                        "best_coverage_ratio": 0.0,
                        "best_residual_unexplained_ratio": 0.0,
                        "best_num_superquadrics": 0,
                        "num_recovered_quadrics": 0,
                        "num_child_segments": 0,
                        "inlier_points": 0,
                    }
                )
                continue
        best_eval_t_end = timeit.default_timer()

        point_seg[layer_idx + 1] = best_eval.child_segments
        point_seg_indices[layer_idx + 1] = best_eval.child_segment_indices
        point_inlier[layer_idx] = best_eval.inlier_groups
        point_inlier_indices[layer_idx] = best_eval.inlier_index_groups
        point_outlier[layer_idx] = best_eval.outlier_groups
        point_outlier_indices[layer_idx] = best_eval.outlier_index_groups
        list_quadrics.extend(best_eval.quadrics)
        for assignment in best_eval.primitive_assignments:
            assignment_with_id = dict(assignment)
            assignment_with_id["primitive_id"] = len(primitive_assignments)
            primitive_assignments.append(assignment_with_id)

        if best_eval.inlier_count > 0 and np.isfinite(best_eval.fitness):
            total_weighted_dist += best_eval.distance_fit * best_eval.inlier_count
            total_inliers += best_eval.inlier_count

        layer_histories.append([float(v) for v in history])
        layer_best_hparams.append(
            {
                "outlier_ratio": best_hp_layer.outlier_ratio,
                "sigma": best_hp_layer.sigma,
                "max_layer": best_hp_layer.max_layer,
                "tau_in": best_hp_layer.tau_in,
                "tau_split": best_hp_layer.tau_split,
                "eps": best_hp_layer.eps,
                "min_points": best_hp_layer.min_points,
            }
        )
        layer_best_fitness.append(float(best_f))
        layer_best_distance_fit.append(float(best_eval.distance_fit))
        layer_best_coverage_ratio.append(float(best_eval.coverage_ratio))
        layer_best_residual_unexplained_ratio.append(float(best_eval.residual_unexplained_ratio))
        layer_best_num_superquadrics.append(int(best_eval.num_superquadrics))
        layer_t_end = timeit.default_timer()
        layer_timing.append(
            {
                "layer": int(layer_idx),
                "status": "ok",
                "num_input_segments": int(len(segments)),
                "cma_iterations": int(len(history)),
                "cma_time_ms": float((cma_t_end - cma_t_start) * 1000.0),
                "best_eval_time_ms": float((best_eval_t_end - best_eval_t_start) * 1000.0),
                "layer_total_time_ms": float((layer_t_end - layer_t_start) * 1000.0),
                "best_fitness": float(best_f),
                "best_distance_fit": float(best_eval.distance_fit),
                "best_coverage_ratio": float(best_eval.coverage_ratio),
                "best_residual_unexplained_ratio": float(best_eval.residual_unexplained_ratio),
                "best_num_superquadrics": int(best_eval.num_superquadrics),
                "num_recovered_quadrics": int(len(best_eval.quadrics)),
                "num_child_segments": int(len(best_eval.child_segments)),
                "inlier_points": int(best_eval.inlier_count),
            }
        )

    final_distance_fit = float("inf") if total_inliers == 0 else total_weighted_dist / total_inliers
    global_coverage_ratio = (
        0.0 if point_cloud_working.shape[0] == 0 else float(total_inliers / point_cloud_working.shape[0])
    )
    total_residual_points = 0
    for _, groups in point_outlier.items():
        for arr in groups:
            arr_np = np.asarray(arr)
            if arr_np.ndim >= 1:
                total_residual_points += int(arr_np.shape[0])
    global_residual_unexplained_ratio = (
        0.0 if point_cloud_working.shape[0] == 0 else float(total_residual_points / point_cloud_working.shape[0])
    )
    total_num_superquadrics = int(len(list_quadrics))
    final_fitness = compose_fitness(
        distance_fit=final_distance_fit,
        coverage_ratio=global_coverage_ratio,
        residual_unexplained_ratio=global_residual_unexplained_ratio,
        num_superquadrics=total_num_superquadrics,
        fitness_mode=fitness_mode,
        lambda_cov=lambda_cov,
        lambda_out=lambda_out,
        lambda_comp=lambda_comp,
    )
    summary = {
        "num_superquadrics": len(list_quadrics),
        "inlier_points_total": int(total_inliers),
        "coverage_ratio": float(global_coverage_ratio),
        "residual_unexplained_ratio": float(global_residual_unexplained_ratio),
        "distance_fit": float(final_distance_fit),
        "point_seg_count_per_layer": {str(k): len(v) for k, v in point_seg.items()},
        "point_inlier_count_per_layer": {str(k): len(v) for k, v in point_inlier.items()},
        "point_outlier_count_per_layer": {str(k): len(v) for k, v in point_outlier.items()},
    }

    if norm_meta is not None and output_in_original_scale:
        denormalize_superquadrics_inplace(list_quadrics, norm_meta)
        _denormalize_nested_point_dict(point_seg, norm_meta)
        _denormalize_nested_point_dict(point_inlier, norm_meta)
        _denormalize_nested_point_dict(point_outlier, norm_meta)
        point_cloud_used = denormalize_points(point_cloud_working, norm_meta)
    else:
        point_cloud_used = point_cloud_working

    return {
        "fitness": float(final_fitness),
        "point_seg": point_seg,
        "point_inlier": point_inlier,
        "point_outlier": point_outlier,
        "point_seg_indices": point_seg_indices,
        "point_inlier_indices": point_inlier_indices,
        "point_outlier_indices": point_outlier_indices,
        "list_quadrics": list_quadrics,
        "primitive_assignments": primitive_assignments,
        "point_cloud_used": point_cloud_used,
        "summary": summary,
        "layer_histories": layer_histories,
        "layer_best_hparams": layer_best_hparams,
        "layer_best_fitness": layer_best_fitness,
        "layer_best_distance_fit": layer_best_distance_fit,
        "layer_best_coverage_ratio": layer_best_coverage_ratio,
        "layer_best_residual_unexplained_ratio": layer_best_residual_unexplained_ratio,
        "layer_best_num_superquadrics": layer_best_num_superquadrics,
        "layer_timing": layer_timing,
        "fitness_details": {
            "mode": fitness_mode,
            "lambda_cov": float(lambda_cov),
            "lambda_out": float(lambda_out),
            "lambda_comp": float(lambda_comp),
            "min_coverage": float(min_coverage),
            "lambda_min_coverage": float(lambda_min_coverage),
            "distance_fit": float(final_distance_fit),
            "coverage_ratio": float(global_coverage_ratio),
            "residual_unexplained_ratio": float(global_residual_unexplained_ratio),
            "num_superquadrics": int(total_num_superquadrics),
            "fitness": float(final_fitness),
        },
        "normalization": {
            "enabled": bool(global_normalize),
            "method": global_norm_method if global_normalize else "none",
            "output_in_original_scale": bool(output_in_original_scale),
            "meta": None if norm_meta is None else norm_meta.to_jsonable(),
        },
        "search_config": {
            "method": "cmaes",
            "popsize": int(cma_cfg.popsize),
            "maxiter": int(cma_cfg.maxiter),
            "sigma": float(cma_cfg.sigma),
        },
    }


def run_layerwise_gridsearch(
    point_cloud: np.ndarray,
    base_hp: RecoverHyperParams,
    lb: np.ndarray,
    ub: np.ndarray,
    decode_particle: Callable[[np.ndarray, RecoverHyperParams], RecoverHyperParams],
    grid_cfg: GridSearchConfig,
    global_normalize: bool = False,
    global_norm_method: str = "ems_matlab",
    output_in_original_scale: bool = True,
    fitness_mode: str = "legacy",
    lambda_cov: float = 0.0,
    lambda_out: float = 0.0,
    lambda_comp: float = 0.0,
    min_coverage: float = 0.0,
    lambda_min_coverage: float = 0.0,
) -> dict:
    def _denormalize_nested_point_dict(point_dict: dict, meta: NormalizationMeta) -> None:
        for key, groups in point_dict.items():
            if not isinstance(groups, list):
                continue
            converted = []
            for arr in groups:
                arr_np = np.asarray(arr, dtype=float)
                if arr_np.size == 0:
                    converted.append(arr_np.reshape((-1, 3)))
                else:
                    converted.append(denormalize_points(arr_np, meta))
            point_dict[key] = converted

    norm_meta: NormalizationMeta | None = None
    if global_normalize:
        point_cloud_working, norm_meta = normalize_points(point_cloud, method=global_norm_method)
    else:
        point_cloud_working = np.asarray(point_cloud, dtype=float)

    point_seg = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_inlier = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_outlier = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_seg_indices = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_inlier_indices = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_outlier_indices = {key: [] for key in range(0, base_hp.max_layer + 1)}
    point_seg[0] = [point_cloud_working]
    point_seg_indices[0] = [np.arange(point_cloud_working.shape[0], dtype=np.int64)]

    list_quadrics = []
    primitive_assignments: list[dict] = []
    total_weighted_dist = 0.0
    total_inliers = 0

    layer_histories: list[list[float]] = []
    layer_best_hparams: list[dict] = []
    layer_best_fitness: list[float] = []
    layer_best_distance_fit: list[float] = []
    layer_best_coverage_ratio: list[float] = []
    layer_best_residual_unexplained_ratio: list[float] = []
    layer_best_num_superquadrics: list[int] = []
    layer_timing: list[dict] = []

    dim = int(lb.shape[0])
    if dim <= 0:
        raise ValueError("Search space must have at least one dimension.")
    steps = int(grid_cfg.steps_per_dim)
    if steps < 2:
        raise ValueError("grid_cfg.steps_per_dim must be >= 2.")

    grid_axes = [np.linspace(lb[d], ub[d], steps) for d in range(dim)]
    planned_eval_count = int(np.prod([len(axis) for axis in grid_axes]))

    for layer_idx in range(base_hp.max_layer):
        layer_t_start = timeit.default_timer()
        segments = point_seg[layer_idx]
        segment_indices = point_seg_indices[layer_idx]
        if len(segments) == 0:
            layer_histories.append([])
            layer_best_hparams.append({})
            layer_best_fitness.append(float("inf"))
            layer_best_distance_fit.append(float("inf"))
            layer_best_coverage_ratio.append(0.0)
            layer_best_residual_unexplained_ratio.append(0.0)
            layer_best_num_superquadrics.append(0)
            layer_timing.append(
                {
                    "layer": int(layer_idx),
                    "status": "skipped_empty",
                    "num_input_segments": 0,
                    "grid_evaluations_planned": int(planned_eval_count),
                    "grid_evaluations_effective": 0,
                    "grid_search_time_ms": 0.0,
                    "best_eval_time_ms": 0.0,
                    "layer_total_time_ms": 0.0,
                    "best_fitness": float("inf"),
                    "best_distance_fit": float("inf"),
                    "best_coverage_ratio": 0.0,
                    "best_residual_unexplained_ratio": 0.0,
                    "best_num_superquadrics": 0,
                    "num_recovered_quadrics": 0,
                    "num_child_segments": 0,
                    "inlier_points": 0,
                }
            )
            continue

        cache: dict[tuple, tuple[RecoverHyperParams, LayerEval]] = {}
        best_eval: LayerEval | None = None
        best_hp_layer: RecoverHyperParams | None = None
        best_f = float("inf")
        history: list[float] = []

        grid_t_start = timeit.default_timer()
        for candidate in product(*grid_axes):
            x = np.asarray(candidate, dtype=float)
            hp = decode_particle(x, base_hp)
            hp_key = (
                round(hp.outlier_ratio, 8),
                round(hp.sigma, 8),
                round(hp.eps, 8),
                int(hp.min_points),
            )
            if hp_key in cache:
                eval_result = cache[hp_key][1]
            else:
                try:
                    eval_result = evaluate_layer(
                        segments=segments,
                        hp=hp,
                        layer_idx=layer_idx,
                        total_points=point_cloud_working.shape[0],
                        segment_indices=segment_indices,
                        fitness_mode=fitness_mode,
                        lambda_cov=lambda_cov,
                        lambda_out=lambda_out,
                        lambda_comp=lambda_comp,
                        min_coverage=min_coverage,
                        lambda_min_coverage=lambda_min_coverage,
                    )
                except Exception:
                    continue
                cache[hp_key] = (hp, eval_result)

            if best_eval is None or eval_result.fitness < best_f:
                best_eval = eval_result
                best_hp_layer = hp
                best_f = float(eval_result.fitness)
            history.append(float(best_f))
        grid_t_end = timeit.default_timer()

        if best_eval is None or best_hp_layer is None:
            layer_histories.append(history)
            layer_best_hparams.append({})
            layer_best_fitness.append(float("inf"))
            layer_best_distance_fit.append(float("inf"))
            layer_best_coverage_ratio.append(0.0)
            layer_best_residual_unexplained_ratio.append(0.0)
            layer_best_num_superquadrics.append(0)
            layer_t_end = timeit.default_timer()
            layer_timing.append(
                {
                    "layer": int(layer_idx),
                    "status": "failed_no_valid_candidate",
                    "num_input_segments": int(len(segments)),
                    "grid_evaluations_planned": int(planned_eval_count),
                    "grid_evaluations_effective": int(len(cache)),
                    "grid_search_time_ms": float((grid_t_end - grid_t_start) * 1000.0),
                    "best_eval_time_ms": 0.0,
                    "layer_total_time_ms": float((layer_t_end - layer_t_start) * 1000.0),
                    "best_fitness": float("inf"),
                    "best_distance_fit": float("inf"),
                    "best_coverage_ratio": 0.0,
                    "best_residual_unexplained_ratio": 0.0,
                    "best_num_superquadrics": 0,
                    "num_recovered_quadrics": 0,
                    "num_child_segments": 0,
                    "inlier_points": 0,
                }
            )
            point_seg[layer_idx + 1] = []
            point_seg_indices[layer_idx + 1] = []
            point_inlier[layer_idx] = []
            point_inlier_indices[layer_idx] = []
            point_outlier[layer_idx] = segments
            point_outlier_indices[layer_idx] = segment_indices
            continue

        best_eval_t_start = timeit.default_timer()
        _ = best_eval
        best_eval_t_end = timeit.default_timer()

        point_seg[layer_idx + 1] = best_eval.child_segments
        point_seg_indices[layer_idx + 1] = best_eval.child_segment_indices
        point_inlier[layer_idx] = best_eval.inlier_groups
        point_inlier_indices[layer_idx] = best_eval.inlier_index_groups
        point_outlier[layer_idx] = best_eval.outlier_groups
        point_outlier_indices[layer_idx] = best_eval.outlier_index_groups
        list_quadrics.extend(best_eval.quadrics)
        for assignment in best_eval.primitive_assignments:
            assignment_with_id = dict(assignment)
            assignment_with_id["primitive_id"] = len(primitive_assignments)
            primitive_assignments.append(assignment_with_id)

        if best_eval.inlier_count > 0 and np.isfinite(best_eval.fitness):
            total_weighted_dist += best_eval.distance_fit * best_eval.inlier_count
            total_inliers += best_eval.inlier_count

        layer_histories.append(history)
        layer_best_hparams.append(
            {
                "outlier_ratio": best_hp_layer.outlier_ratio,
                "sigma": best_hp_layer.sigma,
                "max_layer": best_hp_layer.max_layer,
                "tau_in": best_hp_layer.tau_in,
                "tau_split": best_hp_layer.tau_split,
                "eps": best_hp_layer.eps,
                "min_points": best_hp_layer.min_points,
            }
        )
        layer_best_fitness.append(float(best_f))
        layer_best_distance_fit.append(float(best_eval.distance_fit))
        layer_best_coverage_ratio.append(float(best_eval.coverage_ratio))
        layer_best_residual_unexplained_ratio.append(float(best_eval.residual_unexplained_ratio))
        layer_best_num_superquadrics.append(int(best_eval.num_superquadrics))
        layer_t_end = timeit.default_timer()
        layer_timing.append(
            {
                "layer": int(layer_idx),
                "status": "ok",
                "num_input_segments": int(len(segments)),
                "grid_evaluations_planned": int(planned_eval_count),
                "grid_evaluations_effective": int(len(cache)),
                "grid_search_time_ms": float((grid_t_end - grid_t_start) * 1000.0),
                "best_eval_time_ms": float((best_eval_t_end - best_eval_t_start) * 1000.0),
                "layer_total_time_ms": float((layer_t_end - layer_t_start) * 1000.0),
                "best_fitness": float(best_f),
                "best_distance_fit": float(best_eval.distance_fit),
                "best_coverage_ratio": float(best_eval.coverage_ratio),
                "best_residual_unexplained_ratio": float(best_eval.residual_unexplained_ratio),
                "best_num_superquadrics": int(best_eval.num_superquadrics),
                "num_recovered_quadrics": int(len(best_eval.quadrics)),
                "num_child_segments": int(len(best_eval.child_segments)),
                "inlier_points": int(best_eval.inlier_count),
            }
        )

    final_distance_fit = float("inf") if total_inliers == 0 else total_weighted_dist / total_inliers
    global_coverage_ratio = (
        0.0 if point_cloud_working.shape[0] == 0 else float(total_inliers / point_cloud_working.shape[0])
    )
    total_residual_points = 0
    for _, groups in point_outlier.items():
        for arr in groups:
            arr_np = np.asarray(arr)
            if arr_np.ndim >= 1:
                total_residual_points += int(arr_np.shape[0])
    global_residual_unexplained_ratio = (
        0.0 if point_cloud_working.shape[0] == 0 else float(total_residual_points / point_cloud_working.shape[0])
    )
    total_num_superquadrics = int(len(list_quadrics))
    final_fitness = compose_fitness(
        distance_fit=final_distance_fit,
        coverage_ratio=global_coverage_ratio,
        residual_unexplained_ratio=global_residual_unexplained_ratio,
        num_superquadrics=total_num_superquadrics,
        fitness_mode=fitness_mode,
        lambda_cov=lambda_cov,
        lambda_out=lambda_out,
        lambda_comp=lambda_comp,
    )
    summary = {
        "num_superquadrics": len(list_quadrics),
        "inlier_points_total": int(total_inliers),
        "coverage_ratio": float(global_coverage_ratio),
        "residual_unexplained_ratio": float(global_residual_unexplained_ratio),
        "distance_fit": float(final_distance_fit),
        "point_seg_count_per_layer": {str(k): len(v) for k, v in point_seg.items()},
        "point_inlier_count_per_layer": {str(k): len(v) for k, v in point_inlier.items()},
        "point_outlier_count_per_layer": {str(k): len(v) for k, v in point_outlier.items()},
    }

    if norm_meta is not None and output_in_original_scale:
        denormalize_superquadrics_inplace(list_quadrics, norm_meta)
        _denormalize_nested_point_dict(point_seg, norm_meta)
        _denormalize_nested_point_dict(point_inlier, norm_meta)
        _denormalize_nested_point_dict(point_outlier, norm_meta)
        point_cloud_used = denormalize_points(point_cloud_working, norm_meta)
    else:
        point_cloud_used = point_cloud_working

    return {
        "fitness": float(final_fitness),
        "point_seg": point_seg,
        "point_inlier": point_inlier,
        "point_outlier": point_outlier,
        "point_seg_indices": point_seg_indices,
        "point_inlier_indices": point_inlier_indices,
        "point_outlier_indices": point_outlier_indices,
        "list_quadrics": list_quadrics,
        "primitive_assignments": primitive_assignments,
        "point_cloud_used": point_cloud_used,
        "summary": summary,
        "layer_histories": layer_histories,
        "layer_best_hparams": layer_best_hparams,
        "layer_best_fitness": layer_best_fitness,
        "layer_best_distance_fit": layer_best_distance_fit,
        "layer_best_coverage_ratio": layer_best_coverage_ratio,
        "layer_best_residual_unexplained_ratio": layer_best_residual_unexplained_ratio,
        "layer_best_num_superquadrics": layer_best_num_superquadrics,
        "layer_timing": layer_timing,
        "fitness_details": {
            "mode": fitness_mode,
            "lambda_cov": float(lambda_cov),
            "lambda_out": float(lambda_out),
            "lambda_comp": float(lambda_comp),
            "min_coverage": float(min_coverage),
            "lambda_min_coverage": float(lambda_min_coverage),
            "distance_fit": float(final_distance_fit),
            "coverage_ratio": float(global_coverage_ratio),
            "residual_unexplained_ratio": float(global_residual_unexplained_ratio),
            "num_superquadrics": int(total_num_superquadrics),
            "fitness": float(final_fitness),
        },
        "normalization": {
            "enabled": bool(global_normalize),
            "method": global_norm_method if global_normalize else "none",
            "output_in_original_scale": bool(output_in_original_scale),
            "meta": None if norm_meta is None else norm_meta.to_jsonable(),
        },
        "search_config": {
            "method": "grid",
            "steps_per_dim": int(steps),
            "dimensions": int(dim),
            "planned_evaluations_per_layer": int(planned_eval_count),
            "fitness_mode": fitness_mode,
        },
    }
