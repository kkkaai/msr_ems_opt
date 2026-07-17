from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass
class NormalizationMeta:
    enabled: bool
    method: str
    center: np.ndarray
    scale: float

    def to_jsonable(self) -> dict:
        data = asdict(self)
        data["center"] = np.asarray(self.center, dtype=float).tolist()
        data["scale"] = float(self.scale)
        return data


def _validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N, 3), got {points.shape}")
    return points


def normalize_points(points: np.ndarray, method: str = "ems_matlab") -> tuple[np.ndarray, NormalizationMeta]:
    points = _validate_points(points)
    method = method.lower()

    if method not in {"ems_matlab", "unit_box", "unit_sphere"}:
        raise ValueError(f"Unsupported normalization method: {method}")

    if method == "ems_matlab":
        center = np.mean(points, axis=0)
        shifted = points - center
        max_length = float(np.max(shifted))
        if max_length <= 1e-12:
            max_length = float(np.max(np.abs(shifted)))
        if max_length <= 1e-12:
            max_length = 1.0
        scale = max_length / 10.0
        if scale <= 1e-12:
            scale = 1.0
        normalized = shifted / scale
        meta = NormalizationMeta(enabled=True, method=method, center=center, scale=scale)
        return normalized, meta

    if method == "unit_box":
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        center = (bbox_min + bbox_max) / 2.0
        extent = bbox_max - bbox_min
        scale = float(np.max(extent))
        if scale <= 1e-12:
            scale = 1.0
        normalized = (points - center) / scale
        meta = NormalizationMeta(enabled=True, method=method, center=center, scale=scale)
        return normalized, meta

    # unit_sphere
    center = np.mean(points, axis=0)
    shifted = points - center
    radius = float(np.max(np.linalg.norm(shifted, axis=1)))
    scale = radius if radius > 1e-12 else 1.0
    normalized = shifted / scale
    meta = NormalizationMeta(enabled=True, method=method, center=center, scale=scale)
    return normalized, meta


def denormalize_points(points: np.ndarray, meta: NormalizationMeta | None) -> np.ndarray:
    points = _validate_points(points)
    if meta is None or not meta.enabled:
        return points
    return points * float(meta.scale) + np.asarray(meta.center, dtype=float)


def denormalize_superquadrics_inplace(quadrics: Sequence, meta: NormalizationMeta | None) -> None:
    if meta is None or not meta.enabled:
        return
    scale = float(meta.scale)
    center = np.asarray(meta.center, dtype=float)
    for sq in quadrics:
        sq.scale = np.asarray(sq.scale, dtype=float) * scale
        sq.translation = np.asarray(sq.translation, dtype=float) * scale + center
