"""Geometry utilities for recovered superquadric models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def signed_power(value: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(value) * np.abs(value) ** exponent


@dataclass(frozen=True)
class Superquadric:
    id: int
    shape: np.ndarray
    scale: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    @property
    def volume_proxy(self) -> float:
        return float(np.prod(self.scale))

    def world_to_local(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=float) - self.translation) @ self.rotation

    def local_to_world(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float) @ self.rotation.T + self.translation

    def implicit(self, points: np.ndarray) -> np.ndarray:
        local = self.world_to_local(points)
        e1, e2 = np.maximum(self.shape, 1e-4)
        a1, a2, a3 = np.maximum(self.scale, 1e-6)
        x, y, z = local[:, 0], local[:, 1], local[:, 2]
        xy = (np.abs(x / a1) ** (2.0 / e2) + np.abs(y / a2) ** (2.0 / e2))
        return xy ** (e2 / e1) + np.abs(z / a3) ** (2.0 / e1)

    def sample_surface(self, n_eta: int = 36, n_omega: int = 72) -> np.ndarray:
        e1, e2 = (float(x) for x in self.shape)
        a1, a2, a3 = (float(x) for x in self.scale)
        eta = np.linspace(-np.pi / 2, np.pi / 2, n_eta)
        omega = np.linspace(-np.pi, np.pi, n_omega)
        eta_grid, omega_grid = np.meshgrid(eta, omega, indexing="ij")
        ce = signed_power(np.cos(eta_grid), e1)
        se = signed_power(np.sin(eta_grid), e1)
        co = signed_power(np.cos(omega_grid), e2)
        so = signed_power(np.sin(omega_grid), e2)
        local = np.stack((a1 * ce * co, a2 * ce * so, a3 * se), axis=-1)
        return self.local_to_world(local.reshape(-1, 3))

    def normal_at(self, points: np.ndarray) -> np.ndarray:
        local = self.world_to_local(points)
        normal_local = local / np.maximum(self.scale, 1e-6) ** 2
        normal = normal_local @ self.rotation.T
        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        return normal / np.maximum(norm, 1e-9)


class SuperquadricModel:
    def __init__(self, object_name: str, superquadrics: list[Superquadric]):
        self.object_name = object_name
        self.superquadrics = superquadrics
        self.surface_points_by_id = {
            sq.id: sq.sample_surface() for sq in self.superquadrics
        }
        self.surface_points = np.concatenate(list(self.surface_points_by_id.values()), axis=0)
        self.surface_tree = cKDTree(self.surface_points)
        self.aabb_min = self.surface_points.min(axis=0)
        self.aabb_max = self.surface_points.max(axis=0)
        self.center = 0.5 * (self.aabb_min + self.aabb_max)
        self.extent = self.aabb_max - self.aabb_min
        self.reference_length = float(max(np.max(self.extent), 1e-6))

    @classmethod
    def from_json(cls, path: Path) -> "SuperquadricModel":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        object_name = path.name.replace("_20cm.json", "").replace(".json", "")
        superquadrics = []
        for index, raw in enumerate(data["superquadrics"], start=1):
            sq_id = int(raw.get("id", index))
            superquadrics.append(
                Superquadric(
                    id=sq_id,
                    shape=np.asarray(raw["shape"], dtype=float),
                    scale=np.asarray(raw["scale"], dtype=float),
                    rotation=Rotation.from_euler("ZYX", raw["euler"]).as_matrix(),
                    translation=np.asarray(raw["translation"], dtype=float),
                )
            )
        return cls(object_name, superquadrics)

    def sq_by_id(self, sq_id: int) -> Superquadric:
        for sq in self.superquadrics:
            if sq.id == sq_id:
                return sq
        raise KeyError(f"Unknown SQ id: {sq_id}")

    def target_ids_by_size(self, limit: int) -> list[int]:
        ordered = sorted(self.superquadrics, key=lambda sq: sq.volume_proxy, reverse=True)
        return [sq.id for sq in ordered[:limit]]

    def nearest_surface_distance(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distance, index = self.surface_tree.query(np.asarray(points, dtype=float))
        return distance, self.surface_points[index]

    def union_inside_margin(self, points: np.ndarray) -> np.ndarray:
        values = np.column_stack([sq.implicit(points) for sq in self.superquadrics])
        return 1.0 - values.min(axis=1)

    def target_anchor_pair(self, target_id: int, closing_axis: np.ndarray) -> dict:
        sq = self.sq_by_id(target_id)
        points = self.surface_points_by_id[target_id]
        axis = np.asarray(closing_axis, dtype=float)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        projection = (points - sq.translation) @ axis
        low = int(np.argmin(projection))
        high = int(np.argmax(projection))
        anchors = np.vstack((points[low], points[high]))
        normals = sq.normal_at(anchors)
        return {
            "left_anchor": anchors[0],
            "right_anchor": anchors[1],
            "left_normal": normals[0],
            "right_normal": normals[1],
        }
