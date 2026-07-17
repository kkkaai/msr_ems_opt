"""A small analytic hand model used before a real dexterous-hand FK adapter exists."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class ParallelJawPose:
    translation: np.ndarray
    rotation_matrix: np.ndarray
    width: float

    def as_vector(self) -> np.ndarray:
        return np.r_[self.translation, Rotation.from_matrix(self.rotation_matrix).as_rotvec(), self.width]

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "ParallelJawPose":
        vector = np.asarray(vector, dtype=float)
        return cls(
            translation=vector[:3].copy(),
            rotation_matrix=Rotation.from_rotvec(vector[3:6]).as_matrix(),
            width=float(vector[6]),
        )


class ParallelJawHand:
    """Simple two-finger contact model with a palm clearance point.

    The x-axis is the closing direction, and the z-axis points from the object
    toward the pregrasp side.
    """

    width_bounds = (0.018, 0.18)

    def contact_points(self, pose: ParallelJawPose) -> np.ndarray:
        half = 0.5 * pose.width
        local = np.array(
            [
                [-half, 0.0, 0.0],
                [half, 0.0, 0.0],
                [-half, 0.012, 0.006],
                [half, -0.012, 0.006],
            ],
            dtype=float,
        )
        return local @ pose.rotation_matrix.T + pose.translation

    def collision_points(self, pose: ParallelJawPose) -> np.ndarray:
        half = 0.5 * pose.width
        xs = np.linspace(-half, half, 7)
        palm = np.column_stack((xs, np.zeros_like(xs), np.full_like(xs, 0.035)))
        left = np.column_stack((np.full(6, -half), np.linspace(-0.016, 0.016, 6), np.linspace(0.0, 0.035, 6)))
        right = np.column_stack((np.full(6, half), np.linspace(-0.016, 0.016, 6), np.linspace(0.0, 0.035, 6)))
        local = np.vstack((palm, left, right))
        return local @ pose.rotation_matrix.T + pose.translation

    def segment_points(self, pose: ParallelJawPose) -> tuple[np.ndarray, np.ndarray]:
        points = self.contact_points(pose)
        palm_center = pose.translation + pose.rotation_matrix @ np.array([0.0, 0.0, 0.035])
        return points[:2], palm_center


def make_rotation(closing_axis: np.ndarray, approach_axis: np.ndarray) -> np.ndarray:
    x_axis = np.asarray(closing_axis, dtype=float)
    z_axis = np.asarray(approach_axis, dtype=float)
    z_axis = z_axis / max(float(np.linalg.norm(z_axis)), 1e-9)
    x_axis = x_axis - z_axis * float(np.dot(x_axis, z_axis))
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-9)
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack((x_axis, y_axis, z_axis))
