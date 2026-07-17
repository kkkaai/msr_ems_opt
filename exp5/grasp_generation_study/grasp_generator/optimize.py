"""Pregrasp-local grasp optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .pregrasp import PregraspCandidate
from .simple_hand import ParallelJawHand, ParallelJawPose
from .superquadric_geometry import SuperquadricModel


@dataclass
class GraspResult:
    object_name: str
    target_sq: int
    pregrasp: ParallelJawPose
    grasp: ParallelJawPose
    objective: float
    contact_distance_m: float
    max_penetration_proxy_m: float
    q1_proxy: float
    success: bool


class GraspOptimizer:
    def __init__(self, model: SuperquadricModel, hand: ParallelJawHand | None = None):
        self.model = model
        self.hand = hand or ParallelJawHand()
        self.L = max(float(model.reference_length), 1e-6)

    def optimize(self, candidate: PregraspCandidate, maxiter: int = 180) -> GraspResult:
        start = candidate.pose.as_vector()
        anchors = self.model.target_anchor_pair(candidate.target_sq, candidate.closing_axis)
        anchor_points = np.vstack((anchors["left_anchor"], anchors["right_anchor"]))
        anchor_normals = np.vstack((anchors["left_normal"], anchors["right_normal"]))

        def unpack(vector: np.ndarray) -> ParallelJawPose:
            pose = ParallelJawPose.from_vector(vector)
            pose.width = float(np.clip(pose.width, *self.hand.width_bounds))
            return pose

        def energy(vector: np.ndarray) -> float:
            pose = unpack(vector)
            contacts = self.hand.contact_points(pose)
            tips = contacts[:2]
            tip_error = np.mean(np.sum((tips - anchor_points) ** 2, axis=1)) / (self.L**2)
            aux_distance, _ = self.model.nearest_surface_distance(contacts[2:])
            aux_error = float(np.mean((aux_distance / self.L) ** 2))
            collision = self.hand.collision_points(pose)
            inside_margin = self.model.union_inside_margin(collision)
            collision_distance, _ = self.model.nearest_surface_distance(collision)
            penetration = float(np.mean(((inside_margin > 0.0) * collision_distance / self.L) ** 2))
            width_penalty = 0.01 * ((pose.width - candidate.pose.width) / self.L) ** 2
            pose_drift = 0.02 * float(np.sum((pose.translation - start[:3]) ** 2)) / (self.L**2)
            return 100.0 * tip_error + 15.0 * aux_error + 100.0 * penetration + width_penalty + pose_drift

        bounds = [
            (start[0] - 0.08, start[0] + 0.08),
            (start[1] - 0.08, start[1] + 0.08),
            (start[2] - 0.08, start[2] + 0.08),
            (start[3] - 0.6, start[3] + 0.6),
            (start[4] - 0.6, start[4] + 0.6),
            (start[5] - 0.6, start[5] + 0.6),
            self.hand.width_bounds,
        ]
        result = minimize(
            energy,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-9, "maxls": 30},
        )
        grasp = unpack(result.x)
        return self.evaluate(candidate, grasp, float(result.fun), anchor_normals)

    def evaluate(
        self,
        candidate: PregraspCandidate,
        grasp: ParallelJawPose,
        objective: float,
        anchor_normals: np.ndarray | None = None,
    ) -> GraspResult:
        contacts = self.hand.contact_points(grasp)
        distance, nearest = self.model.nearest_surface_distance(contacts)
        collision = self.hand.collision_points(grasp)
        inside_margin = self.model.union_inside_margin(collision)
        collision_distance, _ = self.model.nearest_surface_distance(collision)
        inside_distance = collision_distance[inside_margin > 0.0]
        penetration_proxy = float(np.max(inside_distance)) if len(inside_distance) else 0.0
        contact_distance = float(np.mean(distance[:2]))
        if anchor_normals is None:
            normals = self.model.sq_by_id(candidate.target_sq).normal_at(nearest[:2])
        else:
            normals = anchor_normals
        normal_balance = max(0.0, -float(np.dot(normals[0], normals[1])))
        span = float(np.linalg.norm(contacts[1] - contacts[0]))
        q1_proxy = normal_balance * min(1.0, span / max(0.02, 0.5 * self.L))
        success = contact_distance <= 0.006 and penetration_proxy <= 0.004 and q1_proxy >= 0.15
        return GraspResult(
            object_name=self.model.object_name,
            target_sq=candidate.target_sq,
            pregrasp=candidate.pose,
            grasp=grasp,
            objective=objective,
            contact_distance_m=contact_distance,
            max_penetration_proxy_m=penetration_proxy,
            q1_proxy=float(q1_proxy),
            success=success,
        )
