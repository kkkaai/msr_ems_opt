"""Pregrasp initialization around target superquadrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simple_hand import ParallelJawPose, make_rotation
from .superquadric_geometry import SuperquadricModel


@dataclass
class PregraspCandidate:
    object_name: str
    target_sq: int
    pose: ParallelJawPose
    approach_axis: np.ndarray
    closing_axis: np.ndarray


def principal_axes_for_target(model: SuperquadricModel, target_sq: int) -> list[np.ndarray]:
    sq = model.sq_by_id(target_sq)
    axes = [sq.rotation[:, 0], sq.rotation[:, 1], sq.rotation[:, 2]]
    return axes + [-axis for axis in axes]


def generate_pregrasps(
    model: SuperquadricModel,
    target_limit: int = 5,
    pregrasp_offset_m: float = 0.045,
    clearance_m: float = 0.018,
) -> list[PregraspCandidate]:
    candidates: list[PregraspCandidate] = []
    target_ids = model.target_ids_by_size(min(target_limit, len(model.superquadrics)))

    for target_id in target_ids:
        sq = model.sq_by_id(target_id)
        axes = principal_axes_for_target(model, target_id)
        for approach_axis in axes:
            approach_axis = approach_axis / max(float(np.linalg.norm(approach_axis)), 1e-9)
            closing_options = sorted(
                axes[:3],
                key=lambda axis: abs(float(np.dot(axis, approach_axis))),
            )
            closing_axis = closing_options[0]
            closing_axis = closing_axis - approach_axis * float(np.dot(closing_axis, approach_axis))
            closing_axis = closing_axis / max(float(np.linalg.norm(closing_axis)), 1e-9)

            anchors = model.target_anchor_pair(target_id, closing_axis)
            midpoint = 0.5 * (anchors["left_anchor"] + anchors["right_anchor"])
            span = float(np.linalg.norm(anchors["right_anchor"] - anchors["left_anchor"]))
            width = float(np.clip(span + clearance_m, 0.025, 0.16))
            rotation = make_rotation(closing_axis, approach_axis)
            translation = midpoint + approach_axis * pregrasp_offset_m
            candidates.append(
                PregraspCandidate(
                    object_name=model.object_name,
                    target_sq=target_id,
                    pose=ParallelJawPose(translation=translation, rotation_matrix=rotation, width=width),
                    approach_axis=approach_axis,
                    closing_axis=closing_axis,
                )
            )
    return candidates
