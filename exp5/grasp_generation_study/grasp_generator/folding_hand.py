"""Approximate kinematic adapter for ``hand/folding_hand_right``.

This adapter intentionally avoids a hard dependency on MuJoCo. It parses the
MJCF body tree and evaluates forward kinematics for the folding hand using the
actuator-style controls defined in ``folding_hand_right.xml``. It is sufficient for
pregrasp-local geometric grasp generation and visualization. Dynamic validation
should still be done with MuJoCo after installing the simulator bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


RIGHT_FINGER_TIP_BODIES = {
    "th": "th_4",
    "ff": "ff_3",
    "mf": "mf_2",
    "rf": "rf_3",
    "lf": "lf_3",
}

RIGHT_TIP_LOCAL_OFFSETS = {
    "th": np.array([0.018, 0.0, 0.018]),
    "ff": np.array([0.026, 0.0, 0.0]),
    "mf": np.array([0.030, 0.0, 0.0]),
    "rf": np.array([0.027, 0.0, -0.002]),
    "lf": np.array([0.021, 0.0, -0.004]),
}

RIGHT_CONTROL_NAMES = ["plam_th", "th", "plam_other", "ff", "mf", "rf", "lf"]
RIGHT_CONTROL_LIMITS = {
    "plam_th": (0.0, 1.5708),
    "th": (0.0, 1.0),
    "plam_other": (0.0, 1.5708),
    "ff": (0.0, 2.4),
    "mf": (0.0, 2.4),
    "rf": (0.0, 2.4),
    "lf": (0.0, 2.4),
}


@dataclass
class BodyNode:
    name: str
    parent: str | None
    pos: np.ndarray
    quat_wxyz: np.ndarray
    joint_name: str | None
    joint_axis: np.ndarray | None
    children: list[str]


@dataclass
class FoldingHandPose:
    translation: np.ndarray
    rotation_matrix: np.ndarray
    controls: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.r_[self.translation, Rotation.from_matrix(self.rotation_matrix).as_rotvec(), self.controls]

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "FoldingHandPose":
        vector = np.asarray(vector, dtype=float)
        return cls(
            translation=vector[:3].copy(),
            rotation_matrix=Rotation.from_rotvec(vector[3:6]).as_matrix(),
            controls=vector[6:].copy(),
        )


def _parse_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    return np.asarray([float(x) for x in text.split()], dtype=float)


def _quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def _parse_mjcf(xml_path: Path) -> dict[str, BodyNode]:
    tree = ET.parse(xml_path)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise ValueError(f"Missing worldbody in {xml_path}")
    nodes: dict[str, BodyNode] = {}

    def visit(body: ET.Element, parent: str | None) -> None:
        name = body.attrib["name"]
        joint = body.find("joint")
        joint_name = joint.attrib["name"] if joint is not None else None
        joint_axis = _parse_vec(joint.attrib.get("axis"), (0.0, 0.0, 1.0)) if joint is not None else None
        node = BodyNode(
            name=name,
            parent=parent,
            pos=_parse_vec(body.attrib.get("pos"), (0.0, 0.0, 0.0)),
            quat_wxyz=_parse_vec(body.attrib.get("quat"), (1.0, 0.0, 0.0, 0.0)),
            joint_name=joint_name,
            joint_axis=joint_axis,
            children=[],
        )
        nodes[name] = node
        if parent is not None:
            nodes[parent].children.append(name)
        for child in body.findall("body"):
            visit(child, name)

    for body in worldbody.findall("body"):
        visit(body, None)
    return nodes


class FoldingHandRightKinematics:
    def __init__(self, xml_path: Path):
        self.xml_path = Path(xml_path)
        self.nodes = _parse_mjcf(self.xml_path)
        self.body_meshes = self._parse_body_meshes(self.xml_path)
        self.mesh_dir = self.xml_path.parent / "meshes"
        self._mesh_cache: dict[str, np.ndarray] = {}
        self.root_name = "base_link"

    @classmethod
    def default(cls) -> "FoldingHandRightKinematics":
        repo_root = Path(__file__).resolve().parents[3]
        return cls(repo_root / "hand/folding_hand_right/folding_hand_right.xml")

    def controls_to_joint_angles(self, controls: np.ndarray) -> dict[str, float]:
        ctrl = {
            name: float(np.clip(value, *RIGHT_CONTROL_LIMITS[name]))
            for name, value in zip(RIGHT_CONTROL_NAMES, controls)
        }
        return {
            "th_1_joint": ctrl["plam_th"],
            "th_2_joint": 0.5 * ctrl["th"],
            "th_3_joint": 0.8 * ctrl["th"],
            "th_4_joint": 0.8 * ctrl["th"],
            "ff_1_joint": ctrl["plam_other"],
            "ff_2_joint": ctrl["ff"],
            "ff_3_joint": ctrl["ff"],
            "mf_1_joint": ctrl["mf"],
            "mf_2_joint": ctrl["mf"],
            "rf_1_joint": ctrl["plam_other"],
            "rf_2_joint": ctrl["rf"],
            "rf_3_joint": ctrl["rf"],
            "lf_1_joint": min(2.0 * ctrl["plam_other"], RIGHT_CONTROL_LIMITS["plam_other"][1]),
            "lf_2_joint": ctrl["lf"],
            "lf_3_joint": ctrl["lf"],
        }

    def forward_kinematics(self, pose: FoldingHandPose) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        joint_angles = self.controls_to_joint_angles(pose.controls)
        transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        def visit(name: str, parent_translation: np.ndarray, parent_rotation: np.ndarray) -> None:
            node = self.nodes[name]
            rotation = parent_rotation @ _quat_to_matrix_wxyz(node.quat_wxyz)
            translation = parent_translation + parent_rotation @ node.pos
            if node.joint_name is not None and node.joint_axis is not None:
                angle = joint_angles.get(node.joint_name, 0.0)
                rotation = rotation @ _axis_angle(node.joint_axis, angle)
            transforms[name] = (translation, rotation)
            for child in node.children:
                visit(child, translation, rotation)

        visit(self.root_name, pose.translation, pose.rotation_matrix)
        return transforms

    def contact_points(self, pose: FoldingHandPose) -> np.ndarray:
        transforms = self.forward_kinematics(pose)
        points = []
        for finger, body_name in RIGHT_FINGER_TIP_BODIES.items():
            translation, rotation = transforms[body_name]
            points.append(translation + rotation @ RIGHT_TIP_LOCAL_OFFSETS[finger])
        return np.vstack(points)

    def collision_points(self, pose: FoldingHandPose) -> np.ndarray:
        transforms = self.forward_kinematics(pose)
        points = []
        for finger, body_name in RIGHT_FINGER_TIP_BODIES.items():
            translation, rotation = transforms[body_name]
            offset = RIGHT_TIP_LOCAL_OFFSETS[finger]
            for alpha in np.linspace(0.25, 1.0, 4):
                points.append(translation + rotation @ (alpha * offset))
        palm_offsets = [
            np.array([0.0, 0.000, 0.065]),
            np.array([0.0, 0.012, 0.045]),
            np.array([0.0, -0.012, 0.045]),
        ]
        for offset in palm_offsets:
            points.append(pose.translation + pose.rotation_matrix @ offset)
        return np.vstack(points)

    def open_controls(self) -> np.ndarray:
        return np.array([0.35, 0.08, 0.20, 0.12, 0.12, 0.12, 0.12], dtype=float)

    def mid_closed_controls(self) -> np.ndarray:
        return np.array([0.65, 0.75, 0.45, 1.05, 1.05, 1.05, 1.05], dtype=float)

    @staticmethod
    def _parse_body_meshes(xml_path: Path) -> dict[str, str]:
        tree = ET.parse(xml_path)
        result: dict[str, str] = {}
        for body in tree.getroot().findall(".//body"):
            body_name = body.attrib.get("name")
            if not body_name:
                continue
            geom = body.find("geom[@type='mesh']")
            if geom is not None and "mesh" in geom.attrib:
                result[body_name] = geom.attrib["mesh"]
        return result

    @staticmethod
    def _read_binary_stl(path: Path) -> np.ndarray:
        with path.open("rb") as handle:
            handle.read(80)
            count = int(np.frombuffer(handle.read(4), dtype="<u4")[0])
            dtype = np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attr", "<u2"),
                ]
            )
            data = np.fromfile(handle, dtype=dtype, count=count)
        return data["vertices"].astype(float)

    def mesh_triangles(self, mesh_name: str, max_triangles: int = 1400) -> np.ndarray:
        if mesh_name not in self._mesh_cache:
            path = self.mesh_dir / mesh_name
            triangles = self._read_binary_stl(path)
            if len(triangles) > max_triangles:
                step = int(np.ceil(len(triangles) / max_triangles))
                triangles = triangles[::step]
            self._mesh_cache[mesh_name] = triangles
        return self._mesh_cache[mesh_name]

    def transformed_mesh_triangles(self, pose: FoldingHandPose, max_triangles_per_mesh: int = 1000) -> list[np.ndarray]:
        transforms = self.forward_kinematics(pose)
        triangles_world = []
        for body_name, mesh_name in self.body_meshes.items():
            if body_name not in transforms:
                continue
            translation, rotation = transforms[body_name]
            triangles = self.mesh_triangles(mesh_name, max_triangles=max_triangles_per_mesh)
            triangles_world.append(triangles @ rotation.T + translation)
        return triangles_world
