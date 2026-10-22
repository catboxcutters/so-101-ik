"""Minimal URDF -> serial kinematic chain, with forward kinematics and Jacobian.

Pure NumPy, no Placo/Pinocchio, so it installs anywhere (including native Windows).
Only what a serial arm needs: revolute/continuous/prismatic/fixed joints,
<origin xyz rpy>, <axis>, <limit>. Meshes, inertias and collisions are ignored.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np


def _rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _axis_angle_R(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _homog(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@dataclass
class Joint:
    name: str
    type: str                 # revolute | continuous | prismatic | fixed
    parent: str
    child: str
    origin: np.ndarray        # 4x4 parent -> joint frame at q = 0
    axis: np.ndarray          # unit axis in the joint frame
    lower: float = -math.inf  # radians (or metres for prismatic)
    upper: float = math.inf

    @property
    def movable(self) -> bool:
        return self.type != "fixed"


@dataclass
class FKResult:
    T: np.ndarray                    # 4x4 base -> tip
    joint_origins: list[np.ndarray] = field(default_factory=list)  # world position of each active joint
    joint_axes: list[np.ndarray] = field(default_factory=list)     # world axis of each active joint


class Chain:
    """Serial chain from the URDF root link to `tip_link`.

    `active_joints` restricts which movable joints the solver may move. Any movable
    joint on the path that is not active is held at 0 (useful to freeze wrist_roll).
    """

    def __init__(self, joints_on_path: list[Joint], active_joints: list[str]):
        self.path = joints_on_path
        names_on_path = [j.name for j in joints_on_path if j.movable]
        missing = [n for n in active_joints if n not in names_on_path]
        if missing:
            raise ValueError(
                f"Joints {missing} are not on the path to the tip link. "
                f"Movable joints on the path: {names_on_path}"
            )
        self.active = [j for j in joints_on_path if j.name in active_joints]
        self.active.sort(key=lambda j: active_joints.index(j.name))
        self._index = {j.name: i for i, j in enumerate(self.active)}

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_urdf(cls, urdf_path: str, tip_link: str, active_joints: list[str]) -> "Chain":
        root = ET.parse(urdf_path).getroot()
        by_child: dict[str, Joint] = {}
        for el in root.findall("joint"):
            o = el.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            a = el.find("axis")
            axis = np.array([float(v) for v in (a.get("xyz", "1 0 0") if a is not None else "1 0 0").split()])
            n = np.linalg.norm(axis)
            axis = axis / n if n > 0 else np.array([1.0, 0.0, 0.0])
            lim = el.find("limit")
            jtype = el.get("type")
            lower, upper = -math.inf, math.inf
            if lim is not None and jtype in ("revolute", "prismatic"):
                lower = float(lim.get("lower", -math.inf))
                upper = float(lim.get("upper", math.inf))
            j = Joint(
                name=el.get("name"),
                type=jtype,
                parent=el.find("parent").get("link"),
                child=el.find("child").get("link"),
                origin=_homog(_rpy_to_R(*rpy), xyz),
                axis=axis,
                lower=lower,
                upper=upper,
            )
            by_child[j.child] = j

        links = {el.get("name") for el in root.findall("link")}
        if tip_link not in links:
            raise ValueError(f"Link '{tip_link}' not in URDF. Links: {sorted(links)}")

        path: list[Joint] = []
        link = tip_link
        while link in by_child:
            j = by_child[link]
            path.append(j)
            link = j.parent
        path.reverse()
        return cls(path, active_joints)

    # ---- properties ---------------------------------------------------------

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.active]

    @property
    def lower(self) -> np.ndarray:
        return np.array([j.lower for j in self.active])

    @property
    def upper(self) -> np.ndarray:
        return np.array([j.upper for j in self.active])

    def clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower, self.upper)

    # ---- kinematics ---------------------------------------------------------

    def fk(self, q: np.ndarray) -> FKResult:
        """Forward kinematics. q in radians, ordered as `joint_names`."""
        n = len(self.active)
        origins: list = [None] * n
        axes: list = [None] * n
        T = np.eye(4)
        for j in self.path:
            T = T @ j.origin
            if j.movable:
                i = self._index.get(j.name)
                qi = q[i] if i is not None else 0.0
                if i is not None:
                    origins[i] = T[:3, 3].copy()
                    axes[i] = T[:3, :3] @ j.axis
                if j.type == "prismatic":
                    T = T @ _homog(np.eye(3), j.axis * qi)
                else:
                    T = T @ _homog(_axis_angle_R(j.axis, qi), np.zeros(3))
        return FKResult(T=T, joint_origins=origins, joint_axes=axes)

    def jacobian(self, fk: FKResult) -> tuple[np.ndarray, np.ndarray]:
        """Geometric Jacobian (3xN linear, 3xN angular) in the base frame."""
        p = fk.T[:3, 3]
        n = len(self.active)
        Jv, Jw = np.zeros((3, n)), np.zeros((3, n))
        for i, j in enumerate(self.active):
            z, o = fk.joint_axes[i], fk.joint_origins[i]
            if j.type == "prismatic":
                Jv[:, i] = z
            else:
                Jv[:, i] = np.cross(z, p - o)
                Jw[:, i] = z
        return Jv, Jw
