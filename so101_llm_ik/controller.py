"""Validates targets, solves IK, moves the arm and reports back what actually happened.

Every command returns a dict with "ok": bool. On failure there is always an "error"
string saying why, and the arm has not moved.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .chain import Chain
from .config import ArmConfig
from .robot import Robot, run_trajectory
from .solver import IKSolver

APPROACHES = ("any", "down", "forward")


class ArmController:
    def __init__(self, config: ArmConfig, robot: Robot, realtime: bool = True):
        self.cfg = config
        self.robot = robot
        self.realtime = realtime
        self.chain = Chain.from_urdf(config.urdf_path, config.tip_link, config.arm_joints)
        self.solver = IKSolver(
            self.chain,
            tool_axis_local=config.tool_axis_local,
            position_tol_mm=config.position_tol_mm,
            approach_tol_deg=config.approach_tol_deg,
        )
        m = math.radians(config.joint_limit_margin_deg)
        self._lo, self._hi = self.chain.lower + m, self.chain.upper - m

    # ---- describe ------------------------------------------------------------

    def workspace_description(self) -> str:
        w = self.cfg.workspace
        return (
            "Robot base frame: x forward, y left, z up, units metres. "
            f"Allowed targets: x in [{w.x[0]}, {w.x[1]}], y in [{w.y[0]}, {w.y[1]}], "
            f"z in [{w.z[0]}, {w.z[1]}]. Maximum reach from the shoulder is about "
            f"{self.solver.max_reach_m:.2f} m, and points near the edge of reach may only work "
            "with approach='any'."
        )

    def inspect(self) -> str:
        """Joint limits and where the tip's local axes point at the all-zero pose,
        for picking `tool_axis_local`."""
        ch = self.chain
        lines = ["Active joints and limits (deg):"]
        lines += [f"  {n:15s} [{lo:7.1f}, {hi:7.1f}]"
                  for n, lo, hi in zip(ch.joint_names, np.degrees(ch.lower), np.degrees(ch.upper))]
        T = ch.fk(np.zeros(len(ch.joint_names))).T
        lines.append(f"\nTip '{self.cfg.tip_link}' at all-zero joints: position {np.round(T[:3, 3], 4)} m")
        lines += [f"  tip local {ax}-axis points along {np.round(T[:3, i], 3)} in the base frame"
                  for i, ax in enumerate("xyz")]
        lines.append(f"\nConfigured tool axis: {self.cfg.tool_axis_local}")
        lines.append(f"Max reach from shoulder: {self.solver.max_reach_m:.3f} m")
        lines.append("\n" + self.workspace_description())
        return "\n".join(lines)

    def driver_limits_deg(self) -> dict[str, tuple[float, float]]:
        """Allowed range of each joint in driver degrees, joint limit margin included."""
        out = {}
        for n, lo, hi in zip(self.chain.joint_names, np.degrees(self._lo), np.degrees(self._hi)):
            lo = lo if math.isfinite(lo) else -180.0
            hi = hi if math.isfinite(hi) else 180.0
            c = self.cfg.cal(n)
            a, b = c.sign * lo + c.offset_deg, c.sign * hi + c.offset_deg
            out[n] = (float(min(a, b)), float(max(a, b)))
        return out

    def pose_geometry(self, joints_deg: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Base, every joint and the tip in metres (base frame), plus the direction the
        gripper points. For drawing. `joints_deg` are driver degrees; omit for the current pose."""
        q = self._read_urdf_deg() if joints_deg is None else self._from_driver(joints_deg)
        fk = self.chain.fk(np.radians(q))
        pts = np.array([np.zeros(3), *fk.joint_origins, fk.T[:3, 3]])
        return pts, fk.T[:3, :3] @ self.solver.tool_axis_local

    # ---- commands ------------------------------------------------------------

    def get_arm_state(self) -> dict[str, Any]:
        q_urdf = self._read_urdf_deg()
        p, a = self.solver.tip_pose(q_urdf)
        state = self.robot.read()
        return {
            "ok": True,
            "position_m": [round(v, 4) for v in p],
            "pointing_direction": [round(v, 3) for v in a],
            "joints_deg": {n: round(state[n], 2) for n in self.cfg.arm_joints},
            "gripper_open_percent": round(state.get("gripper", float("nan")), 1),
        }

    def check_reachable(self, x: float, y: float, z: float, approach: str = "any") -> dict[str, Any]:
        problem = self._validate_target(x, y, z, approach)
        if problem:
            return {"ok": False, "reachable": False, "error": problem}
        res = self._solve(x, y, z, approach)
        out = {"ok": True, "reachable": res.reached, **res.to_dict()}
        out.pop("reached")
        if res.reached:
            limit = self._limit_problem(res.joints_deg)
            if limit:
                out.update(reachable=False, reason=limit)
        out["joints_deg"] = {k: round(v, 2) for k, v in self._to_driver(res.joints_deg).items()}
        return out

    def move_to(self, x: float, y: float, z: float, approach: str = "any") -> dict[str, Any]:
        problem = self._validate_target(x, y, z, approach)
        if problem:
            return {"ok": False, "moved": False, "error": problem}
        res = self._solve(x, y, z, approach)
        if not res.reached:
            return {"ok": False, "moved": False, "error": res.reason,
                    "closest_reachable_m": [round(v, 4) for v in res.position_m]}
        limit = self._limit_problem(res.joints_deg)
        if limit:
            return {"ok": False, "moved": False, "error": limit}

        start = self.robot.read()
        goal = self._to_driver(res.joints_deg)
        run_trajectory(self.robot, start, goal, self.cfg.max_joint_speed_deg_s,
                       self.cfg.control_hz, realtime=self.realtime)

        # Report where the arm really is, from the encoders, not from the plan.
        p_actual, a_actual = self.solver.tip_pose(self._read_urdf_deg())
        err_mm = float(np.linalg.norm(p_actual - np.array([x, y, z]))) * 1000.0
        return {
            "ok": True,
            "moved": True,
            "position_m": [round(v, 4) for v in p_actual],
            "error_from_target_mm": round(err_mm, 1),
            "pointing_direction": [round(v, 3) for v in a_actual],
        }

    def move_joints(self, joints_deg: dict[str, float]) -> dict[str, Any]:
        """Move straight to driver-degree joint angles, skipping IK. Joints left out keep
        their current value. Reports the resulting tip pose from FK."""
        unknown = [n for n in joints_deg if n not in self.cfg.arm_joints]
        if unknown:
            return {"ok": False, "moved": False,
                    "error": f"Unknown joint(s) {unknown}. Arm joints are {self.cfg.arm_joints}."}
        start = self.robot.read()
        goal = {n: float(joints_deg.get(n, start[n])) for n in self.cfg.arm_joints}
        limit = self._limit_problem(dict(zip(self.chain.joint_names, self._from_driver(goal))))
        if limit:
            return {"ok": False, "moved": False, "error": limit}
        run_trajectory(self.robot, start, goal, self.cfg.max_joint_speed_deg_s,
                       self.cfg.control_hz, realtime=self.realtime)
        return {"ok": True, "moved": True, **{k: v for k, v in self.get_arm_state().items() if k != "ok"}}

    def set_gripper(self, open_percent: float) -> dict[str, Any]:
        if not 0 <= open_percent <= 100:
            return {"ok": False, "error": "open_percent must be between 0 and 100."}
        start = self.robot.read()
        run_trajectory(self.robot, start, {"gripper": float(open_percent)},
                       max_speed_deg_s=100.0, hz=self.cfg.control_hz, realtime=self.realtime)
        return {"ok": True, "gripper_open_percent": round(self.robot.read().get("gripper", open_percent), 1)}

    def go_home(self) -> dict[str, Any]:
        goal = {n: float(self.cfg.home_deg.get(n, 0.0)) for n in self.cfg.arm_joints}
        run_trajectory(self.robot, self.robot.read(), goal, self.cfg.max_joint_speed_deg_s,
                       self.cfg.control_hz, realtime=self.realtime)
        return {"ok": True, **{k: v for k, v in self.get_arm_state().items() if k != "ok"}}

    # ---- helpers -------------------------------------------------------------

    def _validate_target(self, x, y, z, approach) -> str | None:
        for v in (x, y, z):
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                return "x, y and z must be finite numbers in metres."
        if max(abs(x), abs(y), abs(z)) > 2.0:
            return ("Coordinates look like millimetres or centimetres. Units are metres "
                    f"(e.g. 0.2 for 20 cm). Got ({x}, {y}, {z}).")
        if approach not in APPROACHES:
            return f"approach must be one of {list(APPROACHES)}."
        w = self.cfg.workspace
        out = [f"{ax}={v} not in [{lo}, {hi}]"
               for ax, v, (lo, hi) in (("x", x, w.x), ("y", y, w.y), ("z", z, w.z))
               if not lo <= v <= hi]
        if out:
            return "Target is outside the safe workspace: " + "; ".join(out) + "."
        return None

    def _solve(self, x, y, z, approach):
        target = np.array([x, y, z], dtype=float)
        a = None
        if approach == "down":
            a = np.array([0.0, 0.0, -1.0])
        elif approach == "forward":
            r = np.array([x, y, 0.0])
            a = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-6 else np.array([1.0, 0.0, 0.0])
        return self.solver.solve(target, self._read_urdf_deg(), approach_dir=a)

    def _limit_problem(self, joints_urdf_deg: dict[str, float]) -> str | None:
        q = np.radians([joints_urdf_deg[n] for n in self.chain.joint_names])
        bad = [n for n, v, lo, hi in zip(self.chain.joint_names, q, self._lo, self._hi) if not lo <= v <= hi]
        if bad:
            return (f"The solution puts {', '.join(bad)} within {self.cfg.joint_limit_margin_deg} deg "
                    "of its limit, which is not allowed. Try a point closer to the middle of the workspace.")
        return None

    def _read_urdf_deg(self) -> np.ndarray:
        return self._from_driver(self.robot.read())

    def _from_driver(self, joints_deg: dict[str, float]) -> np.ndarray:
        return np.array([(joints_deg[n] - self.cfg.cal(n).offset_deg) / self.cfg.cal(n).sign
                         for n in self.chain.joint_names])

    def _to_driver(self, joints_urdf_deg: dict[str, float]) -> dict[str, float]:
        return {n: self.cfg.cal(n).sign * v + self.cfg.cal(n).offset_deg for n, v in joints_urdf_deg.items()}
