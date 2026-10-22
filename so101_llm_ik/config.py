"""Arm configuration. Load from JSON so you can keep one file per arm."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class JointCalibration:
    """Maps URDF angles to the angles your robot driver expects:
    driver_deg = sign * urdf_deg + offset_deg.
    With LeRobot's degree mode and so101_new_calib.urdf this should be sign=1,
    offset=0 for every joint, but verify it (see README, "Check calibration")."""
    sign: float = 1.0
    offset_deg: float = 0.0


@dataclass
class Workspace:
    """Box that targets must lie in, in metres, robot base frame (z up)."""
    x: tuple[float, float] = (-0.05, 0.40)
    y: tuple[float, float] = (-0.30, 0.30)
    z: tuple[float, float] = (0.01, 0.40)   # z min keeps the gripper off the table


@dataclass
class ArmConfig:
    urdf_path: str
    tip_link: str = "gripper_frame_link"
    arm_joints: list[str] = field(default_factory=lambda: [
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])
    # Which axis of tip_link points out of the gripper, in that link's own frame.
    # Check this on the real URDF (README, "Find the tool axis").
    tool_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    calibration: dict[str, JointCalibration] = field(default_factory=dict)
    workspace: Workspace = field(default_factory=Workspace)
    home_deg: dict[str, float] = field(default_factory=dict)  # empty = all zeros
    joint_limit_margin_deg: float = 3.0
    max_joint_speed_deg_s: float = 45.0
    control_hz: float = 30.0
    position_tol_mm: float = 2.0
    approach_tol_deg: float = 3.0

    @classmethod
    def from_json(cls, path: str | Path) -> "ArmConfig":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        urdf = Path(raw["urdf_path"])
        if not urdf.is_absolute():
            raw["urdf_path"] = str((path.parent / urdf).resolve())
        raw["calibration"] = {k: JointCalibration(**v) for k, v in raw.get("calibration", {}).items()}
        if "workspace" in raw:
            raw["workspace"] = Workspace(**{k: tuple(v) for k, v in raw["workspace"].items()})
        if "tool_axis_local" in raw:
            raw["tool_axis_local"] = tuple(raw["tool_axis_local"])
        return cls(**raw)

    def to_json(self, path: str | Path) -> None:
        """Write the config. The URDF path is stored relative to the file when it lies below it."""
        path = Path(path)
        raw = asdict(self)
        try:
            raw["urdf_path"] = Path(self.urdf_path).resolve().relative_to(path.parent.resolve()).as_posix()
        except ValueError:
            pass
        path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    def cal(self, joint: str) -> JointCalibration:
        return self.calibration.get(joint, JointCalibration())
