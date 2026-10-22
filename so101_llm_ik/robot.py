"""Robot backends. Both speak *driver degrees* (what LeRobot uses) plus a gripper value
0 (closed) .. 100 (open)."""

from __future__ import annotations

import time
from typing import Protocol

READ_RETRIES = 3

# Feetech STS3215 control table: name -> (address, bytes, sign bit or None)
_STS_REGS = {
    "torque": (40, 1, None), "goal": (42, 2, 15), "present": (56, 2, 15), "load": (60, 2, 10),
    "voltage": (62, 1, None), "temperature": (63, 1, None), "status": (65, 1, None), "current": (69, 2, None),
}
_STS_FAULTS = ((1, "voltage"), (2, "angle sensor"), (4, "overheat"), (8, "overcurrent"), (32, "overload"))


class Robot(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def read(self) -> dict[str, float]: ...          # {"shoulder_pan": deg, ..., "gripper": 0..100}
    def write(self, targets: dict[str, float]) -> None: ...


class DryRunRobot:
    """No hardware. Moves instantly to whatever it is told. Use it to try targets
    and check what the arm *would* do before connecting the real one."""

    def __init__(self, joint_names: list[str], start: dict[str, float] | None = None):
        self.state = {n: 0.0 for n in joint_names}
        self.state["gripper"] = 50.0
        if start:
            self.state.update(start)
        self.log: list[dict[str, float]] = []

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    def read(self) -> dict[str, float]:
        return dict(self.state)

    def write(self, targets: dict[str, float]) -> None:
        self.state.update(targets)
        self.log.append(dict(targets))


class LeRobotSO101:
    """Real SO-101 follower through LeRobot (`pip install "lerobot[feetech]"`).
    Expects the arm to be calibrated already with the same robot id."""

    def __init__(self, port: str, robot_id: str):
        try:  # module was renamed between LeRobot releases
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        self._robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))

    def connect(self) -> None:
        self._robot.connect()

    def disconnect(self) -> None:
        self._robot.disconnect()

    def read(self) -> dict[str, float]:
        # Straight from the bus rather than get_observation(), which reads only once: a
        # stalled gripper drawing current can garble a single status packet.
        pos = self._robot.bus.sync_read("Present_Position", num_retry=READ_RETRIES)
        return {k: float(v) for k, v in pos.items()}

    def write(self, targets: dict[str, float]) -> None:
        self._robot.send_action({f"{k}.pos": float(v) for k, v in targets.items()})

    def diagnostics(self) -> dict[str, dict]:
        """Health of each servo, read one at a time so a silent one shows up by name.
        Goes through the Feetech SDK directly: LeRobot's reads raise as soon as a servo
        sets a fault flag, which is exactly when these numbers matter."""
        bus = self._robot.bus
        out = {}
        for name, motor in bus.motors.items():
            raw, err_bits = {}, 0
            try:
                for reg, (addr, length, sign_bit) in _STS_REGS.items():
                    read_fn = bus.packet_handler.read1ByteTxRx if length == 1 else bus.packet_handler.read2ByteTxRx
                    for _ in range(1 + READ_RETRIES):
                        value, comm, err = read_fn(bus.port_handler, motor.id, addr)
                        if comm == 0:  # COMM_SUCCESS
                            break
                    else:
                        raise ConnectionError(f"no reply reading {reg}: {bus.packet_handler.getTxRxResult(comm)}")
                    if sign_bit is not None and value & (1 << sign_bit):  # sign-magnitude
                        value = -(value & ((1 << sign_bit) - 1))
                    raw[reg], err_bits = value, err_bits | err
            except Exception as e:
                out[name] = {"error": f"{type(e).__name__}: {e}"}
                continue
            bits = raw["status"] | err_bits
            out[name] = {
                "voltage_v": raw["voltage"] / 10,
                "temperature_c": raw["temperature"],
                "current_ma": raw["current"] * 6.5,
                "load_percent": raw["load"] / 10,
                "torque_on": bool(raw["torque"]),
                # Goal_Position reads 0 until something writes it (e.g. right after power-on)
                "goal_minus_present_deg": (raw["goal"] - raw["present"]) * 360 / 4096 if raw["goal"] else None,
                "faults": [f for bit, f in _STS_FAULTS if bits & bit],
            }
        return out


def run_trajectory(robot: Robot, start: dict[str, float], goal: dict[str, float],
                   max_speed_deg_s: float, hz: float, realtime: bool = True) -> int:
    """Linear joint-space interpolation at a capped speed. Returns the number of steps."""
    delta = max(abs(goal[k] - start.get(k, goal[k])) for k in goal) if goal else 0.0
    duration = delta / max_speed_deg_s
    steps = max(1, int(round(duration * hz)))
    for i in range(1, steps + 1):
        a = i / steps
        robot.write({k: start.get(k, goal[k]) + a * (goal[k] - start.get(k, goal[k])) for k in goal})
        if realtime:
            time.sleep(1.0 / hz)
    return steps
