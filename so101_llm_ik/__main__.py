"""Command line.

    python -m so101_llm_ik gui      [--config arm.json] [--port COM3 --id my_follower]
    python -m so101_llm_ik shell    --config arm.json [--port COM3 --id my_follower] [--units cm]
    python -m so101_llm_ik inspect  --config arm.json
    python -m so101_llm_ik solve    --config arm.json 0.2 0.0 0.1 [--approach down]
    python -m so101_llm_ik move     --config arm.json 0.2 0.0 0.1 --port /dev/ttyACM0 --id my_follower

`solve` and `move` take metres. `gui` and `shell` without --port are dry runs.
"""

from __future__ import annotations

import argparse
import json

from .config import ArmConfig
from .controller import APPROACHES, ArmController
from .robot import DryRunRobot, LeRobotSO101
from .shell import UNITS, Shell


def main() -> None:
    ap = argparse.ArgumentParser(prog="so101_llm_ik")
    ap.add_argument("command", choices=["gui", "shell", "inspect", "solve", "move"])
    ap.add_argument("xyz", nargs="*", type=float)
    ap.add_argument("--config", help="arm config JSON (gui default: arm.json, created from the Setup tab)")
    ap.add_argument("--approach", default="any", choices=list(APPROACHES))
    ap.add_argument("--units", default="cm", choices=list(UNITS), help="gui/shell input units")
    ap.add_argument("--port")
    ap.add_argument("--id")
    a = ap.parse_args()

    if a.command == "gui":
        from .gui import run  # tkinter is only needed here
        run(a.config or "arm.json", a.port, a.id, a.units)
        return
    if not a.config:
        ap.error(f"{a.command} needs --config")

    cfg = ArmConfig.from_json(a.config)
    real = a.command == "move" or (a.command == "shell" and a.port)
    if real:
        if not (a.port and a.id):
            ap.error(f"{a.command} on the real arm needs --port and --id")
        robot = LeRobotSO101(a.port, a.id)
    else:
        robot = DryRunRobot(cfg.arm_joints, start=cfg.home_deg)
    robot.connect()
    ctl = ArmController(cfg, robot, realtime=bool(real))

    try:
        if a.command == "shell":
            Shell(ctl, confirm=bool(real), units=a.units).run()
        elif a.command == "inspect":
            print(ctl.inspect())
        elif len(a.xyz) != 3:
            ap.error("give x y z in metres")
        elif a.command == "solve":
            print(json.dumps(ctl.check_reachable(*a.xyz, approach=a.approach), indent=2))
        else:
            print(json.dumps(ctl.move_to(*a.xyz, approach=a.approach), indent=2))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
