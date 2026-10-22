"""Interactive prompt for trying targets by hand.

    python -m so101_llm_ik shell --config arm.json                                  # dry run
    python -m so101_llm_ik shell --config arm.json --port COM3 --id my_follower     # real arm

On the real arm every move shows the solved joint angles first and waits for 'y'.
"""

from __future__ import annotations

import shlex
from typing import Any, Callable

from .controller import APPROACHES, ArmController

UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001}

HELP = """\
Commands (coordinates in the current units, base frame: x forward, y left, z up):
  check x y z [approach]      solve IK only, nothing moves         (alias: c)
  go x y z [approach]         solve IK and move there              (alias: g)
  joints v1 v2 v3 v4 v5       move to joint angles in driver deg   (alias: j)
  joints name=deg ...         move only the named joints, e.g. j elbow_flex=30
  gripper percent             go to percent open: 0 = closed, 100 = open
  gripper +step | -step       open / close by step percent from where it is now
  home                        move to the home pose
  state                       current tip position and joint angles (alias: s)
  units m|cm|mm               unit for typed coordinates (now: {units})
  workspace                   allowed target box and reach
  diag                        servo voltage, temperature, current, load and faults (real arm)
  help | quit
approach is one of: {approaches} (default any)"""


class Shell:
    def __init__(self, ctl: ArmController, confirm: bool, units: str = "cm",
                 input_fn: Callable[[str], str] = input, print_fn: Callable[..., None] = print):
        self.ctl = ctl
        self.confirm = confirm
        self.units = units
        self.input = input_fn
        self.print = print_fn
        self.commands: dict[str, Callable[[list[str]], None]] = {
            "check": self.cmd_check, "c": self.cmd_check,
            "go": self.cmd_go, "g": self.cmd_go,
            "joints": self.cmd_joints, "j": self.cmd_joints,
            "gripper": self.cmd_gripper,
            "home": self.cmd_home,
            "state": self.cmd_state, "s": self.cmd_state,
            "units": self.cmd_units,
            "diag": self.cmd_diag,
            "workspace": lambda _: self.print(self.ctl.workspace_description()),
            "help": lambda _: self.print(self._help()), "?": lambda _: self.print(self._help()),
        }

    # ---- loop ----------------------------------------------------------------

    def run(self) -> None:
        mode = "REAL ARM, moves ask for confirmation" if self.confirm else "dry run, nothing physical moves"
        self.print(f"SO-101 IK shell ({mode}). Type 'help' for commands.\n")
        self.cmd_state([])
        while True:
            try:
                line = self.input(f"\n[{self.units}]> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.print()
                return
            if not self.handle(line):
                return

    def handle(self, line: str) -> bool:
        """Run one command line. Returns False when the user wants to quit."""
        try:
            words = shlex.split(line)
        except ValueError as e:
            self.print(f"error: {e}")
            return True
        if not words:
            return True
        cmd, args = words[0].lower(), words[1:]
        if cmd in ("quit", "exit", "q"):
            return False
        fn = self.commands.get(cmd)
        if fn is None:
            self.print(f"Unknown command '{cmd}'. Type 'help'.")
            return True
        try:
            fn(args)
        except Exception as e:  # keep the session alive whatever goes wrong
            self.print(f"error: {type(e).__name__}: {e}")
        return True

    # ---- commands ------------------------------------------------------------

    def cmd_check(self, args: list[str]) -> None:
        xyz, approach = self._parse_target(args)
        if xyz is None:
            return
        self._show(self.ctl.check_reachable(*xyz, approach=approach))

    def cmd_go(self, args: list[str]) -> None:
        xyz, approach = self._parse_target(args)
        if xyz is None:
            return
        if self.confirm:
            plan = self.ctl.check_reachable(*xyz, approach=approach)
            self._show(plan)
            if not plan.get("reachable") or not self._ask("Move?"):
                return
        self._show(self.ctl.move_to(*xyz, approach=approach))

    def cmd_joints(self, args: list[str]) -> None:
        names = self.ctl.cfg.arm_joints
        if not args:
            self.print(f"usage: joints v1 ... v{len(names)}   or   joints name=deg ...  ({', '.join(names)})")
            return
        if all("=" in a for a in args):
            goal = {k: float(v) for k, v in (a.split("=", 1) for a in args)}
        elif len(args) == len(names):
            goal = dict(zip(names, map(float, args)))
        else:
            self.print(f"Give all {len(names)} joint angles in order ({', '.join(names)}), or name=deg pairs.")
            return
        if self.confirm:
            current = self.ctl.robot.read()
            for n, v in goal.items():
                self.print(f"  {n:15s} {current.get(n, float('nan')):8.2f} -> {v:8.2f} deg")
            if not self._ask("Move?"):
                return
        self._show(self.ctl.move_joints(goal))

    def cmd_gripper(self, args: list[str]) -> None:
        if len(args) != 1:
            self.print("usage: gripper percent | +step | -step   (0 = closed, 100 = open)")
            return
        v = float(args[0])
        if args[0][0] in "+-":  # relative step
            v += self.ctl.robot.read()["gripper"]
        self._show(self.ctl.set_gripper(min(100.0, max(0.0, v))))

    def cmd_diag(self, args: list[str]) -> None:
        diagnostics = getattr(self.ctl.robot, "diagnostics", None)
        if diagnostics is None:
            self.print("diag reads the servos, so it only works on the real arm.")
            return
        self.print(f"  {'motor':14s} {'volt':>6s} {'temp':>5s} {'current':>8s} {'load':>6s} "
                   f"{'torque':>6s} {'goal-now':>9s}  faults")
        notes, volts = [], []
        for name, d in diagnostics().items():
            if "error" in d:
                self.print(f"  {name:14s} NOT RESPONDING: {d['error']}")
                notes.append(f"{name} did not answer: check its cable and the power supply.")
                continue
            volts.append(d["voltage_v"])
            gap = d["goal_minus_present_deg"]
            self.print(f"  {name:14s} {d['voltage_v']:5.1f}V {d['temperature_c']:4d}C {d['current_ma']:6.0f}mA "
                       f"{d['load_percent']:5.1f}% {'on' if d['torque_on'] else 'off':>6s} "
                       + (f"{gap:+8.1f}°" if gap is not None else f"{'-':>9s}") + "  "
                       + (", ".join(d["faults"]) or "-"))
            if d["torque_on"] and gap is not None and abs(gap) > 3 and abs(d["load_percent"]) > 20:
                notes.append(f"{name} is stalled: pushing {abs(gap):.0f}° toward a target it can't reach. "
                             "Command a position it can get to.")
            if d["temperature_c"] >= 50:
                notes.append(f"{name} is at {d['temperature_c']} C. Let it cool before using it hard.")
        if len(volts) > 1 and max(volts) - min(volts) >= 0.5:
            notes.append(f"Voltage differs by {max(volts) - min(volts):.1f} V along the chain: something near the "
                         "low end is drawing a lot of current, or a connector there is poor.")
        for n in notes:
            self.print("  ! " + n)

    def cmd_home(self, args: list[str]) -> None:
        if self.confirm and not self._ask("Move to home pose?"):
            return
        self._show(self.ctl.go_home())

    def cmd_state(self, args: list[str]) -> None:
        self._show(self.ctl.get_arm_state())

    def cmd_units(self, args: list[str]) -> None:
        if len(args) != 1 or args[0] not in UNITS:
            self.print(f"usage: units {'|'.join(UNITS)}   (now: {self.units})")
            return
        self.units = args[0]

    # ---- helpers -------------------------------------------------------------

    def _help(self) -> str:
        return HELP.format(units=self.units, approaches=", ".join(APPROACHES))

    def _parse_target(self, args: list[str]) -> tuple[list[float] | None, str]:
        if len(args) not in (3, 4):
            self.print(f"usage: x y z [{'|'.join(APPROACHES)}]   (units: {self.units})")
            return None, "any"
        scale = UNITS[self.units]
        return [float(v) * scale for v in args[:3]], (args[3].lower() if len(args) == 4 else "any")

    def _ask(self, question: str) -> bool:
        try:
            return self.input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def _fmt_pos(self, p_m: list[float]) -> str:
        s = UNITS[self.units]
        return "(" + ", ".join(f"{v / s:.4g}" for v in p_m) + f") {self.units}"

    def _show(self, out: dict[str, Any]) -> None:
        if not out.get("ok") or out.get("reachable") is False:
            self.print("FAILED: " + str(out.get("error") or out.get("reason") or "unknown error"))
            if "closest_reachable_m" in out:
                self.print(f"  closest reachable: {self._fmt_pos(out['closest_reachable_m'])}")
            if "position_m" in out and "reachable" in out:
                self.print(f"  best attempt reaches {self._fmt_pos(out['position_m'])}, "
                           f"{out['position_error_mm']:.1f} mm off")
            return
        if "moved" in out:
            self.print("OK, moved." if out["moved"] else "OK.")
        elif "reachable" in out:
            self.print(f"Reachable ({out['iterations']} solver iterations).")
        if "position_m" in out:
            self.print(f"  tip position:   {self._fmt_pos(out['position_m'])}")
        for key, label in (("error_from_target_mm", "target error"), ("position_error_mm", "position error")):
            if key in out:
                self.print(f"  {label + ':':16s}{out[key]:.2f} mm")
        if out.get("approach_error_deg") is not None:
            self.print(f"  approach error: {out['approach_error_deg']:.2f} deg")
        if "pointing_direction" in out:
            self.print("  gripper points: (" + ", ".join(f"{float(v):.3f}" for v in out["pointing_direction"]) + ")")
        if "joints_deg" in out:
            self.print("  joints (driver deg):")
            for n, v in out["joints_deg"].items():
                self.print(f"    {n:15s} {v:8.2f}")
        if "gripper_open_percent" in out:
            self.print(f"  gripper:        {out['gripper_open_percent']} % open")
