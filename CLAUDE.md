# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pure-NumPy IK stack for the 5-joint SO-101 arm: typed Cartesian target → workspace check → IK → joint-limit check → speed-limited move → report from encoders + FK. No Placo/Pinocchio, so it runs on native Windows. The only runtime dependency is `numpy`; `lerobot[feetech]` is needed only for the real arm. There is no `pyproject.toml`/`setup.py` and no lint config. Run things from the repo root as `python -m so101_llm_ik ...`.

## Commands

The project venv is `.venv` (Windows layout: `.venv/Scripts/python.exe`).

```bash
python -m pytest tests -q                                  # all tests
python -m pytest tests/test_ik.py -q                       # IK/controller/shell tests only
python -m pytest tests/test_ik.py::test_round_trip_in_front_of_robot -q   # single test

python -m so101_llm_ik gui     --config arm.json           # Tk dashboard, dry run
python -m so101_llm_ik shell   --config arm.json           # text prompt, dry run
python -m so101_llm_ik inspect --config arm.json           # joint limits + tip axes at zero pose
python -m so101_llm_ik solve   --config arm.json 0.20 0.0 0.08 --approach down   # metres
```

Adding `--port COM3 --id my_follower` to `gui`/`shell` (and always for `move`) drives the **real arm**. Don't run those unless the user asks.

`tests/test_gui.py` drives the real Tk `Dashboard` end to end. It needs a display, skips without one, and shares one module-scoped `Tk()` root because creating many roots is flaky on Windows. Its `pump()` helper spins the event loop until `dash.busy` clears.

## Architecture

The layers, bottom to top (each depends only on the ones below it):

- `chain.py`: minimal URDF parser → `Chain` (serial path from the root to `tip_link`), FK, geometric Jacobian. Works in **URDF radians**. Movable joints on the path that aren't in `arm_joints` are held at 0. The gripper jaw isn't on the path to `gripper_frame_link`, so it is never part of IK.
- `solver.py`: `IKSolver`, damped least squares. Targets are a position plus an optional approach *direction*, never a full 6D pose, because a 5-DOF arm can't satisfy one. It tries the current pose first, then structured seeds (base yawed toward the target × elbow shapes, which assumes joint 0 is base yaw), then seeded-random restarts (`rng_seed=0`, so results are deterministic). Ranking: success, then error, then joint distance from the seed. A conservative reach-sphere check rejects far targets before iterating. Every result is re-checked with FK, and `reason` explains failures in plain words. When an approach-constrained solve fails it re-solves position-only, to tell "unreachable" apart from "wrong approach".
- `controller.py`: `ArmController` is **the** API the shell and GUI call. Every public command returns a dict with `"ok"`; on failure there is an `"error"` string and **nothing has moved**. It owns all safety checks: input units (|coord| > 2 is rejected as probably mm/cm), the workspace box, and joint limits shrunk by `joint_limit_margin_deg`.
- `robot.py`: the `Robot` protocol (`connect/disconnect/read/write`) with `DryRunRobot` (instant, keeps a `log` of writes) and `LeRobotSO101` (imports LeRobot lazily and handles the module rename between releases). `run_trajectory` does linear joint-space interpolation capped at `max_joint_speed_deg_s`. `realtime=False` skips the sleeps, which is what tests use.
- `shell.py` / `gui.py`: front ends. The shell takes injectable `input_fn`/`print_fn` for testing. The GUI runs every controller call on a worker thread via `Dashboard._run`, and results come back through a queue polled with `root.after`. Never read the real arm while a worker is using the serial bus.
- `__main__.py`: argparse CLI. `gui` imports tkinter lazily.

### Two angle spaces: keep them straight

- **URDF degrees/radians**: what `Chain` and `IKSolver` use.
- **Driver degrees**: what `Robot.read/write` and every controller dict exposed to users use (`joints_deg`). The conversion is `driver = sign * urdf + offset_deg` per joint (`JointCalibration`), and only `ArmController._to_driver`/`_from_driver` do it.
- The gripper is a 0–100 open percent and sits in the robot state dict alongside the arm joints.

### Config

`ArmConfig.from_json` resolves a relative `urdf_path` against the **config file's directory**, and `to_json` writes it back relative when possible. `arm.example.json` is the template. `arm.json` is the user's real config (the GUI Setup tab edits and saves it). `Aigis.json` is another per-arm config of the same shape. `tool_axis_local` must point out of the gripper fingers, or `approach="down"` is wrong. The real URDF sets it to `[0,0,1]` and the stand-in to `[1,0,0]`.

### Test fixture caveat

`tests/fixtures/so101_standin.urdf` has the real joint names but **approximate geometry**. It is for tests only and must never be used to drive hardware. Tests build configs directly with `ArmConfig(urdf_path=..., tool_axis_local=(1, 0, 0))` and use `realtime=False`.

## Known limits (by design)

- The approach direction can only lie in the vertical plane through the base and the target.
- Moves are straight lines in joint space, not Cartesian, so the tip can arc between waypoints.
- No collision checking beyond the workspace box.
