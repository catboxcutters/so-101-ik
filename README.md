# so101_llm_ik

You type a Cartesian target; this package turns it into safe SO-101 joint
commands and reports back what actually happened.

```
you  --typed target-->  shell  -->  workspace check  -->  IK solver  -->  limit check  -->  robot
 ^                                                                                         |
 +----------------- tip position + joint angles (from encoders + FK) ----------------------+
```

- **Pure NumPy IK**: no Placo/Pinocchio, so it installs on native Windows too. Damped least
  squares with a convergence loop, structured restarts and a fast reach check. About 8 ms per solve.
- **Targets the SO-101 can actually hit**: a position plus an optional *approach direction*
  (`any`, `down`, `forward`), not a full 6D pose, which a 5-joint arm can't satisfy in general.
- **Honest results**: every solution is re-checked with FK. Failures say *why*
  (out of reach, outside the workspace, impossible approach, near a joint limit).
- **Nothing moves on failure**, and moves are speed-limited joint-space interpolations.

## Layout

| File | What it does |
|---|---|
| `so101_llm_ik/chain.py` | URDF → serial chain, forward kinematics, Jacobian |
| `so101_llm_ik/solver.py` | IK with convergence loop and `IKResult` |
| `so101_llm_ik/controller.py` | `ArmController`: validate → solve → move → report |
| `so101_llm_ik/gui.py` | Dashboard: targets, jogging, joint sliders, live arm view, setup editor |
| `so101_llm_ik/shell.py` | Same commands as a text prompt |
| `so101_llm_ik/robot.py` | `DryRunRobot`, `LeRobotSO101`, trajectory interpolation |
| `so101_llm_ik/config.py` | `ArmConfig` loaded from JSON |
| `tests/` | Tests against a stand-in URDF |

## Setup

```bash
pip install numpy                        # core
pip install "lerobot[feetech]"           # only for the real arm
```

Get the real URDF and its meshes from TheRobotStudio's SO-ARM100 repo
(`Simulation/SO101/so101_new_calib.urdf`) into `urdf/`, then
`cp arm.example.json arm.json`. Only the URDF is parsed; meshes aren't needed.

> `tests/fixtures/so101_standin.urdf` has the same joint names but **approximate**
> geometry. It is for tests only. Never drive hardware with it.

## Dashboard

```bash
python -m so101_llm_ik gui --config arm.json
```

- **Control tab**
  - **Cartesian target:** type x, y and z (in m, cm or mm) and pick an approach. **Check** solves without moving and **Go** moves.
  - **Jog buttons:** step the tip ±X, ±Y or ±Z from where it is now.
  - **Joint sliders:** dragging one previews the pose in the view. **Move joints** sends it, and **Load IK solution** copies the last Check result onto the sliders.
  - **Gripper and Home buttons.**
  - **Arm view:** a top view and a side view of the current pose (solid), the planned pose (dashed), the target, the workspace box and the reach limit.
- **Setup tab**
  - **Config editor:** edits everything in `arm.json`, including URDF, tool axis, per-joint sign, offset and home, workspace, speeds and tolerances.
  - **Inspect URDF:** shows which tip axis to use.
  - **Save / Apply:** **Save** writes the file and **Apply** uses the edits without saving. If `arm.json` doesn't exist yet, the tab starts with defaults and **Save** creates it.

Dry run is the default. For the real arm, pick **Real arm**, enter the port and robot id, and
press **Connect**. With **Confirm real-arm moves** ticked, every move shows each joint's
current → planned angle and waits for **Yes**. What you confirm is exactly what gets sent.
The arm must already be calibrated with LeRobot under that robot id.

## Before the first real move

**1. Find the tool axis.** Run:

```bash
python -m so101_llm_ik inspect --config arm.json
```

It prints where each local axis of `gripper_frame_link` points at the all-zero pose.
Set `tool_axis_local` to the one pointing *out of the gripper fingers*. If it's wrong,
`approach: "down"` points the wrong way.

**2. Check calibration.** Put the arm in a pose with the LeRobot follower, read the joints,
and compare `python -m so101_llm_ik solve` output for a point you measure with a ruler
(or use `joints` in the shell and measure where the tip lands).
With `so101_new_calib.urdf` and LeRobot degree mode, `sign: 1, offset_deg: 0` *should* be
right for every joint. If a joint moves the opposite way, set its sign to -1. If it's offset,
set `offset_deg`.

**3. Set the workspace** in `arm.json` to the space above your table that's free of
obstacles. `z` min is your table clearance.

**4. Dry run first.** The dashboard starts in dry run. The text shell does the same
without `--port`:

```bash
python -m so101_llm_ik shell --config arm.json
```

```
[cm]> check 20 0 8 down        # solve only: joint angles + residual error
[cm]> go 20 0 8 down           # solve and move
[cm]> j elbow_flex=30          # drive joints directly (driver degrees), see where the tip lands
[cm]> state                    # current tip position and joints
[cm]> units m                  # switch input units (m, cm, mm; default cm)
[cm]> help
```

Approach is `any` (default), `down` or `forward`. One-shot versions, in metres:

```bash
python -m so101_llm_ik solve --config arm.json 0.20 0.0 0.08 --approach down
```

**5. Real arm**, with a hand near the power switch. Every move shows the solved joints
and waits for confirmation before anything happens:

```bash
python -m so101_llm_ik gui   --config arm.json --port /dev/ttyACM0 --id my_follower
python -m so101_llm_ik shell --config arm.json --port /dev/ttyACM0 --id my_follower
```

On Windows the port is `COM3` or similar.

## Using it from Python

```python
from so101_llm_ik import ArmConfig, ArmController, DryRunRobot

cfg = ArmConfig.from_json("arm.json")
ctl = ArmController(cfg, DryRunRobot(cfg.arm_joints))
ctl.move_to(0.2, 0.0, 0.1, approach="down")
# {'ok': True, 'moved': True, 'position_m': [0.2, 0.0, 0.1], 'error_from_target_mm': 0.1, ...}
```

## Known limits

- The approach direction can only lie in the vertical plane through the base and the target.
  That's a property of the arm, and the solver tells you when you hit it.
- Moves are straight lines in joint space, not in Cartesian space. The tip can arc between
  waypoints, so plan pick-and-place via a point above the object.
- No collision checking beyond the workspace box.
- Gripper is commanded 0–100 directly and isn't part of IK.

## Tests

```bash
pip install pytest && python -m pytest tests -q
```
