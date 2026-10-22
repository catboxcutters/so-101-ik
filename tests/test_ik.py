"""Run with:  python -m pytest tests -q
Uses the stand-in URDF in tests/fixtures (approximate geometry, same joint names)."""

from pathlib import Path

import numpy as np
import pytest

from so101_llm_ik import ArmConfig, ArmController, Chain, DryRunRobot, IKSolver, Shell

URDF = str(Path(__file__).parent / "fixtures" / "so101_standin.urdf")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@pytest.fixture
def chain():
    return Chain.from_urdf(URDF, "gripper_frame_link", JOINTS)


@pytest.fixture
def solver(chain):
    return IKSolver(chain, tool_axis_local=(1, 0, 0))


@pytest.fixture
def ctl():
    cfg = ArmConfig(urdf_path=URDF, tool_axis_local=(1, 0, 0))
    return ArmController(cfg, DryRunRobot(JOINTS), realtime=False)


def test_jacobian_matches_finite_differences(chain):
    q = np.radians([10, 20, -30, 15, 5])
    fk = chain.fk(q)
    Jv, _ = chain.jacobian(fk)
    num = np.array([(chain.fk(q + 1e-6 * np.eye(5)[i]).T[:3, 3] - fk.T[:3, 3]) / 1e-6 for i in range(5)]).T
    assert np.abs(num - Jv).max() < 1e-5


def test_gripper_jaw_is_not_in_chain(chain):
    assert "gripper" not in chain.joint_names


def test_round_trip_in_front_of_robot(solver, chain):
    """FK a random pose, IK back to it. Only poses in front of and above the base,
    which is where the workspace check allows targets anyway."""
    rng = np.random.default_rng(0)
    tried = 0
    while tried < 150:
        q = np.degrees(rng.uniform(chain.lower, chain.upper))
        p, a = solver.tip_pose(q)
        if p[0] < 0.05 or p[2] < 0.01:
            continue
        tried += 1
        r = solver.solve(p, np.zeros(5), approach_dir=a if tried % 2 else None)
        assert r.reached, (q, p, r.reason)
        assert r.position_error_mm <= 2.0


def test_unreachable_is_fast_and_explained(solver):
    r = solver.solve(np.array([0.8, 0, 0.1]), np.zeros(5))
    assert not r.reached and r.iterations == 0
    assert "out of reach" in r.reason


def test_impossible_approach_is_diagnosed(solver):
    # pointing straight along +x at a target off to the side is outside the arm's plane
    r = solver.solve(np.array([0.15, -0.1, 0.15]), np.zeros(5), approach_dir=np.array([1, 0, 0]))
    assert not r.reached
    assert "approach" in r.reason


def test_move_to_moves_and_reports_actual_position(ctl):
    out = ctl.move_to(0.2, 0.05, 0.1)
    assert out["ok"] and out["moved"]
    assert out["error_from_target_mm"] <= 2.0
    state = ctl.get_arm_state()
    assert np.allclose(state["position_m"], [0.2, 0.05, 0.1], atol=0.003)


def test_top_down_grasp(ctl):
    out = ctl.move_to(0.18, -0.05, 0.05, approach="down")
    assert out["ok"], out
    assert out["pointing_direction"][2] < -0.99


def test_rejections_do_not_move(ctl):
    before = ctl.robot.read()
    for args, needle in [
        ({"x": 200, "y": 0, "z": 100}, "millimetres"),
        ({"x": 0.2, "y": 0, "z": -0.05}, "outside the safe workspace"),
        ({"x": 0.39, "y": 0.29, "z": 0.39}, "reach"),
        ({"x": 0.2, "y": 0, "z": 0.1, "approach": "sideways"}, "approach must be"),
    ]:
        out = ctl.move_to(**args)
        assert not out["ok"] and not out["moved"], out
        assert needle in out["error"], out["error"]
    assert ctl.robot.read() == before


def test_bad_commands_return_errors(ctl):
    before = ctl.robot.read()
    assert not ctl.set_gripper(150)["ok"]
    assert not ctl.move_joints({"elbow": 10})["ok"]
    assert not ctl.move_joints({"elbow_flex": 1000})["ok"]
    assert ctl.robot.read() == before


def test_check_reachable_does_not_move(ctl):
    before = ctl.robot.read()
    out = ctl.check_reachable(0.2, 0.0, 0.12)
    assert out["ok"] and out["reachable"]
    assert ctl.robot.read() == before


def test_move_joints_matches_check_reachable(ctl):
    plan = ctl.check_reachable(0.2, 0.0, 0.12)
    out = ctl.move_joints(plan["joints_deg"])
    assert out["ok"] and np.allclose(out["position_m"], [0.2, 0.0, 0.12], atol=0.003)


def test_shell_converts_units_and_moves(ctl):
    printed = []
    sh = Shell(ctl, confirm=False, units="cm", print_fn=lambda *a: printed.append(" ".join(map(str, a))))
    assert sh.handle("go 20 5 10")
    assert np.allclose(ctl.get_arm_state()["position_m"], [0.2, 0.05, 0.1], atol=0.003)
    assert sh.handle("units mm") and sh.handle("check 2000 0 100")
    assert any("FAILED" in line for line in printed)
    assert sh.handle("bogus") and sh.handle("go 1 2")  # bad input never raises
    assert not sh.handle("quit")


def test_shell_gripper_absolute_and_relative(ctl):
    sh = Shell(ctl, confirm=False, print_fn=lambda *a: None)
    sh.handle("gripper 40")
    assert ctl.robot.read()["gripper"] == 40
    sh.handle("gripper +5")
    assert ctl.robot.read()["gripper"] == 45
    sh.handle("gripper -60")  # clamped at fully closed
    assert ctl.robot.read()["gripper"] == 0


def test_shell_diag(ctl):
    from types import SimpleNamespace
    from so101_llm_ik import LeRobotSO101

    # gripper: torque on, goal 100 ticks past where it is, high load, overload flag, one bad read first
    regs = {6: {40: 1, 42: 2000, 56: 2100, 60: 600, 62: 120, 63: 55, 65: 32, 69: 200},
            1: {40: 1, 42: 0, 56: 2047, 60: 30 | 1 << 10, 62: 121, 63: 30, 65: 0, 69: 10}}  # goal never written
    flaky = {"left": 1}

    def read(_port, id_, addr):
        if id_ == 6 and flaky["left"]:
            flaky["left"] -= 1
            return 0, -6, 0  # garbled reply, retried
        if id_ == 2:
            return 0, -6, 0  # never answers
        return regs[id_][addr], 0, regs[id_][65]

    ph = SimpleNamespace(read1ByteTxRx=read, read2ByteTxRx=read, getTxRxResult=lambda c: "no status packet")
    motors = {"shoulder_pan": SimpleNamespace(id=1), "shoulder_lift": SimpleNamespace(id=2),
              "gripper": SimpleNamespace(id=6)}
    robot = LeRobotSO101.__new__(LeRobotSO101)
    robot._robot = SimpleNamespace(bus=SimpleNamespace(motors=motors, packet_handler=ph, port_handler=None))

    d = robot.diagnostics()
    assert d["shoulder_pan"]["load_percent"] == -3.0 and d["shoulder_pan"]["faults"] == []
    assert d["shoulder_pan"]["goal_minus_present_deg"] is None
    assert "error" in d["shoulder_lift"]
    g = d["gripper"]
    assert g["voltage_v"] == 12.0 and g["faults"] == ["overload"] and g["goal_minus_present_deg"] < -8

    printed = []
    ctl.robot = robot
    Shell(ctl, confirm=False, print_fn=lambda *a: printed.append(" ".join(map(str, a)))).handle("diag")
    text = "\n".join(printed)
    assert "shoulder_lift  NOT RESPONDING" in text and "gripper is stalled" in text and "55 C" in text


def test_shell_diag_on_dry_run(ctl):
    printed = []
    Shell(ctl, confirm=False, print_fn=lambda *a: printed.append(" ".join(map(str, a)))).handle("diag")
    assert "only works on the real arm" in printed[0]


def test_shell_confirmation_can_cancel(ctl):
    before = ctl.robot.read()
    sh = Shell(ctl, confirm=True, units="m", input_fn=lambda _: "n", print_fn=lambda *a: None)
    sh.handle("go 0.2 0 0.1")
    sh.handle("joints elbow_flex=20")
    sh.handle("home")
    assert ctl.robot.read() == before
