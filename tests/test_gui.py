"""Drives the dashboard window end to end on DryRunRobot (needs a display)."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from so101_llm_ik import ArmConfig  # noqa: E402
from so101_llm_ik.gui import Dashboard  # noqa: E402

URDF = Path(__file__).parent / "fixtures" / "so101_standin.urdf"


def pump(dash, timeout=10.0):
    """Process Tk events until background work is finished."""
    end = time.time() + timeout
    while time.time() < end:
        dash.root.update()
        if not dash.busy and dash._q.empty():
            dash.root.update()
            return
        time.sleep(0.01)
    raise TimeoutError(dash._busy_label)


@pytest.fixture(scope="module")
def tk_root():
    # One interpreter per process: creating many Tk() instances is flaky on Windows.
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def dash(tmp_path, tk_root):
    cfg_path = tmp_path / "arm.json"
    cfg_path.write_text(json.dumps({"urdf_path": str(URDF), "tool_axis_local": [1, 0, 0]}))
    win = tk.Toplevel(tk_root)
    d = Dashboard(win, cfg_path, animate=False)
    pump(d)
    yield d
    win.destroy()


def log_text(d):
    return d.log_box.get("1.0", "end")


def tip(d):
    return np.array(d.state["position_m"])


def test_connects_dry_run_on_start(dash):
    assert dash.ctl is not None and not dash.real
    assert "Dry run" in dash.status.get()
    assert set(dash.joint_vars) == set(dash.cfg.arm_joints)


def test_check_then_go_then_jog(dash):
    for var, v in zip(dash.xyz, ("18", "-5", "5")):
        var.set(v)
    dash.approach.set("down")
    dash.check()
    pump(dash)
    assert dash.plan is not None and dash.ghost[1]

    before = tip(dash).copy()
    dash.check()
    pump(dash)
    assert np.allclose(tip(dash), before)  # check never moves

    dash.go()
    pump(dash)
    assert np.allclose(tip(dash), [0.18, -0.05, 0.05], atol=0.003)

    dash.jog(2, +1)  # +1 cm in z
    pump(dash)
    assert np.allclose(tip(dash), [0.18, -0.05, 0.06], atol=0.003)
    assert "Moved" in log_text(dash)


def test_rejected_target_does_not_move(dash):
    before = tip(dash).copy()
    for var, v in zip(dash.xyz, ("80", "0", "10")):
        var.set(v)
    dash.go()
    pump(dash)
    assert "outside the safe workspace" in log_text(dash)
    assert np.allclose(tip(dash), before)


def test_joint_sliders_move_the_arm(dash):
    dash.joint_vars["elbow_flex"].set(20.0)
    assert dash.ghost is not None  # dragging previews the pose
    dash.move_joints()
    pump(dash)
    assert dash.state["joints_deg"]["elbow_flex"] == pytest.approx(20.0, abs=0.1)


def test_units_switch_converts_entries(dash):
    for var, v in zip(dash.xyz, ("20", "0", "10")):
        var.set(v)
    dash.units.set("mm")
    dash._on_units()
    assert [v.get() for v in dash.xyz] == ["200", "0", "100"]


def test_setup_save_writes_config(dash):
    dash.v_ws["z", 0].set("0.05")
    dash.calib_vars["elbow_flex"][1].set("2.5")
    dash.save_config()
    saved = ArmConfig.from_json(dash.config_path)
    assert saved.workspace.z[0] == 0.05
    assert saved.cal("elbow_flex").offset_deg == 2.5
    assert dash.ctl.cfg.workspace.z[0] == 0.05  # applied to the running controller


def test_bad_setup_is_refused(dash, monkeypatch):
    errors = []
    monkeypatch.setattr("tkinter.messagebox.showerror", lambda *a, **k: errors.append(a))
    dash.v_ws["x", 0].set("abc")
    assert not dash.apply_config()
    assert errors


def test_view_draws(dash):
    dash.root.geometry("1300x850")
    dash.root.update()
    dash.draw()
    assert len(dash.canvas.find_all()) > 50


def test_hover_shows_coordinates(dash):
    dash.root.geometry("1300x850")
    dash.root.update()
    dash.draw()
    top, side = dash._views
    p = tip(dash)
    which, xy = dash.hover_position(*top.pt(-p[1], p[0]))
    assert which == "top" and np.allclose(xy, p[:2], atol=1e-6)
    which, xyz = dash.hover_position(*side.pt(float(np.hypot(p[0], p[1])), p[2]))
    assert which == "side" and np.allclose(xyz, p, atol=1e-6)
    assert dash.hover_position(1, 1) is None

    dash._mouse = tuple(int(v) for v in top.pt(-p[1], p[0]))
    dash._draw_hover()
    assert dash.canvas.find_withtag("hover")
    dash.draw()  # survives the periodic redraw
    assert dash.canvas.find_withtag("hover")
