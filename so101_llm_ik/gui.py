"""Tkinter dashboard for testing IK by hand: type targets, jog, drag joints, watch the arm.

    python -m so101_llm_ik gui --config arm.json                                   # dry run
    python -m so101_llm_ik gui --config arm.json --port COM3 --id my_follower       # real arm

Everything goes through ArmController, so the same workspace, IK and joint-limit checks
apply as in the shell. On the real arm every move shows the planned joint angles and
waits for confirmation, and what you confirm is exactly what gets sent.
"""

from __future__ import annotations

import math
import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Callable

import numpy as np

from .config import ArmConfig, JointCalibration, Workspace
from .controller import APPROACHES, ArmController
from .robot import DryRunRobot, LeRobotSO101
from .shell import UNITS

DECIMALS = {"m": 4, "cm": 2, "mm": 1}
TOOL_AXES = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0), "+y": (0.0, 1.0, 0.0),
             "-y": (0.0, -1.0, 0.0), "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0)}
DEFAULT_URDF = "urdf/so101_new_calib.urdf"
TUNING = (  # ArmConfig field, label
    ("max_joint_speed_deg_s", "Max joint speed (deg/s)"),
    ("control_hz", "Control rate (Hz)"),
    ("joint_limit_margin_deg", "Joint limit margin (deg)"),
    ("position_tol_mm", "Position tolerance (mm)"),
    ("approach_tol_deg", "Approach tolerance (deg)"),
)

ARM, GHOST_OK, GHOST_BAD, TARGET = "#2b2f36", "#e08a00", "#c62828", "#c62828"
WORKSPACE, REACH, GRID, AXIS, POINTER = "#2f6fdb", "#9e9e9e", "#eeeeee", "#c4c4c4", "#2f6fdb"
OK_FG, BAD_FG, INFO_FG = "#1a7f37", "#b42318", "#555555"

SETUP_NOTES = """Setup checklist

1. URDF: put so101_new_calib.urdf (TheRobotStudio SO-ARM100 repo,
   Simulation/SO101/) in urdf/ and point the URDF path at it.

2. Tool axis: press "Inspect URDF". Pick the tip axis that points out
   of the gripper fingers. If it's wrong, approach "down" points the
   wrong way.

3. Calibration: with so101_new_calib.urdf and LeRobot degree mode,
   sign 1 / offset 0 should be right for every joint. Check it on the
   real arm: drive one joint in the Control tab and compare. If it
   turns the other way, set sign -1; if it's shifted, set the offset.

4. Workspace: the box above your table that's free of obstacles,
   in metres. z min is your table clearance.

Apply uses the form without saving; Save writes the file and applies.

The real arm must already be calibrated with LeRobot under the same
robot id, otherwise connecting will wait for calibration input in the
terminal.
"""


def _num(x: float, decimals: int) -> str:
    s = f"{x + 0.0:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _plane_dir(xy_points: np.ndarray) -> np.ndarray:
    """Unit horizontal direction of the arm's vertical plane, taken from the outermost point."""
    for p in xy_points[::-1]:
        n = float(np.linalg.norm(p))
        if n > 1e-3:
            return p / n
    return np.array([1.0, 0.0])


class _View:
    """Maps one 2D projection (h right, v up, metres) into a rectangle on the canvas."""

    def __init__(self, h_range, v_range, x0, y0, scale):
        self.h0, self.h1 = h_range
        self.v0, self.v1 = v_range
        self.x0, self.y0, self.s = x0, y0, scale

    @property
    def x1(self) -> float:
        return self.x0 + (self.h1 - self.h0) * self.s

    @property
    def y1(self) -> float:
        return self.y0 + (self.v1 - self.v0) * self.s

    def pt(self, h: float, v: float) -> tuple[float, float]:
        return self.x0 + (h - self.h0) * self.s, self.y0 + (self.v1 - v) * self.s

    def hv(self, x: float, y: float) -> tuple[float, float]:
        """Inverse of pt: canvas pixel to (h, v) metres."""
        return self.h0 + (x - self.x0) / self.s, self.v1 - (y - self.y0) / self.s

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


class Dashboard:
    def __init__(self, root: tk.Tk | tk.Toplevel, config_path: str | Path, port: str | None = None,
                 robot_id: str | None = None, units: str = "cm", animate: bool = True):
        self.root = root
        self.config_path = Path(config_path).resolve()
        self.animate = animate              # dry run moves at real speed, so you see the motion
        self.cfg: ArmConfig | None = None
        self.ctl: ArmController | None = None
        self.robot = None
        self.real = False
        self.busy = False
        self._busy_label = ""
        self.state: dict[str, Any] | None = None
        self.ghost: tuple[dict[str, float], bool] | None = None   # planned / previewed pose
        self.plan: dict[str, float] | None = None                 # last reachable IK solution
        self.target_m: list[float] | None = None
        self._q: queue.Queue = queue.Queue()
        self._last_refresh = 0.0
        self._last_read_error = ""
        self._loading_sliders = False
        self._redraw_pending = False
        self._views: tuple[_View, _View] | None = None   # (top, side) as last drawn
        self._side_u = np.array([1.0, 0.0])               # horizontal direction of the side view's plane
        self._mouse: tuple[int, int] | None = None

        self.units = tk.StringVar(value=units)
        self._units_prev = units
        self.mode = tk.StringVar(value="real" if port else "dry")
        self.port = tk.StringVar(value=port or "")
        self.robot_id = tk.StringVar(value=robot_id or "")
        self.confirm = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Not connected")
        self.bold = tkfont.nametofont("TkDefaultFont").copy()
        self.bold.configure(weight="bold")

        root.title("SO-101 IK dashboard")
        root.geometry("1320x860")
        root.minsize(1100, 720)
        self._build_topbar()
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.control_tab = ttk.Frame(self.nb, padding=8)
        self.setup_tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(self.control_tab, text="  Control  ")
        self.nb.add(self.setup_tab, text="  Setup  ")
        self._build_control()
        self._build_setup()

        loaded = self._load_config(self.config_path)
        self._update_widgets()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(50, self._poll)
        if loaded and self.mode.get() == "dry" and Path(self.cfg.urdf_path).exists():
            self.connect()
        elif not loaded:
            self.nb.select(self.setup_tab)

    # ======================================================================= layout

    def _build_topbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 8, 8, 6))
        bar.pack(fill="x")
        self.rb_dry = ttk.Radiobutton(bar, text="Dry run", value="dry", variable=self.mode)
        self.rb_real = ttk.Radiobutton(bar, text="Real arm", value="real", variable=self.mode)
        self.rb_dry.pack(side="left")
        self.rb_real.pack(side="left", padx=(6, 16))
        ttk.Label(bar, text="Port").pack(side="left")
        self.port_entry = ttk.Entry(bar, textvariable=self.port, width=12)
        self.port_entry.pack(side="left", padx=(4, 10))
        ttk.Label(bar, text="Robot id").pack(side="left")
        self.id_entry = ttk.Entry(bar, textvariable=self.robot_id, width=14)
        self.id_entry.pack(side="left", padx=(4, 10))
        self.connect_btn = ttk.Button(bar, text="Connect", width=12, command=self.toggle_connect)
        self.connect_btn.pack(side="left")
        ttk.Checkbutton(bar, text="Confirm real-arm moves", variable=self.confirm).pack(side="left", padx=16)
        self.status_lbl = ttk.Label(bar, textvariable=self.status, font=self.bold)
        self.status_lbl.pack(side="right")

    def _build_control(self) -> None:
        t = self.control_tab
        t.columnconfigure(1, weight=1)
        t.rowconfigure(0, weight=1)
        left = ttk.Frame(t)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        right = ttk.Frame(t)
        right.grid(row=0, column=1, sticky="nsew")
        self.actions: list[ttk.Button] = []

        def button(parent, text, cmd, **pack):
            b = ttk.Button(parent, text=text, command=cmd)
            b.pack(side="left", expand=True, fill="x", padx=2, **pack)
            self.actions.append(b)
            return b

        # --- Cartesian target
        f = ttk.LabelFrame(left, text="Cartesian target", padding=8)
        f.pack(fill="x")
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text="Units").pack(side="left")
        cb = ttk.Combobox(row, textvariable=self.units, values=list(UNITS), width=4, state="readonly")
        cb.pack(side="left", padx=(4, 14))
        cb.bind("<<ComboboxSelected>>", self._on_units)
        ttk.Label(row, text="Approach").pack(side="left")
        self.approach = tk.StringVar(value="any")
        ttk.Combobox(row, textvariable=self.approach, values=list(APPROACHES), width=9,
                     state="readonly").pack(side="left", padx=4)

        row = ttk.Frame(f)
        row.pack(fill="x", pady=8)
        self.xyz = [tk.StringVar() for _ in range(3)]
        for i, (name, var) in enumerate(zip("XYZ", self.xyz)):
            ttk.Label(row, text=name, font=self.bold).pack(side="left", padx=(0 if i == 0 else 10, 4))
            e = ttk.Entry(row, textvariable=var, width=9, justify="right")
            e.pack(side="left")
            e.bind("<Return>", lambda _e: self.check())
        self._set_xyz([0.20, 0.0, 0.10])

        row = ttk.Frame(f)
        row.pack(fill="x")
        button(row, "Use current", self.use_current)
        button(row, "Check", self.check)
        button(row, "Go", self.go)

        row = ttk.Frame(f)
        row.pack(fill="x", pady=(12, 0))
        ttk.Label(row, text="Jog step").pack(side="left")
        self.jog_step = tk.StringVar(value=_num(0.01 / UNITS[self.units.get()], DECIMALS[self.units.get()]))
        ttk.Entry(row, textvariable=self.jog_step, width=7, justify="right").pack(side="left", padx=4)
        ttk.Label(row, textvariable=self.units).pack(side="left")
        grid = ttk.Frame(f)
        grid.pack(fill="x", pady=(4, 0))
        for col, axis in enumerate("XYZ"):
            grid.columnconfigure(col, weight=1)
            for r, sign in enumerate((1, -1)):
                b = ttk.Button(grid, text=f"{'+' if sign > 0 else '−'}{axis}",
                               command=lambda a=col, s=sign: self.jog(a, s))
                b.grid(row=r, column=col, padx=2, pady=2, sticky="ew")
                self.actions.append(b)

        # --- Joints
        f = ttk.LabelFrame(left, text="Joints (driver degrees)", padding=8)
        f.pack(fill="x", pady=10)
        self.joint_rows = ttk.Frame(f)
        self.joint_rows.pack(fill="x")
        self.joint_rows.columnconfigure(1, weight=1)
        self.joint_vars: dict[str, tk.DoubleVar] = {}
        self.joint_text: dict[str, tk.StringVar] = {}
        self.joint_range: dict[str, tuple[float, float]] = {}
        ttk.Label(self.joint_rows, text="Connect to load the joints.", foreground=INFO_FG).grid(row=0, column=0)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(8, 0))
        button(row, "Load current", self.load_current_joints)
        button(row, "Load IK solution", self.load_solution)
        button(row, "Move joints", self.move_joints)

        # --- Gripper / home
        f = ttk.LabelFrame(left, text="Gripper (0 = closed, 100 = open)", padding=8)
        f.pack(fill="x")
        self.grip = tk.DoubleVar(value=50)
        tk.Scale(f, from_=0, to=100, orient="horizontal", resolution=1, variable=self.grip,
                 showvalue=True, length=220).pack(side="left", fill="x", expand=True)
        b = ttk.Button(f, text="Set", command=self.set_gripper, width=8)
        b.pack(side="left", padx=(8, 0), anchor="s")
        self.actions.append(b)
        row = ttk.Frame(left)
        row.pack(fill="x", pady=(10, 0))
        button(row, "Home", self.home)
        button(row, "Refresh state", self.refresh_state)

        # --- Right: view, state, log
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=2)
        right.rowconfigure(2, weight=1)
        self.canvas = tk.Canvas(right, background="white", highlightthickness=1,
                                highlightbackground="#d0d0d0", cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _e: self._schedule_draw())
        self.canvas.bind("<Motion>", self._on_hover)
        self.canvas.bind("<Leave>", self._on_hover)

        sf = ttk.LabelFrame(right, text="Arm state", padding=(8, 4))
        sf.grid(row=1, column=0, sticky="ew", pady=8)
        sf.columnconfigure(1, weight=1)
        sf.columnconfigure(3, weight=1)
        self.state_vars = {k: tk.StringVar(value="–") for k in ("tip", "points", "gripper", "joints")}
        for i, (key, label) in enumerate((("tip", "Tip position"), ("points", "Gripper points"),
                                          ("gripper", "Gripper"), ("joints", "Joints"))):
            r, c = divmod(i, 2) if key != "joints" else (2, 0)
            ttk.Label(sf, text=label, foreground=INFO_FG).grid(row=r, column=c * 2, sticky="w", padx=(0, 8))
            ttk.Label(sf, textvariable=self.state_vars[key], font=self.bold).grid(
                row=r, column=c * 2 + 1, sticky="w", columnspan=3 if key == "joints" else 1)

        self.log_box = ScrolledText(right, height=9, wrap="word", font=("Consolas", 9),
                                    state="disabled", relief="flat", borderwidth=1)
        self.log_box.grid(row=2, column=0, sticky="nsew")
        for tag, fg in (("ok", OK_FG), ("bad", BAD_FG), ("info", INFO_FG)):
            self.log_box.tag_configure(tag, foreground=fg)

    def _build_setup(self) -> None:
        t = self.setup_tab
        t.columnconfigure(1, weight=1)
        t.rowconfigure(1, weight=1)

        top = ttk.Frame(t)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(top, text="Config file").pack(side="left")
        self.config_var = tk.StringVar(value=str(self.config_path))
        ttk.Entry(top, textvariable=self.config_var, state="readonly").pack(
            side="left", fill="x", expand=True, padx=6)
        for text, cmd in (("Open…", self.open_config), ("Save", self.save_config),
                          ("Save as…", self.save_config_as), ("Apply", self.apply_config)):
            ttk.Button(top, text=text, command=cmd).pack(side="left", padx=2)

        form = ttk.Frame(t)
        form.grid(row=1, column=0, sticky="nw")

        f = ttk.LabelFrame(form, text="Robot model", padding=8)
        f.pack(fill="x")
        self.v_urdf, self.v_tip = tk.StringVar(), tk.StringVar()
        self.v_joints, self.v_axis = tk.StringVar(), tk.StringVar()
        ttk.Label(f, text="URDF").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.v_urdf, width=46).grid(row=0, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(f, text="Browse…", command=self.browse_urdf).grid(row=0, column=2)
        ttk.Label(f, text="Tip link").grid(row=1, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.v_tip, width=46).grid(row=1, column=1, sticky="ew", padx=6, pady=2)
        ttk.Label(f, text="Arm joints").grid(row=2, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.v_joints, width=46).grid(row=2, column=1, sticky="ew", padx=6, pady=2)
        ttk.Label(f, text="Tool axis").grid(row=3, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.v_axis, values=list(TOOL_AXES), width=10).grid(
            row=3, column=1, sticky="w", padx=6, pady=2)
        ttk.Button(f, text="Inspect URDF", command=self.inspect_urdf).grid(row=3, column=2)

        self.calib_frame = ttk.LabelFrame(form, text="Joint calibration   (driver = sign × URDF + offset)",
                                          padding=8)
        self.calib_frame.pack(fill="x", pady=10)
        self.calib_vars: dict[str, tuple[tk.StringVar, tk.StringVar, tk.StringVar]] = {}

        row = ttk.Frame(form)
        row.pack(fill="x")
        f = ttk.LabelFrame(row, text="Workspace (metres, base frame)", padding=8)
        f.pack(side="left", fill="y")
        self.v_ws = {(ax, i): tk.StringVar() for ax in "xyz" for i in (0, 1)}
        ttk.Label(f, text="min").grid(row=0, column=1)
        ttk.Label(f, text="max").grid(row=0, column=2)
        for r, ax in enumerate("xyz", start=1):
            ttk.Label(f, text=ax, font=self.bold).grid(row=r, column=0, padx=(0, 6))
            for i in (0, 1):
                ttk.Entry(f, textvariable=self.v_ws[ax, i], width=8, justify="right").grid(
                    row=r, column=i + 1, padx=2, pady=2)
        ttk.Label(f, text="z min = table clearance", foreground=INFO_FG).grid(row=4, column=0, columnspan=3)

        f = ttk.LabelFrame(row, text="Motion and tolerances", padding=8)
        f.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self.v_tuning = {k: tk.StringVar() for k, _ in TUNING}
        for r, (k, label) in enumerate(TUNING):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w")
            ttk.Entry(f, textvariable=self.v_tuning[k], width=8, justify="right").grid(
                row=r, column=1, padx=6, pady=2)

        out = ttk.LabelFrame(t, text="Notes and URDF inspection", padding=8)
        out.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
        self.inspect_box = ScrolledText(out, wrap="word", font=("Consolas", 9), state="disabled",
                                        relief="flat")
        self.inspect_box.pack(fill="both", expand=True)
        self._set_text(self.inspect_box, SETUP_NOTES)

    def _build_calib_rows(self, cfg: ArmConfig) -> None:
        for w in self.calib_frame.winfo_children():
            w.destroy()
        self.calib_vars = {}
        for c, head in enumerate(("Joint", "Sign", "Offset (deg)", "Home (deg)")):
            ttk.Label(self.calib_frame, text=head, foreground=INFO_FG).grid(row=0, column=c, sticky="w", padx=4)
        for r, n in enumerate(cfg.arm_joints, start=1):
            cal = cfg.cal(n)
            v = (tk.StringVar(value=_num(cal.sign, 3)), tk.StringVar(value=_num(cal.offset_deg, 3)),
                 tk.StringVar(value=_num(cfg.home_deg.get(n, 0.0), 3)))
            ttk.Label(self.calib_frame, text=n).grid(row=r, column=0, sticky="w", padx=4)
            ttk.Combobox(self.calib_frame, textvariable=v[0], values=("1", "-1"), width=4,
                         state="readonly").grid(row=r, column=1, padx=4, pady=1)
            ttk.Entry(self.calib_frame, textvariable=v[1], width=9, justify="right").grid(row=r, column=2, padx=4)
            ttk.Entry(self.calib_frame, textvariable=v[2], width=9, justify="right").grid(row=r, column=3, padx=4)
            self.calib_vars[n] = v

    def _build_joint_sliders(self) -> None:
        for w in self.joint_rows.winfo_children():
            w.destroy()
        self.joint_vars, self.joint_text = {}, {}
        if self.ctl is None:
            ttk.Label(self.joint_rows, text="Connect to load the joints.", foreground=INFO_FG).grid(row=0, column=0)
            return
        self.joint_range = self.ctl.driver_limits_deg()
        for r, (n, (lo, hi)) in enumerate(self.joint_range.items()):
            var, text = tk.DoubleVar(value=min(max(0.0, lo), hi)), tk.StringVar()
            ttk.Label(self.joint_rows, text=n).grid(row=r, column=0, sticky="w")
            tk.Scale(self.joint_rows, from_=lo, to=hi, orient="horizontal", resolution=0.1,
                     variable=var, showvalue=False, length=190).grid(row=r, column=1, sticky="ew", padx=6)
            sb = ttk.Spinbox(self.joint_rows, from_=lo, to=hi, increment=1, textvariable=text, width=7,
                             justify="right", command=lambda n=n: self._commit_spin(n))
            sb.grid(row=r, column=2, pady=1)
            sb.bind("<Return>", lambda _e, n=n: self._commit_spin(n))
            sb.bind("<FocusOut>", lambda _e, n=n: self._commit_spin(n))
            var.trace_add("write", lambda *_a, n=n: self._on_slider(n))
            self.joint_vars[n], self.joint_text[n] = var, text
            text.set(_num(var.get(), 1))

    # ======================================================================= config

    def _load_config(self, path: Path) -> bool:
        """Fill the Setup form from a file. Returns False (and fills defaults) when it can't."""
        self.config_path = path.resolve()
        self.config_var.set(str(self.config_path))
        cfg, ok = None, False
        if path.exists():
            try:
                cfg, ok = ArmConfig.from_json(path), True
            except Exception as e:
                messagebox.showerror("Could not read config", f"{path}\n\n{type(e).__name__}: {e}")
        if cfg is None:
            cfg = ArmConfig(urdf_path=str(self.config_path.parent / DEFAULT_URDF))
            if not path.exists():
                self.log(f"No config at {path}. Fill in the Setup tab and press Save to create it.", "info")
        self._config_to_form(cfg)
        self.cfg = cfg if ok else None
        return ok

    def _config_to_form(self, cfg: ArmConfig) -> None:
        urdf = Path(cfg.urdf_path)
        try:
            shown = urdf.resolve().relative_to(self.config_path.parent).as_posix()
        except ValueError:
            shown = str(urdf)
        self.v_urdf.set(shown)
        self.v_tip.set(cfg.tip_link)
        self.v_joints.set(", ".join(cfg.arm_joints))
        axis = tuple(float(v) for v in cfg.tool_axis_local)
        self.v_axis.set(next((k for k, v in TOOL_AXES.items() if np.allclose(v, axis)),
                             " ".join(_num(v, 4) for v in axis)))
        self._build_calib_rows(cfg)
        for ax in "xyz":
            for i in (0, 1):
                self.v_ws[ax, i].set(_num(getattr(cfg.workspace, ax)[i], 4))
        for k, _ in TUNING:
            self.v_tuning[k].set(_num(getattr(cfg, k), 3))

    def _form_to_config(self) -> ArmConfig:
        def num(var: tk.StringVar, label: str) -> float:
            try:
                v = float(var.get())
            except ValueError:
                raise ValueError(f"{label}: '{var.get()}' is not a number.") from None
            if not math.isfinite(v):
                raise ValueError(f"{label} must be finite.")
            return v

        joints = [s.strip() for s in self.v_joints.get().replace(";", ",").split(",") if s.strip()]
        if not joints:
            raise ValueError("List at least one arm joint.")
        urdf = Path(self.v_urdf.get().strip())
        if not str(urdf).strip() or str(urdf) == ".":
            raise ValueError("Set the URDF path.")
        if not urdf.is_absolute():
            urdf = self.config_path.parent / urdf
        axis_txt = self.v_axis.get().strip()
        if axis_txt in TOOL_AXES:
            axis = TOOL_AXES[axis_txt]
        else:
            try:
                axis = tuple(float(v) for v in axis_txt.replace(",", " ").split())
            except ValueError:
                axis = ()
            if len(axis) != 3 or not any(axis):
                raise ValueError("Tool axis must be one of +x … -z, or three numbers like '0 0 1'.")
        calibration, home = {}, {}
        for n in joints:
            if n in self.calib_vars:
                sign, off, h = self.calib_vars[n]
                calibration[n] = JointCalibration(sign=num(sign, f"{n} sign"), offset_deg=num(off, f"{n} offset"))
                home[n] = num(h, f"{n} home")
        ws = {}
        for ax in "xyz":
            lo, hi = num(self.v_ws[ax, 0], f"workspace {ax} min"), num(self.v_ws[ax, 1], f"workspace {ax} max")
            if lo >= hi:
                raise ValueError(f"Workspace {ax}: min must be below max.")
            ws[ax] = (lo, hi)
        tuning = {k: num(self.v_tuning[k], label) for k, label in TUNING}
        for k, label in TUNING:
            if tuning[k] <= 0 and k != "joint_limit_margin_deg":
                raise ValueError(f"{label} must be positive.")
        return ArmConfig(urdf_path=str(urdf.resolve()), tip_link=self.v_tip.get().strip(), arm_joints=joints,
                         tool_axis_local=axis, calibration=calibration, workspace=Workspace(**ws),
                         home_deg=home, **tuning)

    def _form_or_complain(self) -> ArmConfig | None:
        try:
            cfg = self._form_to_config()
        except ValueError as e:
            messagebox.showerror("Setup", str(e))
            self.nb.select(self.setup_tab)
            return None
        if set(cfg.arm_joints) != set(self.calib_vars):
            self._build_calib_rows(cfg)
        return cfg

    def apply_config(self) -> bool:
        if self.busy:
            return False
        cfg = self._form_or_complain()
        if cfg is None:
            return False
        if self.ctl is not None:
            try:
                robot = self.robot
                if not self.real and list(cfg.arm_joints) != list(self.cfg.arm_joints):
                    robot = DryRunRobot(cfg.arm_joints, start=self.robot.read())
                ctl = ArmController(cfg, robot, realtime=self.real or self.animate)
            except Exception as e:
                messagebox.showerror("Could not load the arm model", f"{type(e).__name__}: {e}")
                return False
            self.robot, self.ctl = robot, ctl
            self.ghost = self.plan = None
            self._build_joint_sliders()
            self.load_current_joints()
            self.refresh_state()
        self.cfg = cfg
        self.log("Setup applied." + ("" if self.ctl else " Press Connect to use it."), "ok")
        return True

    def save_config(self, path: Path | None = None) -> None:
        cfg = self._form_or_complain()   # URDF path resolved against the current file's folder
        if cfg is None:
            return
        if path is not None:
            self.config_path = path.resolve()
            self.config_var.set(str(self.config_path))
        try:
            cfg.to_json(self.config_path)
        except OSError as e:
            messagebox.showerror("Save failed", str(e))
            return
        self._config_to_form(cfg)        # shows the URDF path relative to the saved file
        self.log(f"Saved {self.config_path}.", "ok")
        self.apply_config()

    def save_config_as(self) -> None:
        path = filedialog.asksaveasfilename(initialdir=self.config_path.parent, initialfile=self.config_path.name,
                                            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            self.save_config(Path(path))

    def open_config(self) -> None:
        path = filedialog.askopenfilename(initialdir=self.config_path.parent, filetypes=[("JSON", "*.json")])
        if path and self._load_config(Path(path)):
            self.log(f"Loaded {path}. Press Apply (or Connect) to use it.", "info")

    def browse_urdf(self) -> None:
        path = filedialog.askopenfilename(initialdir=self.config_path.parent,
                                          filetypes=[("URDF", "*.urdf"), ("All files", "*.*")])
        if path:
            p = Path(path)
            try:
                p = p.resolve().relative_to(self.config_path.parent)
            except ValueError:
                pass
            self.v_urdf.set(p.as_posix())

    def inspect_urdf(self) -> None:
        cfg = self._form_or_complain()
        if cfg is None:
            return
        try:
            text = ArmController(cfg, DryRunRobot(cfg.arm_joints)).inspect()
        except Exception as e:
            text = f"Could not load the URDF:\n{type(e).__name__}: {e}"
        self._set_text(self.inspect_box, text + "\n\n" + "-" * 60 + "\n\n" + SETUP_NOTES)

    # ======================================================================= connection

    def toggle_connect(self) -> None:
        if self.busy:
            return
        if self.ctl is not None:
            self.disconnect()
        else:
            self.connect()

    def connect(self) -> None:
        cfg = self._form_or_complain()
        if cfg is None:
            return
        if not Path(cfg.urdf_path).exists():
            messagebox.showerror("URDF not found", f"{cfg.urdf_path}\ndoes not exist. Set the URDF path in Setup.")
            self.nb.select(self.setup_tab)
            return
        real = self.mode.get() == "real"
        port, rid = self.port.get().strip(), self.robot_id.get().strip()
        if real and not (port and rid):
            messagebox.showerror("Real arm", "Enter the serial port (e.g. COM3) and the robot id "
                                             "you calibrated the arm with in LeRobot.")
            return
        animate = self.animate

        def work():
            robot = LeRobotSO101(port, rid) if real else DryRunRobot(cfg.arm_joints, start=cfg.home_deg)
            ctl = ArmController(cfg, robot, realtime=real or animate)  # parse the URDF before touching hardware
            robot.connect()
            return robot, ctl

        def done(res):
            self.robot, self.ctl = res
            self.cfg, self.real = cfg, real
            self.ghost = self.plan = self.target_m = None
            self._build_joint_sliders()
            self.load_current_joints()
            self.log(f"Connected: {'REAL ARM on ' + port if real else 'dry run, nothing physical moves'}.", "ok")
            self.log(self.ctl.workspace_description(), "info")

        self._run("Connecting", work, done)

    def disconnect(self) -> None:
        robot = self.robot
        self.ctl = self.robot = self.state = None
        self.ghost = self.plan = None
        try:
            robot.disconnect()
        except Exception as e:
            self.log(f"Disconnect: {type(e).__name__}: {e}", "bad")
        self._build_joint_sliders()
        self._update_state_panel()
        self._update_widgets()
        self.draw()
        self.log("Disconnected.", "info")

    def _on_close(self) -> None:
        if self.busy and self.real and not messagebox.askyesno(
                "Quit", "The arm is moving. Quit anyway? (The servos hold their last command.)"):
            return
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception:
                pass
        self.root.destroy()

    # ======================================================================= background work

    def _run(self, label: str, fn: Callable[[], Any], done: Callable[[Any], None] | None = None) -> None:
        """Run fn on a worker thread so the window stays responsive; done(result) runs on the UI thread."""
        if self.busy:
            return
        self.busy, self._busy_label = True, label
        self._update_widgets()

        def work():
            try:
                self._q.put((done, fn(), None))
            except Exception as e:
                self._q.put((done, None, e))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self) -> None:
        if not self.root.winfo_exists():
            return
        try:
            while True:
                done, res, err = self._q.get_nowait()
                self.busy = False
                if err is not None:
                    self.log(f"Error: {type(err).__name__}: {err}", "bad")
                elif done is not None:
                    try:
                        done(res)
                    except Exception as e:
                        self.log(f"Error: {type(e).__name__}: {e}", "bad")
                self._update_widgets()
                self.refresh_state()
        except queue.Empty:
            pass
        # Live view: during dry-run moves to animate them, otherwise once a second.
        # Never read the real arm while a worker is using the bus.
        if self.ctl is not None and not (self.real and self.busy):
            if time.monotonic() - self._last_refresh > (0.08 if self.busy else 1.0):
                self.refresh_state()
        self.root.after(50, self._poll)

    def _update_widgets(self) -> None:
        connected = self.ctl is not None
        for w in self.actions:
            w.state(["!disabled"] if connected and not self.busy else ["disabled"])
        for w in (self.rb_dry, self.rb_real, self.port_entry, self.id_entry):
            w.state(["disabled"] if connected or self.busy else ["!disabled"])
        self.connect_btn.configure(text="Disconnect" if connected else "Connect")
        self.connect_btn.state(["disabled"] if self.busy else ["!disabled"])
        if not connected:
            text, color = "Not connected", INFO_FG
        elif self.real:
            text, color = f"REAL ARM · {self.port.get().strip()}", BAD_FG
        else:
            text, color = "Dry run · nothing physical moves", OK_FG
        if self.busy:
            text += f" · {self._busy_label}…"
        self.status.set(text)
        self.status_lbl.configure(foreground=color)

    # ======================================================================= state

    def refresh_state(self) -> None:
        self._last_refresh = time.monotonic()
        if self.ctl is None or (self.real and self.busy):
            return
        try:
            self.state = self.ctl.get_arm_state()
            self._last_read_error = ""
        except Exception as e:
            msg = f"Could not read the arm: {type(e).__name__}: {e}"
            if msg != self._last_read_error:
                self.log(msg, "bad")
            self._last_read_error = msg
            self.state = None
        self._update_state_panel()
        self._schedule_draw()

    def _update_state_panel(self) -> None:
        s = self.state
        if s is None:
            for v in self.state_vars.values():
                v.set("–")
            return
        self.state_vars["tip"].set(self._fmt_pos(s["position_m"]))
        self.state_vars["points"].set("(" + ", ".join(f"{v:+.2f}" for v in s["pointing_direction"]) + ")")
        self.state_vars["gripper"].set(f"{s['gripper_open_percent']:.0f} % open")
        self.state_vars["joints"].set(self._fmt_joints(s["joints_deg"]))

    # ======================================================================= Cartesian commands

    def _read_target(self) -> list[float] | None:
        try:
            return [float(v.get()) * UNITS[self.units.get()] for v in self.xyz]
        except ValueError:
            messagebox.showerror("Target", "X, Y and Z must be numbers.")
            return None

    def _set_xyz(self, p_m) -> None:
        k, d = UNITS[self.units.get()], DECIMALS[self.units.get()]
        for var, v in zip(self.xyz, p_m):
            var.set(_num(v / k, d))

    def _on_units(self, _e=None) -> None:
        old, new = self._units_prev, self.units.get()
        if old == new:
            return
        f = UNITS[old] / UNITS[new]
        for var in [*self.xyz, self.jog_step]:
            try:
                var.set(_num(float(var.get()) * f, DECIMALS[new]))
            except ValueError:
                pass
        self._units_prev = new
        self._update_state_panel()

    def use_current(self) -> None:
        if self.state is not None:
            self._set_xyz(self.state["position_m"])

    def check(self) -> None:
        t = self._read_target()
        if t is None or self.ctl is None:
            return
        a = self.approach.get()
        self._run("Solving", lambda: self.ctl.check_reachable(*t, approach=a),
                  lambda out: self._show_plan("Check", t, a, out))

    def go(self, t: list[float] | None = None) -> None:
        t = t if t is not None else self._read_target()
        if t is None or self.ctl is None:
            return
        a = self.approach.get()
        if self.real and self.confirm.get():
            def after_plan(out):
                if self._show_plan("Plan", t, a, out) and self._confirm_joints(
                        f"Move the tip to {self._fmt_pos(t)}?", out["joints_deg"]):
                    goal = out["joints_deg"]
                    self._run("Moving", lambda: self.ctl.move_joints(goal), lambda res: self._show_move(t, res))
            self._run("Solving", lambda: self.ctl.check_reachable(*t, approach=a), after_plan)
        else:
            self.target_m = t
            self._run("Moving", lambda: self.ctl.move_to(*t, approach=a), lambda res: self._show_move(t, res))

    def jog(self, axis: int, sign: int) -> None:
        if self.state is None:
            return
        try:
            step = float(self.jog_step.get()) * UNITS[self.units.get()]
        except ValueError:
            messagebox.showerror("Jog", "Jog step must be a number.")
            return
        t = list(self.state["position_m"])
        t[axis] += sign * step
        self._set_xyz(t)
        self.go(t)

    def _show_plan(self, verb: str, t: list[float], approach: str, out: dict[str, Any]) -> bool:
        self.target_m = t
        ok = bool(out.get("ok") and out.get("reachable"))
        joints = out.get("joints_deg")
        self.ghost = (joints, ok) if joints else None
        self.plan = joints if ok else None
        where = self._fmt_pos(t) + ("" if approach == "any" else f", approach {approach}")
        if ok:
            extra = "" if out.get("approach_error_deg") is None else f", approach {out['approach_error_deg']:.2f}°"
            self.log(f"{verb} {where}: reachable, {out['position_error_mm']:.2f} mm{extra}\n"
                     f"    joints {self._fmt_joints(joints)}", "ok")
        else:
            msg = f"{verb} {where}: {out.get('error') or out.get('reason')}"
            if joints:
                msg += f"\n    best attempt (orange-red dashes) is {out['position_error_mm']:.1f} mm off"
            self.log(msg, "bad")
        self._schedule_draw()
        return ok

    def _show_move(self, t: list[float], res: dict[str, Any]) -> None:
        self.target_m = t
        if res.get("ok"):
            err = float(np.linalg.norm(np.array(res["position_m"]) - np.array(t))) * 1000.0
            self.log(f"Moved. Tip at {self._fmt_pos(res['position_m'])}, {err:.2f} mm from target.", "ok")
            self.ghost = None
            self.load_current_joints()
        else:
            msg = f"Not moved: {res.get('error')}"
            if "closest_reachable_m" in res:
                msg += f"\n    closest reachable: {self._fmt_pos(res['closest_reachable_m'])}"
            self.log(msg, "bad")
        self._schedule_draw()

    # ======================================================================= joint commands

    def _on_slider(self, n: str) -> None:
        try:
            v = self.joint_vars[n].get()
        except (tk.TclError, KeyError):
            return
        self.joint_text[n].set(_num(v, 1))
        if not self._loading_sliders:
            self.ghost = (self._slider_values(), True)
            self._schedule_draw()

    def _commit_spin(self, n: str) -> None:
        if n not in self.joint_text:
            return
        lo, hi = self.joint_range[n]
        try:
            v = min(max(float(self.joint_text[n].get()), lo), hi)
        except ValueError:
            v = self.joint_vars[n].get()
        self.joint_vars[n].set(round(v, 1))
        self.joint_text[n].set(_num(v, 1))

    def _slider_values(self) -> dict[str, float]:
        return {n: float(v.get()) for n, v in self.joint_vars.items()}

    def _set_sliders(self, joints: dict[str, float]) -> None:
        self._loading_sliders = True
        try:
            for n, var in self.joint_vars.items():
                if n in joints:
                    lo, hi = self.joint_range[n]
                    var.set(round(min(max(joints[n], lo), hi), 1))
        finally:
            self._loading_sliders = False

    def load_current_joints(self) -> None:
        if self.ctl is not None and not (self.real and self.busy):
            state = self.ctl.robot.read()
            self._set_sliders(state)
            self.grip.set(round(state.get("gripper", self.grip.get())))
            if self.ghost is not None and self.ghost[0] is not self.plan:
                self.ghost = None
            self._schedule_draw()

    def load_solution(self) -> None:
        if not self.plan:
            self.log("No IK solution yet. Press Check on a reachable target first.", "info")
            return
        self._set_sliders(self.plan)
        self.ghost = (dict(self.plan), True)
        self._schedule_draw()

    def move_joints(self) -> None:
        if self.ctl is None:
            return
        for n in self.joint_text:
            self._commit_spin(n)
        goal = self._slider_values()
        if self.real and self.confirm.get() and not self._confirm_joints("Move to these joint angles?", goal):
            return

        def done(res):
            if res.get("ok"):
                self.log(f"Moved joints. Tip at {self._fmt_pos(res['position_m'])}.", "ok")
                self.ghost = None
            else:
                self.log(f"Not moved: {res.get('error')}", "bad")

        self._run("Moving", lambda: self.ctl.move_joints(goal), done)

    def set_gripper(self) -> None:
        v = float(self.grip.get())
        self._run("Gripper", lambda: self.ctl.set_gripper(v),
                  lambda res: self.log(f"Gripper {res.get('gripper_open_percent', v):.0f} % open." if res.get("ok")
                                       else f"Gripper: {res.get('error')}", "ok" if res.get("ok") else "bad"))

    def home(self) -> None:
        if self.ctl is None:
            return
        goal = {n: float(self.cfg.home_deg.get(n, 0.0)) for n in self.cfg.arm_joints}
        if self.real and self.confirm.get() and not self._confirm_joints("Move to the home pose?", goal):
            return

        def done(res):
            self.log(f"Home. Tip at {self._fmt_pos(res['position_m'])}.", "ok")
            self.ghost = None
            self.load_current_joints()

        self._run("Homing", self.ctl.go_home, done)

    def _confirm_joints(self, question: str, goal: dict[str, float]) -> bool:
        cur = self.ctl.robot.read()
        lines = [f"{n}:  {cur.get(n, float('nan')):.1f}°  →  {v:.1f}°" for n, v in goal.items()]
        biggest = max((abs(v - cur.get(n, v)) for n, v in goal.items()), default=0.0)
        return messagebox.askyesno(
            "Confirm move", f"{question}\n\n" + "\n".join(lines) + f"\n\nLargest joint change: {biggest:.0f}°",
            icon="warning", default="no")

    # ======================================================================= drawing

    def _schedule_draw(self) -> None:
        if not self._redraw_pending:
            self._redraw_pending = True
            self.root.after_idle(self.draw)

    def draw(self) -> None:
        self._redraw_pending = False
        c = self.canvas
        c.delete("all")
        self._views = None
        W, H = c.winfo_width(), c.winfo_height()
        if W < 80 or H < 80:
            return
        if self.ctl is None:
            c.create_text(W / 2, H / 2, fill=INFO_FG,
                          text="Not connected.\nPick Dry run or Real arm and press Connect.", justify="center")
            return

        R = self.ctl.solver.max_reach_m
        R = 0.5 if not math.isfinite(R) else R + 0.04
        pad, title_h, legend_h = 16, 22, 26
        top_h, top_v = (-R, R), (-0.4 * R, R)              # top view: right = robot's right (-y), up = +x
        side_h, side_v = (-0.4 * R, R), (-0.15 * R, R)     # side view: right = away from base, up = +z
        span_w = (top_h[1] - top_h[0]) + (side_h[1] - side_h[0])
        span_h = max(top_v[1] - top_v[0], side_v[1] - side_v[0])
        s = min((W - 3 * pad) / span_w, (H - 2 * pad - title_h - legend_h) / span_h)
        x0 = (W - span_w * s - pad) / 2
        y0 = pad + title_h
        top = _View(top_h, top_v, x0, y0, s)
        side = _View(side_h, side_v, top.x1 + pad, y0, s)
        self._views = (top, side)

        ws = self.ctl.cfg.workspace
        center = getattr(self.ctl.solver, "_reach_center", np.zeros(3))
        reach = self.ctl.solver.max_reach_m

        for view, title in ((top, "Top view   (x forward ↑, y left ←)"),
                            (side, "Side view   (in the arm's vertical plane)")):
            self._draw_grid(view)
            c.create_text(view.x0, view.y0 - 6, text=title, anchor="sw", fill=INFO_FG)
        c.create_text(top.x0 + 4, top.y1 - 4, text="grid 10 cm", anchor="sw", fill=REACH, font=("Segoe UI", 8))

        # workspace and reach
        if math.isfinite(reach):
            self._draw_circle(top, -center[1], center[0], reach)
            self._draw_circle(side, 0.0, center[2], reach)
        c.create_rectangle(*top.pt(-ws.y[1], ws.x[1]), *top.pt(-ws.y[0], ws.x[0]), outline=WORKSPACE, dash=(6, 4))
        for z, label in ((ws.z[0], "z min"), (ws.z[1], "z max")):
            c.create_line(*side.pt(side.h0, z), *side.pt(side.h1, z), fill=WORKSPACE, dash=(6, 4))
            c.create_text(*side.pt(side.h1, z), text=label + " ", anchor="se", fill=WORKSPACE, font=("Segoe UI", 8))
        c.create_line(*side.pt(side.h0, 0.0), *side.pt(side.h1, 0.0), fill="#8a8a8a", width=2)

        # planned / previewed pose, then the current pose on top
        if self.ghost is not None:
            try:
                pts, a = self.ctl.pose_geometry(self.ghost[0])
                color = GHOST_OK if self.ghost[1] else GHOST_BAD
                self._draw_arm(top, side, pts, a, color, width=3, dash=(7, 4))
            except Exception:
                pass
        if self.state is not None:
            try:
                pts, a = self.ctl.pose_geometry()
                self._draw_arm(top, side, pts, a, ARM, width=6, dash=None)
                self._side_u = _plane_dir(pts[:, :2])
            except Exception:
                pass

        if self.target_m is not None:
            t = np.array(self.target_m)
            u = _plane_dir(t[None, :2])
            for view, (h, v) in ((top, (-t[1], t[0])), (side, (float(t[:2] @ u), t[2]))):
                x, y = view.pt(h, v)
                c.create_line(x - 7, y - 7, x + 7, y + 7, fill=TARGET, width=2)
                c.create_line(x - 7, y + 7, x + 7, y - 7, fill=TARGET, width=2)

        self._draw_legend(pad, H - legend_h + 6)
        self._draw_hover()

    def _on_hover(self, e: tk.Event) -> None:
        self._mouse = None if e.type == tk.EventType.Leave else (e.x, e.y)
        self._draw_hover()

    def hover_position(self, x: float, y: float) -> tuple[str, list[float]] | None:
        """Base-frame position under canvas pixel (x, y): ("top", [x, y]) or ("side", [x, y, z])."""
        if self._views is None:
            return None
        top, side = self._views
        if top.contains(x, y):
            h, v = top.hv(x, y)
            return "top", [v, -h]
        if side.contains(x, y):
            h, v = side.hv(x, y)
            return "side", [h * self._side_u[0], h * self._side_u[1], v]
        return None

    def _draw_hover(self) -> None:
        """Coordinate readout next to the mouse; redrawn after every full redraw."""
        c = self.canvas
        c.delete("hover")
        hit = None if self._mouse is None else self.hover_position(*self._mouse)
        if hit is None:
            return
        which, p = hit
        u = self.units.get()
        k, d = UNITS[u], DECIMALS[u]
        text = "   ".join(f"{name} {_num(v / k, d)}" for name, v in zip("xyz", p)) + f" {u}"
        if which == "side":
            text += f"\nr {_num(math.hypot(p[0], p[1]) / k, d)} {u} (arm plane)"
        mx, my = self._mouse
        right, below = mx < self._views[1].x1 - 170, my < 60
        t = c.create_text(mx + 14 if right else mx - 14, my + 14 if below else my - 14, text=text,
                          anchor=("n" if below else "s") + ("w" if right else "e"),
                          fill=ARM, font=("Consolas", 9), tags="hover")
        x0, y0, x1, y1 = c.bbox(t)
        c.create_rectangle(x0 - 4, y0 - 2, x1 + 4, y1 + 2, fill="white", outline="#c4c4c4", tags="hover")
        c.tag_raise(t)

    def _draw_grid(self, view: _View) -> None:
        c = self.canvas
        c.create_rectangle(view.x0, view.y0, view.x1, view.y1, outline="#dddddd")
        for h in np.arange(math.ceil(view.h0 / 0.1) * 0.1, view.h1, 0.1):
            c.create_line(*view.pt(h, view.v0), *view.pt(h, view.v1), fill=AXIS if abs(h) < 1e-9 else GRID)
        for v in np.arange(math.ceil(view.v0 / 0.1) * 0.1, view.v1, 0.1):
            c.create_line(*view.pt(view.h0, v), *view.pt(view.h1, v), fill=AXIS if abs(v) < 1e-9 else GRID)

    def _draw_circle(self, view: _View, h: float, v: float, r: float) -> None:
        """Dashed circle, clipped to the view (Tk canvases don't clip)."""
        ang = np.linspace(0.0, 2 * math.pi, 241)
        pts = [(h + r * math.cos(a), v + r * math.sin(a)) for a in ang]
        inside = [view.h0 <= ph <= view.h1 and view.v0 <= pv <= view.v1 for ph, pv in pts]
        run: list[float] = []
        for (ph, pv), ok in zip(pts + [pts[0]], inside + [False]):
            if ok:
                run += view.pt(ph, pv)
            elif len(run) >= 4:
                self.canvas.create_line(*run, fill=REACH, dash=(2, 4))
                run = []
            else:
                run = []

    def _draw_arm(self, top: _View, side: _View, pts: np.ndarray, a: np.ndarray,
                  color: str, width: int, dash) -> None:
        c = self.canvas
        u = _plane_dir(pts[:, :2])
        for view, proj in ((top, lambda p: (-p[1], p[0])), (side, lambda p: (float(p[:2] @ u), p[2]))):
            xy = [view.pt(*proj(p)) for p in pts]
            c.create_line(*[k for p in xy for k in p], fill=color, width=width, capstyle="round",
                          joinstyle="round", dash=dash)
            r = 4 if dash is None else 3
            for x, y in xy[1:-1]:
                c.create_oval(x - r, y - r, x + r, y + r, fill="white", outline=color, width=2)
            x, y = xy[-1]
            c.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline=color)
            tip = pts[-1]
            ex, ey = view.pt(*proj(tip + 0.045 * a))
            c.create_line(x, y, ex, ey, fill=POINTER if dash is None else color, width=2, arrow="last")
            bx, by = xy[0]
            c.create_rectangle(bx - 12, by - 3, bx + 12, by + 3, fill="#8a8a8a", outline="")

    def _draw_legend(self, x: float, y: float) -> None:
        c = self.canvas
        items = ((ARM, None, 5, "current pose"), (GHOST_OK, (7, 4), 3, "planned / slider preview"),
                 (GHOST_BAD, (7, 4), 3, "failed attempt"), (WORKSPACE, (6, 4), 1, "workspace"),
                 (REACH, (2, 4), 1, "max reach"))
        for color, dash, width, label in items:
            c.create_line(x, y, x + 26, y, fill=color, dash=dash, width=width)
            t = c.create_text(x + 32, y, text=label, anchor="w", fill=INFO_FG)
            x = c.bbox(t)[2] + 18
        c.create_line(x, y - 6, x + 12, y + 6, fill=TARGET, width=2)
        c.create_line(x, y + 6, x + 12, y - 6, fill=TARGET, width=2)
        c.create_text(x + 18, y, text="target", anchor="w", fill=INFO_FG)
        x += 70
        c.create_line(x, y, x + 24, y, fill=POINTER, width=2, arrow="last")
        c.create_text(x + 30, y, text="gripper direction", anchor="w", fill=INFO_FG)

    # ======================================================================= small helpers

    def _fmt_pos(self, p_m) -> str:
        u = self.units.get()
        return "(" + ", ".join(_num(v / UNITS[u], DECIMALS[u]) for v in p_m) + f") {u}"

    @staticmethod
    def _fmt_joints(joints: dict[str, float]) -> str:
        return "   ".join(f"{n} {round(v, 1) + 0.0:.1f}°" for n, v in joints.items())

    def log(self, msg: str, tag: str = "info") -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", time.strftime("%H:%M:%S  "), "info")
        self.log_box.insert("end", msg + "\n", tag)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    @staticmethod
    def _set_text(box: ScrolledText, text: str) -> None:
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("end", text)
        box.configure(state="disabled")


def run(config_path: str | Path, port: str | None = None, robot_id: str | None = None, units: str = "cm") -> None:
    try:  # crisp text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    Dashboard(root, config_path, port, robot_id, units)
    root.mainloop()
