"""Inverse kinematics with a convergence loop and honest error reporting.

Damped least squares (Levenberg-Marquardt style) on the chain from `chain.py`.

Targets are a position plus an optional *approach direction* (which way the
gripper points) rather than a full orientation. A 5-DOF arm like the SO-101 can
generally satisfy 3 position + 2 direction constraints, but not a full 6D pose,
so this is the right shape of constraint for it.

Every result is re-checked with forward kinematics, so `reached` and the error
numbers describe where the arm would actually end up, not what the solver hoped.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .chain import Chain


@dataclass
class IKResult:
    reached: bool
    joints_deg: dict[str, float]
    position_m: list[float]           # where FK says the tip ends up
    position_error_mm: float
    approach_error_deg: float | None  # None when no approach was requested
    iterations: int
    reason: str | None = None         # why it failed, in plain words

    def to_dict(self) -> dict:
        d = asdict(self)
        d["position_m"] = [round(v, 4) for v in self.position_m]
        d["position_error_mm"] = round(self.position_error_mm, 2)
        if self.approach_error_deg is not None:
            d["approach_error_deg"] = round(self.approach_error_deg, 2)
        d["joints_deg"] = {k: round(v, 2) for k, v in self.joints_deg.items()}
        return d


class IKSolver:
    def __init__(
        self,
        chain: Chain,
        tool_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
        position_tol_mm: float = 2.0,
        approach_tol_deg: float = 3.0,
        max_iters: int = 300,
        restarts: int = 6,
        approach_weight: float = 0.05,   # metres of position error worth 1 rad of approach error
        max_step_rad: float = 0.2,
        rng_seed: int = 0,
    ):
        self.chain = chain
        a = np.asarray(tool_axis_local, dtype=float)
        self.tool_axis_local = a / np.linalg.norm(a)
        self.position_tol = position_tol_mm / 1000.0
        self.approach_tol = math.radians(approach_tol_deg)
        self.max_iters = max_iters
        self.restarts = restarts
        self.w_app = approach_weight
        self.max_step = max_step_rad
        self.rng = np.random.default_rng(rng_seed)
        self._reach_center, self.max_reach_m = self._reach_bound()

    def _reach_bound(self) -> tuple[np.ndarray, float]:
        """Conservative sphere containing everything the tip can reach: centred on the
        base axis at the height of the second joint, radius = that joint's offset from
        the axis plus the straight-line lengths of every segment after it."""
        fk = self.chain.fk(np.zeros(len(self.chain.active)))
        pts = fk.joint_origins + [fk.T[:3, 3]]
        if len(pts) < 3:
            return np.zeros(3), math.inf
        base_axis, base_pt = fk.joint_axes[0], fk.joint_origins[0]
        p1 = pts[1]
        along = (p1 - base_pt) @ base_axis
        center = base_pt + along * base_axis
        radius = np.linalg.norm(p1 - center) + sum(
            np.linalg.norm(pts[k + 1] - pts[k]) for k in range(1, len(pts) - 1)
        )
        return center, float(radius)

    # ---- public API ---------------------------------------------------------

    def tip_pose(self, q_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(position m, approach direction unit vector) for joint angles in degrees."""
        T = self.chain.fk(np.radians(q_deg)).T
        return T[:3, 3].copy(), T[:3, :3] @ self.tool_axis_local

    def solve(
        self,
        target_xyz: np.ndarray,
        seed_deg: np.ndarray,
        approach_dir: np.ndarray | None = None,
    ) -> IKResult:
        target = np.asarray(target_xyz, dtype=float)
        a_des = None
        if approach_dir is not None:
            a_des = np.asarray(approach_dir, dtype=float)
            a_des = a_des / np.linalg.norm(a_des)

        seed = self.chain.clip(np.radians(np.asarray(seed_deg, dtype=float)))

        dist = float(np.linalg.norm(target - self._reach_center))
        if dist > self.max_reach_m * 1.01:
            p_now = self.chain.fk(seed).T[:3, 3]
            return IKResult(
                reached=False,
                joints_deg=dict(zip(self.chain.joint_names, np.degrees(seed).tolist())),
                position_m=p_now.tolist(),
                position_error_mm=float(np.linalg.norm(target - p_now)) * 1000.0,
                approach_error_deg=None,
                iterations=0,
                reason=(
                    f"Target is out of reach: it is {dist * 100:.1f} cm from the shoulder, and the arm "
                    f"reaches at most about {self.max_reach_m * 100:.1f} cm. Choose a closer point."
                ),
            )

        candidates = [seed] + self._structured_seeds(target, seed) + [
            self.rng.uniform(self._finite(self.chain.lower, -math.pi), self._finite(self.chain.upper, math.pi))
            for _ in range(self.restarts)
        ]

        best = None
        total_iters = 0
        for q0 in candidates:
            q, it = self._dls(q0, target, a_des)
            total_iters += it
            pe, ae = self._errors(q, target, a_des)
            ok = pe <= self.position_tol and (ae is None or ae <= self.approach_tol)
            # rank: success first, then error, then how far the joints move from the seed
            key = (not ok, pe + (0 if ae is None else self.w_app * ae), np.abs(q - seed).sum())
            if best is None or key < best[0]:
                best = (key, q, pe, ae, ok)
            if ok and q0 is seed:
                break  # converged from the current pose: smoothest possible move

        _, q, pe, ae, ok = best
        reason = None
        if not ok:
            if a_des is not None and pe > self.position_tol:
                # Was it the position or the approach direction that made this impossible?
                pos_only = self.solve(target, seed_deg, approach_dir=None)
                if pos_only.reached:
                    reason = (
                        "The position is reachable, but not with the requested approach direction "
                        f"(best compromise was {pe * 1000:.1f} mm and {math.degrees(ae):.1f} deg off). "
                        "On this 5-joint arm the gripper can only point within the vertical plane "
                        "through the base and the target. Retry with approach='any' or another direction."
                    )
            if reason is None and pe > self.position_tol:
                reason = (
                    f"Target is not reachable: the closest the arm can get is {pe * 1000:.1f} mm away. "
                    "It is probably outside the arm's reach or blocked by joint limits."
                )
            elif reason is None:
                reason = (
                    f"The position is reachable but the approach direction is off by "
                    f"{math.degrees(ae):.1f} deg. Retry with approach='any' or another direction."
                )
        p_final = self.chain.fk(q).T[:3, 3]
        return IKResult(
            reached=ok,
            joints_deg=dict(zip(self.chain.joint_names, np.degrees(q).tolist())),
            position_m=p_final.tolist(),
            position_error_mm=pe * 1000.0,
            approach_error_deg=None if ae is None else math.degrees(ae),
            iterations=total_iters,
            reason=reason,
        )

    # ---- internals ----------------------------------------------------------

    def _structured_seeds(self, target: np.ndarray, seed: np.ndarray) -> list[np.ndarray]:
        """Seeds with the base already facing the target (and facing away, for reaching
        backwards), combined with elbow-up / elbow-down shapes. Assumes the first
        active joint is the base yaw, which holds for the SO-101."""
        pan = math.atan2(target[1], target[0])
        seeds = []
        for p in (pan, pan + math.pi, pan - math.pi):
            for lift, elbow in ((0.0, 0.0), (0.6, -0.6), (-0.6, 0.6), (0.9, 0.9), (-0.9, -0.9)):
                q = seed.copy()
                q[0] = p
                if len(q) > 2:
                    q[1], q[2] = lift, elbow
                q = self.chain.clip(q)
                if not any(np.allclose(q, s) for s in seeds):
                    seeds.append(q)
        return seeds

    @staticmethod
    def _finite(v: np.ndarray, fallback: float) -> np.ndarray:
        return np.where(np.isfinite(v), v, fallback)

    def _errors(self, q, target, a_des):
        T = self.chain.fk(q).T
        pe = float(np.linalg.norm(target - T[:3, 3]))
        if a_des is None:
            return pe, None
        a = T[:3, :3] @ self.tool_axis_local
        ae = float(math.acos(np.clip(a @ a_des, -1.0, 1.0)))
        return pe, ae

    def _dls(self, q, target, a_des):
        lam = 1e-2
        prev_cost = math.inf
        best_cost, since_best = math.inf, 0
        for it in range(1, self.max_iters + 1):
            fk = self.chain.fk(q)
            p = fk.T[:3, 3]
            Jv, Jw = self.chain.jacobian(fk)
            e = target - p
            J = Jv
            if a_des is not None:
                a = fk.T[:3, :3] @ self.tool_axis_local
                # d(a)/dq_i = w_i x a  for revolute joints
                Ja = np.cross(Jw.T, a).T
                e = np.concatenate([e, self.w_app * (a_des - a)])
                J = np.vstack([Jv, self.w_app * Ja])

            cost = float(e @ e)
            if np.linalg.norm(e[:3]) <= self.position_tol * 0.5 and (
                a_des is None or np.linalg.norm(e[3:]) / self.w_app <= self.approach_tol * 0.5
            ):
                return q, it

            # give up on this start if it has stopped improving (unreachable / local minimum)
            if cost < best_cost * (1 - 1e-6):
                best_cost, since_best = cost, 0
            else:
                since_best += 1
                if since_best >= 25:
                    return q, it

            # adapt damping: shrink when improving, grow when stuck (helps near singularities)
            lam = max(lam * 0.7, 1e-4) if cost < prev_cost else min(lam * 2.0, 1.0)
            prev_cost = cost

            JJt = J @ J.T
            A = JJt + (lam ** 2) * np.eye(JJt.shape[0])
            dq = J.T @ np.linalg.lstsq(A, e, rcond=None)[0]
            step = np.max(np.abs(dq))
            if step > self.max_step:
                dq *= self.max_step / step
            q_new = self.chain.clip(q + dq)
            if np.max(np.abs(q_new - q)) < 1e-7:
                return q_new, it  # stalled (likely against a limit)
            q = q_new
        return q, self.max_iters
