"""Offline simulator backend for the teleop scaffold.

install() injects fake robot-transport modules (rtde_control, rtde_receive,
hande_gripper, abb_rws, abb_egm) into sys.modules so the REAL teleop scripts run
unmodified against a perfect-tracking kinematic sim — no UR15, GoFa, or network.
Every reusable primitive (IK, trapezoidal profile, viser UI, TCP-speed cap,
gripper viz) runs exactly as on hardware; only the transport is faked.

Spec: docs/superpowers/specs/2026-06-08-offline-simulator-design.md
"""
from __future__ import annotations

import sys
import threading
import time
import types

NUM_JOINTS = 6
COND_TIME = 1.0   # mirror PyEgm.mod \CondTime := 1 (s of steady target -> EGMRunJoint exits)

# Sane non-singular starting configs (rad). Perfect tracking means these only set
# the joints before the first command; tune freely.
UR_HOME = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
GOFA_HOME = [0.0, 0.0, 0.0, 0.0, 1.5708, 0.0]
NEUTRAL_HOME = [0.0, -1.0, 1.0, 0.0, 1.0, 0.0]


class SimWorld:
    """Shared simulated robot state. One joint vector is the single source of truth."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.q = list(UR_HOME)
        self.flags = {"egm_go": False, "lead_go": False}
        self.grip_frac = 0.0
        # EGM supervisor bookkeeping (populated by FakeEGM's supervisor thread)
        self.egm_target = None
        self.egm_last_target = None
        self.egm_last_change = 0.0
        self.egm_feedback = None
        self.egm_feedback_time = 0.0
        self.packets_rx = 0
        self.packets_tx = 0

    def set_home(self, q) -> None:
        with self.lock:
            self.q = list(q)


SIM = SimWorld()


# ---------- UR fakes (rtde_control / rtde_receive) ----------
class FakeRTDEControl:
    def __init__(self, *a, **kw) -> None:
        self._last_dt = 1.0 / 125

    def servoJ(self, q, a, v, dt, lookahead, gain) -> None:
        if dt and dt > 0:
            self._last_dt = dt
        with SIM.lock:
            SIM.q = list(q)

    def initPeriod(self) -> float:
        return time.monotonic()

    def waitPeriod(self, t_start) -> None:
        target = t_start + self._last_dt
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)

    def setPayload(self, *a, **kw) -> None: pass
    def teachMode(self, *a, **kw) -> None: pass
    def endTeachMode(self, *a, **kw) -> None: pass
    def servoStop(self, *a, **kw) -> None: pass
    def stopScript(self, *a, **kw) -> None: pass
    def triggerProtectiveStop(self, *a, **kw) -> None: pass
    def disconnect(self, *a, **kw) -> None: pass


class FakeRTDEReceive:
    def __init__(self, *a, **kw) -> None: pass

    def getActualQ(self):
        with SIM.lock:
            return list(SIM.q)

    def getSafetyMode(self) -> int:
        return 1  # NORMAL

    def disconnect(self, *a, **kw) -> None: pass


# ---------- Hand-E fake (hande_gripper) ----------
DEFAULT_PORT = 63352


class FakeHandE:
    def __init__(self, *a, **kw) -> None: pass
    def connect(self) -> None: pass
    def close(self) -> None: pass
    def activate(self, timeout: float = 5.0) -> None: pass
    def reset(self, timeout: float = 5.0) -> None: pass

    def open(self, *a, **kw) -> None:
        with SIM.lock:
            SIM.grip_frac = 0.0

    def close_gripper(self, *a, **kw) -> None:
        with SIM.lock:
            SIM.grip_frac = 1.0

    def move(self, fraction, *a, **kw) -> None:
        with SIM.lock:
            SIM.grip_frac = max(0.0, min(1.0, float(fraction)))

    def wait_until_idle(self, timeout: float = 10.0) -> None: pass

    def status(self):
        with SIM.lock:
            return {"activated": True, "object": False, "fault": 0,
                    "position": int(SIM.grip_frac * 255)}


# ---------- install ----------
def _module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def install(robot_hint: str | None = None) -> None:
    """Inject fake transport modules into sys.modules and seed the home pose.
    Call BEFORE the target script imports any transport module."""
    sys.modules["rtde_control"] = _module("rtde_control", RTDEControlInterface=FakeRTDEControl)
    sys.modules["rtde_receive"] = _module("rtde_receive", RTDEReceiveInterface=FakeRTDEReceive)
    sys.modules["hande_gripper"] = _module(
        "hande_gripper", HandEGripper=FakeHandE, DEFAULT_PORT=DEFAULT_PORT
    )
    home = {"ur15": UR_HOME, "gofa": GOFA_HOME}.get(robot_hint, NEUTRAL_HOME)
    SIM.set_home(home)
