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
        self.reset(UR_HOME)

    def reset(self, home) -> None:
        """Reset to a clean per-launch state: home pose, cleared RAPID flags and EGM
        bookkeeping. install() calls this so each sim launch starts hermetic (no stale
        egm_go / feedback carried over from a prior install() in the same process)."""
        with self.lock:
            self.q = list(home)
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


# ---------- GoFa RWS fake (abb_rws) ----------
class FakeRWS:
    def __init__(self, *a, **kw) -> None: pass
    def request_mastership(self) -> None: pass
    def release_mastership(self) -> None: pass
    def set_motors_on(self) -> None: pass
    def reset_pp(self, *a, **kw) -> None: pass
    def unload_module(self, *a, **kw) -> None: pass
    def start_program(self) -> None: pass
    def stop_program(self) -> None: pass
    def get_operation_mode(self) -> str: return "AUTO"
    def get_execution_state(self) -> str: return "running"
    def get_controller_state(self) -> str: return "motoron"

    def get_joints(self, mechunit: str = "ROB_1"):
        with SIM.lock:
            return list(SIM.q)

    def set_rapid_bool(self, var, value, task: str = "T_ROB1", module: str = "PyEgm") -> None:
        with SIM.lock:
            was = SIM.flags.get(var, False)
            SIM.flags[var] = bool(value)
            # egm_go rising edge: (re)arm the convergence clock so CondTime doesn't
            # fire before any target has streamed.
            if var == "egm_go" and bool(value) and not was:
                SIM.egm_last_change = time.monotonic()
                SIM.egm_last_target = None

    def get_rapid_data(self, var, task: str = "T_ROB1", module: str = "PyEgm") -> str:
        with SIM.lock:
            return "TRUE" if SIM.flags.get(var, False) else "FALSE"


# ---------- GoFa EGM fake + fake RAPID supervisor (abb_egm) ----------
class FakeEGM:
    def __init__(self, local_port: int = 6510, *a, **kw) -> None:
        self.local_port = local_port
        self._stop = threading.Event()
        self._thread = None

    @property
    def packets_rx(self) -> int:
        with SIM.lock:
            return SIM.packets_rx

    @property
    def packets_tx(self) -> int:
        with SIM.lock:
            return SIM.packets_tx

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._supervise, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_target_rad(self, joints_rad) -> None:
        assert len(joints_rad) == 6
        with SIM.lock:
            SIM.egm_target = list(joints_rad)

    def get_feedback_rad(self):
        with SIM.lock:
            return list(SIM.egm_feedback) if SIM.egm_feedback is not None else None

    def has_feedback(self) -> bool:
        with SIM.lock:
            return SIM.egm_feedback is not None

    def is_fresh(self, max_age_s: float = 0.2) -> bool:
        with SIM.lock:
            return (SIM.egm_feedback is not None
                    and (time.time() - SIM.egm_feedback_time) < max_age_s)

    def stats(self):
        with SIM.lock:
            return {
                "rx": SIM.packets_rx, "tx": SIM.packets_tx,
                "age_s": (time.time() - SIM.egm_feedback_time) if SIM.egm_feedback else -1.0,
                "remote": "(sim)",
            }

    def _supervise(self) -> None:
        """Mimic PyEgm.mod's EGMRunJoint: while egm_go, mark feedback fresh and
        apply the streamed target to the joints; when the target holds steady for
        COND_TIME, clear egm_go (the controller-side convergence-and-exit)."""
        while not self._stop.is_set():
            now = time.monotonic()
            with SIM.lock:
                if SIM.flags.get("egm_go", False):
                    SIM.packets_rx += 1
                    SIM.packets_tx += 1
                    SIM.egm_feedback = list(SIM.q)
                    SIM.egm_feedback_time = time.time()
                    tgt = SIM.egm_target
                    if tgt is not None:
                        if tgt != SIM.egm_last_target:
                            SIM.egm_last_target = list(tgt)
                            SIM.egm_last_change = now
                        SIM.q = list(tgt)
                        if now - SIM.egm_last_change >= COND_TIME:
                            SIM.flags["egm_go"] = False
                # egm_go FALSE: do NOT refresh feedback_time, so is_fresh() goes
                # stale (matches the controller halting its stream after exit).
                # lead_go: no-op (arm "compliant"; joints unchanged).
            time.sleep(0.005)


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
    sys.modules["abb_rws"] = _module("abb_rws", RWSClient=FakeRWS)
    sys.modules["abb_egm"] = _module("abb_egm", EGMSession=FakeEGM)
    home = {"ur15": UR_HOME, "gofa": GOFA_HOME}.get(robot_hint, NEUTRAL_HOME)
    SIM.reset(home)
