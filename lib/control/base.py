"""RobotController base: hardware-agnostic state polling + async command executor.

Subclasses (ur.py, gofa.py) implement the hardware primitives:
  _connect/_close, _read_q, _read_safety, _fk_pose, _ik, _run_play,
  _graceful_stop, _hard_stop, _gripper_frac, _gripper_blocking.

Commands are async: a submit returns a command id immediately and the motion runs
on a worker thread. One motion at a time — a submit while busy raises Busy.
stop()/estop() preempt via the _cmd_stop event. A state-poll thread keeps the
latest RobotState fresh for get_state() and (later) telemetry.
"""
from __future__ import annotations

import collections
import copy
import itertools
import math
import threading
import time

import numpy as np

import robot_common as rc

from .state import RobotState


class Busy(Exception):
    """Raised when a motion command is submitted while another is still running."""


class Unsupported(Exception):
    """Raised for an operation the concrete robot does not support (e.g. GoFa gripper)."""


# ---------- straight-line (Cartesian) tool interpolation ----------
# Joint-space lerp sweeps the tool along an arc; these move the *tool pose* in a
# straight line (MoveL) and IK back to joints. numpy-only (no jax) so the control
# core stays lightweight — the fk/ik callables carry the jax cost where they live.

def slerp_wxyz(w0, w1, s: float) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions (wxyz), shorter arc."""
    w0 = np.asarray(w0, dtype=float)
    w1 = np.asarray(w1, dtype=float)
    dot = float(np.dot(w0, w1))
    if dot < 0.0:                      # q and -q are the same rotation; take the short way
        w1, dot = -w1, -dot
    if dot > 0.9995:                   # almost parallel: lerp + renormalize (sin(theta)->0)
        q = w0 + s * (w1 - w0)
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    return (math.sin((1.0 - s) * theta) * w0 + math.sin(s * theta) * w1) / math.sin(theta)


def cartesian_q(fk, ik, q_start, q_goal):
    """Build at(s), s in [0,1], whose TOOL pose moves in a straight line — position
    lerp + orientation slerp — from fk(q_start) to fk(q_goal), solving seeded IK at
    each sample. The joint-space lerp seeds the IK (keeps one kinematic branch); the
    endpoints return q_start / q_goal exactly, so captured configs are hit precisely.

    fk(q) -> (pos, wxyz); ik(pos, wxyz, q_seed) -> q. Pure in s (no hidden state)."""
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    p0, w0 = (np.asarray(v, dtype=float) for v in fk(q_start))
    p1, w1 = (np.asarray(v, dtype=float) for v in fk(q_goal))

    def at(s: float) -> np.ndarray:
        if s <= 0.0:
            return q_start.copy()
        if s >= 1.0:
            return q_goal.copy()
        pos = (1.0 - s) * p0 + s * p1
        wxyz = slerp_wxyz(w0, w1, s)
        seed = (1.0 - s) * q_start + s * q_goal
        return np.asarray(ik(pos, wxyz, seed), dtype=float)
    return at


def step_pose_toward(p_cur, w_cur, p_tgt, w_tgt, max_lin: float, max_ang: float):
    """One bounded step of a tool pose from (p_cur, w_cur) toward (p_tgt, w_tgt):
    translation clamped to max_lin (m), reorientation to max_ang (rad). Returns
    (p_ref, w_ref) — IK it to chase a moving gizmo along a straight tool path."""
    p_cur = np.asarray(p_cur, dtype=float)
    p_tgt = np.asarray(p_tgt, dtype=float)
    w_cur = np.asarray(w_cur, dtype=float)
    w_tgt = np.asarray(w_tgt, dtype=float)
    dp = p_tgt - p_cur
    dist = float(np.linalg.norm(dp))
    p_ref = p_tgt if dist <= max_lin else p_cur + dp * (max_lin / dist)
    dot = min(1.0, abs(float(np.dot(w_cur, w_tgt))))
    angle = 2.0 * math.acos(dot)        # geodesic angle between the two orientations
    w_ref = w_tgt.copy() if angle <= max_ang else slerp_wxyz(w_cur, w_tgt, max_ang / angle)
    return p_ref, w_ref


class RobotController:
    robot_name: str = "?"
    NUM_JOINTS: int = 6
    POLL_HZ: float = 30.0
    _CMD_HISTORY_MAX: int = 64

    def __init__(self) -> None:
        self._lock = threading.Lock()            # guards _state
        self._state: RobotState | None = None
        self._stop_evt = threading.Event()       # shuts the state loop down (close)
        self._cmd_stop = threading.Event()       # preempts the active command (stop/estop)
        self._cmd_lock = threading.Lock()        # guards _active + command start + _freedrive
        self._active: dict | None = None         # {"id","kind","status","progress","error"}
        self._freedrive = False                  # hand-guiding active (mutually exclusive w/ commands)
        self._cmd_history: "collections.OrderedDict[int, dict]" = collections.OrderedDict()
        self._cmd_counter = itertools.count(1)
        self._state_thread: threading.Thread | None = None
        self._cmd_thread: threading.Thread | None = None

    # ---------- lifecycle ----------
    def connect(self) -> None:
        self._connect()
        self._poll_once()                        # seed _state before the loop starts
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._state_thread.start()

    def close(self) -> None:
        """Shut down: signal stop, join the active command worker (preempted by
        _cmd_stop) and the state thread, then _close() to tear down hardware."""
        self._cmd_stop.set()
        self._stop_evt.set()
        if self._cmd_thread is not None:
            self._cmd_thread.join(timeout=2.0)
        if self._state_thread is not None:
            self._state_thread.join(timeout=1.0)
        self._close()

    @property
    def closed(self) -> bool:
        """True once close() has been called (the shutdown event is set)."""
        return self._stop_evt.is_set()

    # ---------- state ----------
    def _state_loop(self) -> None:
        period = 1.0 / self.POLL_HZ
        while not self._stop_evt.is_set():
            self._poll_once()
            time.sleep(period)

    def _poll_once(self) -> None:
        try:
            q = np.asarray(self._read_q(), dtype=float)
            safety, ctrl, conn_ok, health = self._read_safety()
            pos, wxyz = self._fk_pose(q)
        except Exception:
            with self._lock:
                if self._state is not None:
                    self._state.conn_ok = False
            return
        with self._cmd_lock:
            active = copy.deepcopy(self._active) if self._active else None
        st = RobotState(
            ts=time.monotonic(), robot=self.robot_name, q=q.tolist(),
            pose={"pos": [float(v) for v in pos], "wxyz": [float(v) for v in wxyz]},
            gripper_frac=self._gripper_frac(), safety_state=safety,
            controller_state=ctrl, activity=self._activity(active),
            active_command=active, conn_ok=conn_ok, health=health,
        )
        with self._lock:
            self._state = st

    def _activity(self, active: dict | None) -> str:
        # _cmd_stop is read outside _cmd_lock, so activity and active_command can be
        # transiently inconsistent by one poll cycle at a stop/start boundary.
        if active is not None and active["status"] == "running":
            return active["kind"]
        if self._freedrive:
            return "freedrive"
        if self._cmd_stop.is_set():
            return "stopped"
        return "idle"

    def get_state(self) -> RobotState:
        with self._lock:
            if self._state is None:
                raise RuntimeError("controller not connected (no state yet)")
            return copy.deepcopy(self._state)

    # ---------- command executor ----------
    def _submit(self, kind: str, run) -> int:
        """Start a motion if free. `run` is a callable(progress_cb) doing the motion.
        Returns the command id; raises Busy if a motion is already running."""
        with self._cmd_lock:
            if self._freedrive:
                raise Busy("busy with free-drive (stop free-drive before commanding motion)")
            if self._active is not None and self._active["status"] == "running":
                raise Busy(f"busy with command {self._active['id']}")
            self._cmd_stop.clear()   # clear any stale stop atomically with claiming the command
            cid = next(self._cmd_counter)
            self._active = {"id": cid, "kind": kind, "status": "running",
                            "progress": 0.0, "error": None}
        self._cmd_thread = threading.Thread(target=self._run_cmd, args=(cid, run), daemon=True)
        self._cmd_thread.start()
        return cid

    def _run_cmd(self, cid: int, run) -> None:
        try:
            run(self._progress_cb(cid))
            # Edge: if stop() arrives after run() returns, we report "stopped" even
            # though the motion completed — a deliberate trade-off (a finer-grained
            # protocol would be needed to distinguish the two).
            status, err = ("stopped" if self._cmd_stop.is_set() else "done"), None
        except Exception as e:                   # noqa: BLE001 - report any failure as the command result
            status, err = "failed", str(e)
        with self._cmd_lock:
            if self._active is not None and self._active["id"] == cid:
                self._active["status"] = status
                self._active["error"] = err
                if status == "done":
                    self._active["progress"] = 1.0
                self._cmd_history[cid] = dict(self._active)
                while len(self._cmd_history) > self._CMD_HISTORY_MAX:
                    self._cmd_history.popitem(last=False)

    def _progress_cb(self, cid: int):
        def cb(frac: float) -> None:
            with self._cmd_lock:
                if self._active is not None and self._active["id"] == cid:
                    self._active["progress"] = float(frac)
        return cb

    def command_status(self, cid: int) -> dict | None:
        with self._cmd_lock:
            if self._active is not None and self._active["id"] == cid:
                return dict(self._active)
            if cid in self._cmd_history:
                return dict(self._cmd_history[cid])
        return None

    def wait(self, cid: int, timeout: float = 30.0) -> str:
        """Block until command `cid` reaches a terminal status; returns the status
        ("done"/"failed"/"stopped"), "timeout", or "gone" if `cid` was never issued or
        has been evicted from the bounded command history (_CMD_HISTORY_MAX deep)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._cmd_lock:
                a = self._active
                if a is not None and a["id"] == cid and a["status"] != "running":
                    return a["status"]
                if cid in self._cmd_history:
                    return self._cmd_history[cid]["status"]
                if a is None or a["id"] > cid:
                    return "gone"
            time.sleep(0.02)
        return "timeout"

    # ---------- public commands ----------
    def move_to_joints(self, q, speed: float = 1.0) -> int:
        q_goal = np.asarray(q, dtype=float)
        q_start = self._read_q_copy()
        # a joint target is a MoveJ: interpolate in joint space (exact config, no IK)
        return self._submit("moving", lambda cb: self._run_play(
            [(q_start, q_goal, None)], speed, cb, interp="joint"))

    def move_to_pose(self, pos, wxyz, speed: float = 1.0) -> int:
        q_start = self._read_q_copy()
        q_goal = self._ik(np.asarray(pos, dtype=float), np.asarray(wxyz, dtype=float), q_start)
        return self._submit("moving", lambda cb: self._run_play(
            [(q_start, q_goal, None)], speed, cb))

    def play(self, waypoints_or_name, speed: float = 1.0) -> int:
        wps = self._load_waypoints(waypoints_or_name)
        segs = self._build_segments(wps)
        return self._submit("playing", lambda cb: self._run_play(segs, speed, cb))

    def set_gripper(self, frac: float) -> int:
        return self._submit("moving", lambda cb: self._gripper_blocking(float(frac), cb))

    def stop(self) -> None:
        self._cmd_stop.set()
        if self._freedrive:
            self.stop_freedrive()    # release a compliant arm too
        self._graceful_stop()

    def estop(self) -> None:
        self._cmd_stop.set()
        if self._freedrive:
            self.stop_freedrive()
        self._hard_stop()

    def grasp_pose(self, q):
        """FK grasp/EE pose for q -> (pos, wxyz). Used by the recorder dashboard."""
        return self._fk_pose(np.asarray(q, dtype=float))

    def start_freedrive(self) -> None:
        """Enter hand-guiding (UR teachMode / GoFa lead-through). Mutually exclusive with
        the command executor: raises Busy if a motion is running, and while free-drive is
        on _submit refuses, so a servoJ/EGM stream can't fight the compliant arm."""
        with self._cmd_lock:
            if self._active is not None and self._active["status"] == "running":
                raise Busy(f"busy with command {self._active['id']}")
            self._freedrive = True
        try:
            self._start_freedrive()
        except Exception:
            self._freedrive = False   # hardware refused — don't leave the flag stuck on
            raise

    def stop_freedrive(self) -> None:
        self._freedrive = False
        self._stop_freedrive()

    def adjust_grip(self, delta):
        """Nudge the gripper by `delta` (UR only); returns the new fraction or None."""
        return None

    # ---------- shared helpers ----------
    def _read_q_copy(self) -> np.ndarray:
        return np.asarray(self._read_q(), dtype=float).copy()

    def _load_waypoints(self, waypoints_or_name) -> list[dict]:
        """Resolve a play target into a waypoint list. Accepts a trajectory name (str),
        a list of names (chained — each trajectory's waypoints concatenated, in order,
        as one continuous motion), or a list of inline waypoint dicts."""
        if isinstance(waypoints_or_name, str):
            return self._named_waypoints(waypoints_or_name)
        items = list(waypoints_or_name)
        if items and all(isinstance(x, str) for x in items):
            return [wp for nm in items for wp in self._named_waypoints(nm)]
        return items

    def _named_waypoints(self, name: str) -> list[dict]:
        return rc.load_trajectory(name, self.robot_name).get("waypoints", [])

    def _build_segments(self, waypoints: list[dict]):
        """[(q_start, q_goal, grip)] from the current pose through each waypoint.
        A waypoint with 'q' replays those joints; without it, IK from the Cartesian
        pose (sequential seed). Same logic as play_trajectory.build_segments."""
        q = self._read_q_copy()
        segs = []
        for wp in waypoints:
            if wp.get("q") is not None:
                q_next = np.asarray(wp["q"], dtype=float)
            else:
                q_next = self._ik(np.asarray(wp["pos"], dtype=float),
                                  np.asarray(wp["wxyz"], dtype=float), q)
            segs.append((q.copy(), q_next, rc.norm_grip(wp.get("grip"))))
            q = q_next
        return segs

    def _cartesian_q(self, q_start, q_goal):
        """Straight-line tool interpolation (base.cartesian_q) over this robot's FK/IK."""
        return cartesian_q(self._fk_pose, self._ik, q_start, q_goal)

    # ---------- hardware primitives (subclass implements) ----------
    def _connect(self) -> None: raise NotImplementedError
    def _close(self) -> None: raise NotImplementedError
    def _read_q(self): raise NotImplementedError
    def _read_safety(self): raise NotImplementedError          # -> (safety, ctrl, conn_ok, health)
    def _fk_pose(self, q): raise NotImplementedError           # -> (pos, wxyz)
    def _ik(self, pos, wxyz, q_seed): raise NotImplementedError  # -> q
    def _run_play(self, segments, speed, progress_cb, interp="cartesian"): raise NotImplementedError
    def _graceful_stop(self) -> None: raise NotImplementedError
    def _hard_stop(self) -> None: raise NotImplementedError
    def _gripper_frac(self): return None
    def _gripper_blocking(self, frac, progress_cb): raise Unsupported("no gripper")
    def _start_freedrive(self) -> None: raise NotImplementedError
    def _stop_freedrive(self) -> None: raise NotImplementedError
