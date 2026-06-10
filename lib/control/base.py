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

import copy
import itertools
import threading
import time

import numpy as np

import robot_common as rc

from .state import RobotState


class Busy(Exception):
    """Raised when a motion command is submitted while another is still running."""


class Unsupported(Exception):
    """Raised for an operation the concrete robot does not support (e.g. GoFa gripper)."""


class RobotController:
    robot_name: str = "?"
    NUM_JOINTS: int = 6
    POLL_HZ: float = 30.0

    def __init__(self) -> None:
        self._lock = threading.Lock()            # guards _state
        self._state: RobotState | None = None
        self._stop_evt = threading.Event()       # shuts the state loop down (close)
        self._cmd_stop = threading.Event()       # preempts the active command (stop/estop)
        self._cmd_lock = threading.Lock()        # guards _active + command start
        self._active: dict | None = None         # {"id","kind","status","progress","error"}
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
        return None

    def wait(self, cid: int, timeout: float = 30.0) -> str:
        """Block until command `cid` reaches a terminal status; returns the status
        ("done"/"failed"/"stopped"), "timeout", or "gone" if the command is no longer
        tracked (only the most recent command is retained, so a newer command having
        started means `cid` already finished — its terminal status is no longer known)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._cmd_lock:
                a = self._active
                if a is not None and a["id"] == cid and a["status"] != "running":
                    return a["status"]
                if a is None or a["id"] > cid:
                    return "gone"
            time.sleep(0.02)
        return "timeout"

    # ---------- public commands ----------
    def move_to_joints(self, q, speed: float = 1.0) -> int:
        q_goal = np.asarray(q, dtype=float)
        q_start = self._read_q_copy()
        return self._submit("moving", lambda cb: self._run_play(
            [(q_start, q_goal, None)], speed, cb))

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
        self._graceful_stop()

    def estop(self) -> None:
        self._cmd_stop.set()
        self._hard_stop()

    def grasp_pose(self, q):
        """FK grasp/EE pose for q -> (pos, wxyz). Used by the recorder dashboard."""
        return self._fk_pose(np.asarray(q, dtype=float))

    def start_freedrive(self) -> None:
        """Enter hand-guiding (UR teachMode / GoFa lead-through). Bypasses the command
        executor, so the CALLER must ensure no motion command is active first — mixing
        free-drive with an active servoJ/EGM stream conflicts at the controller."""
        self._start_freedrive()

    def stop_freedrive(self) -> None:
        self._stop_freedrive()

    def adjust_grip(self, delta):
        """Nudge the gripper by `delta` (UR only); returns the new fraction or None."""
        return None

    # ---------- shared helpers ----------
    def _read_q_copy(self) -> np.ndarray:
        return np.asarray(self._read_q(), dtype=float).copy()

    def _load_waypoints(self, waypoints_or_name) -> list[dict]:
        if isinstance(waypoints_or_name, str):
            data = rc.load_trajectory(waypoints_or_name)
            return data.get("waypoints", [])
        return list(waypoints_or_name)

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

    # ---------- hardware primitives (subclass implements) ----------
    def _connect(self) -> None: raise NotImplementedError
    def _close(self) -> None: raise NotImplementedError
    def _read_q(self): raise NotImplementedError
    def _read_safety(self): raise NotImplementedError          # -> (safety, ctrl, conn_ok, health)
    def _fk_pose(self, q): raise NotImplementedError           # -> (pos, wxyz)
    def _ik(self, pos, wxyz, q_seed): raise NotImplementedError  # -> q
    def _run_play(self, segments, speed, progress_cb): raise NotImplementedError
    def _graceful_stop(self) -> None: raise NotImplementedError
    def _hard_stop(self) -> None: raise NotImplementedError
    def _gripper_frac(self): return None
    def _gripper_blocking(self, frac, progress_cb): raise Unsupported("no gripper")
    def _start_freedrive(self) -> None: raise NotImplementedError
    def _stop_freedrive(self) -> None: raise NotImplementedError
