# Remote API — Phase 1: RobotController core + headless migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the inlined motion logic from the teleop scripts into one reusable, thread-safe `RobotController` core (UR + GoFa), then migrate the two headless consumers (`play_trajectory.py`, `teleop.py`) onto it — all verified offline against the sim fakes.

**Architecture:** A `lib/control/` package: `RobotController` base (state-poll thread + async command executor + segment building) with `URController` / `GoFaController` subclasses implementing the hardware streaming loops (lifted from the existing tuned code). Commands are async (return a command id; one motion at a time). The controller sits on the same hardware clients the sim fakes shadow, so the whole thing runs under `sim.py`.

**Tech Stack:** Python 3.13; numpy + pyroki/jax/jaxlie/yourdfpy (IK/FK, same as teleop); `robot_common` for config + profile + trajectory I/O. Tests are a stdlib-`assert` smoke script run with the venv python (this project has no pytest), driven against `lib/robot_sim.py`'s fakes via `robot_sim.install()`.

**Spec:** `docs/superpowers/specs/2026-06-09-remote-control-api-design.md`

**Scope of THIS plan:** the controller core + the two headless migrations. The two viser teleop migrations (`teleop_ur15.py`, `teleop_gofa_egm.py`) and the remote API (FastAPI) are SEPARATE later plans. Lease/watchdog/auth are API-level and are NOT in this plan.

---

## Background the implementer needs

- **Run offline / test:** the controller is exercised against the sim fakes. A test does `robot_sim.install("ur15")` (or `"gofa"`) BEFORE constructing the controller, so the controller's hardware clients are the fakes. Perfect tracking: a `_servo(q)` write makes the next `_read_q()` return `q`.
- **Imports:** `lib/` is on `sys.path` via the scripts bootstrap. The smoke test does the same bootstrap. Inside the `lib/control/` package, modules import each other relatively (`from .state import RobotState`) and import top-level libs by bare name (`import robot_common as rc`, `import pyroki as pk`, …).
- **The existing motion code being lifted** lives in `scripts/teleop_ur15.py` (`_play` at line 524, `grasp_pose`/`_grasp_to_tool0` at 208/213, connect at 128, gripper at 484), `scripts/teleop_gofa_egm.py` (`_play` at 473, `_start_egm_session` at 417, `_wait_egm_clear` at 444, `_cap_seg_duration` at 202, connect at 94/125), and `scripts/play_trajectory.py` (`play_ur15` at 147, `play_gofa` at 243, `build_segments` at 70). Read these before extracting.
- **Config** comes from `robot_common` (bind to locals as the scripts do): `UR_*`, `GOFA_*`, `RAMP_FRAC`, `MIN_SEG_DURATION_S`, `DWELL_S`, `GRIP_PREDELAY_S`, `GRIP_EPS`, `alpha_to_s`, `norm_grip`, `load_trajectory`, `make_mesh_resolver`, `TARGET_LINK`.
- **Extraction discipline:** when a step says "lift from X:lines", READ that source and move the logic faithfully, applying only the listed substitutions (e.g. `gui_status.value = …` → `progress_cb`/drop; `rtde_c.servoJ(...)` → unchanged, it's `self._c.servoJ`). Do NOT re-tune constants or re-derive the loop from scratch — these loops are hardware-tuned. Preserve behavior exactly.

## File structure (this plan)

| File | Responsibility |
|---|---|
| `lib/control/__init__.py` (create) | `make_controller(robot)` factory + re-exports |
| `lib/control/state.py` (create) | `RobotState` dataclass (stdlib, JSON-serializable) |
| `lib/control/base.py` (create) | `RobotController` ABC: state loop, async command executor, segment building, `move_to_joints`/`move_to_pose`/`play`/`set_gripper`/`stop`/`estop`/`get_state`/`wait` |
| `lib/control/ur.py` (create) | `URController` — RTDE servoJ + Hand-E + IK/FK (lifts UR loops) |
| `lib/control/gofa.py` (create) | `GoFaController` — EGM + RWS + IK/FK + the EGM handshake (lifts GoFa loops) |
| `scripts/control_smoketest.py` (create) | stdlib-assert smoke test for the controller against the fakes |
| `scripts/play_trajectory.py` (modify) | migrate `play_ur15`/`play_gofa` to `make_controller(...).play(...)` |
| `scripts/teleop.py` (modify) | migrate `URBackend`/`GoFaBackend` reads + freedrive onto the controller |

---

## Task 1: `RobotState` dataclass + smoke-test scaffold

**Files:**
- Create: `lib/control/__init__.py` (empty for now — makes `control` a package)
- Create: `lib/control/state.py`
- Create: `scripts/control_smoketest.py`

- [ ] **Step 1: Write the failing test**

Create `scripts/control_smoketest.py`:

```python
"""Offline smoke test for the RobotController core (lib/control), driven against
the sim fakes (lib/robot_sim). No robot, no network.

  ./robot_control/bin/python scripts/control_smoketest.py

Exits 0 on success, 1 on the first failed assertion.
"""
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import robot_sim  # noqa: E402
from control.state import RobotState  # noqa: E402


def test_state_dataclass():
    s = RobotState(
        ts=1.0, robot="ur15", q=[0.0] * 6,
        pose={"pos": [0.1, 0.2, 0.3], "wxyz": [1.0, 0.0, 0.0, 0.0]},
        gripper_frac=0.0, safety_state="NORMAL", controller_state="ok",
        activity="idle", active_command=None, conn_ok=True,
    )
    d = s.to_dict()
    assert d["robot"] == "ur15"
    assert d["q"] == [0.0] * 6
    assert d["pose"]["pos"] == [0.1, 0.2, 0.3]
    assert d["active_command"] is None
    assert d["health"] == {}
    print("PASS test_state_dataclass")


def main():
    test_state_dataclass()
    print("ALL CONTROL SMOKE TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
```

- [ ] **Step 2: Run it, verify it FAILS**

Run: `./robot_control/bin/python scripts/control_smoketest.py`
Expected: `ModuleNotFoundError: No module named 'control'` (or `control.state`).

- [ ] **Step 3: Implement**

Create `lib/control/__init__.py` as an empty file (one line is fine):

```python
"""RobotController core: one motion implementation behind the teleop scripts and the API."""
```

Create `lib/control/state.py`:

```python
"""RobotState: a JSON-serializable snapshot of the robot, produced by the
controller's state-poll thread and consumed by every surface (viser, API)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RobotState:
    ts: float                       # monotonic timestamp of the snapshot
    robot: str                      # "ur15" | "gofa"
    q: list                         # 6 joint angles (rad)
    pose: dict                      # {"pos": [x,y,z], "wxyz": [w,x,y,z]} grasp/EE pose
    gripper_frac: float | None      # 0=open..1=closed; None if no gripper
    safety_state: str               # robot-reported safety state
    controller_state: str           # robot-reported controller/exec state
    activity: str                   # "idle"|"moving"|"playing"|"stopped"
    active_command: dict | None     # {"id","kind","status","progress","error"} or None
    conn_ok: bool                   # last hardware read succeeded
    health: dict = field(default_factory=dict)   # transport-specific extras

    def to_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Run it, verify it PASSES**

Run: `./robot_control/bin/python scripts/control_smoketest.py`
Expected: `PASS test_state_dataclass`, `ALL CONTROL SMOKE TESTS PASSED`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add lib/control/__init__.py lib/control/state.py scripts/control_smoketest.py
git commit -m "feat(control): RobotState dataclass + control smoke-test scaffold"
```

---

## Task 2: `RobotController` base — state loop + command executor + segment building

**Files:**
- Create: `lib/control/base.py`

This task is the framework only (no hardware). It is exercised end-to-end starting in Task 3 (URController), so there is no standalone runtime test here — the verification is that it imports cleanly and Task 3's `test_ur_connect_state` passes against it. Build it carefully and completely.

- [ ] **Step 1: Implement `lib/control/base.py`**

Create `lib/control/base.py` with EXACTLY this content:

```python
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

    # ---------- lifecycle ----------
    def connect(self) -> None:
        self._connect()
        self._poll_once()                        # seed _state before the loop starts
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._state_thread.start()

    def close(self) -> None:
        self._cmd_stop.set()
        self._stop_evt.set()
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
            cid = next(self._cmd_counter)
            self._active = {"id": cid, "kind": kind, "status": "running",
                            "progress": 0.0, "error": None}
        self._cmd_stop.clear()
        threading.Thread(target=self._run_cmd, args=(cid, run), daemon=True).start()
        return cid

    def _run_cmd(self, cid: int, run) -> None:
        try:
            run(self._progress_cb(cid))
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
        ("done"/"failed"/"stopped") or "timeout"."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._cmd_lock:
                a = self._active
                if a is not None and a["id"] == cid and a["status"] != "running":
                    return a["status"]
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
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `./robot_control/bin/python -c "import sys, os; sys.path[:0]=[os.path.abspath('.'), os.path.abspath('lib')]; import control.base; print('ok')"`
Expected: prints `ok` (no syntax/import error). (Full runtime behavior is verified in Task 3.)

- [ ] **Step 3: Commit**

```bash
git add lib/control/base.py
git commit -m "feat(control): RobotController base — state loop + async command executor"
```

---

## Task 3: `URController` — connect + state + move + stop

**Files:**
- Create: `lib/control/ur.py`
- Modify: `scripts/control_smoketest.py` (add `test_ur_connect_state` + `test_ur_move`)

- [ ] **Step 1: Write the failing tests**

In `scripts/control_smoketest.py`, add these two functions above `main()`:

```python
def test_ur_connect_state():
    robot_sim.install("ur15")
    from control import make_controller
    c = make_controller("ur15")
    c.connect()
    try:
        st = c.get_state()
        assert st.robot == "ur15"
        assert st.q == robot_sim.UR_HOME, "state q should be the seeded UR home"
        assert st.safety_state == "NORMAL"
        assert st.gripper_frac == 0.0
        assert len(st.pose["pos"]) == 3 and len(st.pose["wxyz"]) == 4
        assert st.activity == "idle"
    finally:
        c.close()
    print("PASS test_ur_connect_state")


def test_ur_move():
    robot_sim.install("ur15")
    from control import make_controller
    c = make_controller("ur15")
    c.connect()
    try:
        target = [0.0, -1.4, 1.4, -1.4, -1.4, 0.2]
        cid = c.move_to_joints(target, speed=5.0)
        assert c.wait(cid, timeout=20.0) == "done", "move did not complete"
        st = c.get_state()
        assert max(abs(a - b) for a, b in zip(st.q, target)) < 1e-6, "did not reach target"
        # a second move while idle should also work (busy only blocks concurrent)
        cid2 = c.move_to_joints(robot_sim.UR_HOME, speed=5.0)
        assert c.wait(cid2, timeout=20.0) == "done"
    finally:
        c.close()
    print("PASS test_ur_move")
```

And add them to `main()` (after `test_state_dataclass()`):

```python
def main():
    test_state_dataclass()
    test_ur_connect_state()
    test_ur_move()
    print("ALL CONTROL SMOKE TESTS PASSED")
```

- [ ] **Step 2: Run it, verify it FAILS**

Run: `./robot_control/bin/python scripts/control_smoketest.py`
Expected: fails at `test_ur_connect_state` — `make_controller` / `control.ur` doesn't exist yet (ImportError or AttributeError).

- [ ] **Step 3: Implement `lib/control/ur.py` (connect/state/move primitives)**

Create `lib/control/ur.py`. Build the robot model + Hand-E grasp offset + RTDE connect by **lifting from `scripts/teleop_ur15.py`**: the model/`TOOL0_T_GRASP` setup (lines 85-100), the RTDE construction + `setPayload` + Hand-E connect (lines 128-150), `ee_pose`/`grasp_pose`/`_grasp_to_tool0` (lines 203-219), and the safety-mode map (lines 156-160). Apply these substitutions: module-level globals become instance attributes (`self._robot`, `self._c`, `self._r`, `self._gripper`, `self._tool0_T_grasp`); `print(...)` status lines may stay as `print`. Use `import robot_common as rc` and bind `UR_*` locals as the script does.

This task implements ONLY: `_connect`, `_close`, `_read_q`, `_read_safety`, `_fk_pose`, `_ik`, `_graceful_stop`, `_hard_stop`, `_gripper_frac`, and a `_run_play` that handles the **move/settle** path (single or multi segment, NO gripper actions yet — gripper comes in Task 4). Concretely:

```python
"""URController — UR15 over RTDE servoJ + Hand-E gripper, behind RobotController."""
from __future__ import annotations

import time

import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description

import hande_gripper
import pyroki_snippets as pks
import robot_common as rc

from .base import RobotController

_UR_SAFETY_MODES = {
    1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY",
    5: "SAFEGUARD_STOP", 6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP",
    8: "VIOLATION", 9: "FAULT", 10: "VALIDATE_JOINT_ID", 11: "UNDEFINED",
}


class URController(RobotController):
    robot_name = "ur15"
    POLL_HZ = 30.0

    def __init__(self) -> None:
        super().__init__()
        self._urdf = load_robot_description(rc.UR_ROBOT_DESCRIPTION)
        self._robot = pk.Robot.from_urdf(self._urdf)
        self._tcp = self._robot.links.names.index(rc.TARGET_LINK)
        g = yourdfpy.URDF.load(rc.UR_GRIPPER_URDF_PATH,
                               filename_handler=rc.make_mesh_resolver(rc.UR_MESH_DIR_PREFIX))
        g.update_cfg(np.array([rc.UR_GRIPPER_FINGER_OPEN]))
        self._tool0_T_grasp = jaxlie.SE3.from_matrix(
            jnp.asarray(g.get_transform(rc.UR_GRASP_LINK, rc.TARGET_LINK)))
        self._c = None
        self._r = None
        self._gripper = None
        self._grip_frac = 0.0

    # ---- lifecycle ----
    def _connect(self) -> None:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
        self._r = RTDEReceiveInterface(rc.UR_ROBOT_IP)
        self._c = RTDEControlInterface(rc.UR_ROBOT_IP)
        try:
            self._c.setPayload(rc.UR_GRIPPER_MASS, list(rc.UR_GRIPPER_COG))
        except Exception as e:
            print(f"setPayload failed ({e}).")
        try:
            self._gripper = hande_gripper.HandEGripper(rc.UR_ROBOT_IP, hande_gripper.DEFAULT_PORT)
            self._gripper.connect()
            self._gripper.activate()
            self._gripper.open()
            self._grip_frac = 0.0
        except Exception as e:
            self._gripper = None
            print(f"Hand-E unavailable ({e}); viz/move only.")

    def _close(self) -> None:
        for fn in (lambda: self._c.servoStop(rc.UR_SERVO_STOP_DECEL),
                   self._c.stopScript, self._c.disconnect, self._r.disconnect,
                   (self._gripper.close if self._gripper is not None else (lambda: None))):
            try:
                fn()
            except Exception:
                pass

    # ---- reads ----
    def _read_q(self):
        return np.asarray(self._r.getActualQ(), dtype=float)

    def _read_safety(self):
        try:
            mode = self._r.getSafetyMode()
            return _UR_SAFETY_MODES.get(mode, f"mode {mode}"), str(mode), True, {}
        except Exception:
            return "UNKNOWN", "?", False, {}

    def _fk_pose(self, q):
        Ts = self._robot.forward_kinematics(cfg=jnp.array(q))
        T = jaxlie.SE3(Ts[self._tcp]).multiply(self._tool0_T_grasp)
        return np.asarray(T.translation()), np.asarray(T.rotation().wxyz)

    def _ik(self, pos, wxyz, q_seed):
        # gizmo/waypoint targets are at the grasp point; map back to a tool0 target.
        T_grasp = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(jnp.asarray(wxyz)), jnp.asarray(pos))
        T_tool0 = T_grasp.multiply(self._tool0_T_grasp.inverse())
        return np.asarray(pks.solve_ik_seeded(
            robot=self._robot, target_link_name=rc.TARGET_LINK,
            target_position=np.asarray(T_tool0.translation()),
            target_wxyz=np.asarray(T_tool0.rotation().wxyz),
            q_seed=q_seed, rest_weight=2.0))

    def _gripper_frac(self):
        return self._grip_frac

    # ---- stops ----
    def _graceful_stop(self) -> None:
        try:
            self._c.servoStop(rc.UR_SERVO_STOP_DECEL)
        except Exception:
            pass

    def _hard_stop(self) -> None:
        try:
            self._c.triggerProtectiveStop()
        except Exception:
            pass

    # ---- motion (move/settle; gripper added in Task 4) ----
    def _run_play(self, segments, speed, progress_cb) -> None:
        dt = 1.0 / rc.UR_STREAM_HZ
        n = len(segments)
        for seg_idx, (q_start, q_goal, _grip) in enumerate(segments):
            if self._cmd_stop.is_set():
                break
            delta = q_goal - q_start
            seg_dur = max(rc.MIN_SEG_DURATION_S, float(np.max(np.abs(delta))) / rc.UR_MAX_JOINT_SPEED)
            alpha = 0.0
            while alpha < 1.0:
                if self._cmd_stop.is_set():
                    break
                q = q_start + delta * rc.alpha_to_s(alpha)
                self._c.servoJ(q.tolist(), 0.0, 0.0, dt, rc.UR_SERVO_LOOKAHEAD, rc.UR_SERVO_GAIN)
                time.sleep(dt)
                alpha = min(1.0, alpha + dt * speed / seg_dur)
            if not self._cmd_stop.is_set() and seg_idx < n - 1:
                hold = max(0.0, rc.DWELL_S / max(0.1, speed))
                t = 0.0
                while t < hold and not self._cmd_stop.is_set():
                    self._c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, rc.UR_SERVO_LOOKAHEAD, rc.UR_SERVO_GAIN)
                    time.sleep(dt)
                    t += dt
            progress_cb((seg_idx + 1) / n)
        # final settle: hold the last target until measured joints arrive (lifted
        # from teleop_ur15.py:608-629 / play_trajectory.py:222-236).
        if not self._cmd_stop.is_set():
            q_final = segments[-1][1]
            deadline = time.monotonic() + rc.UR_SETTLE_MAX_S
            best, stalls = float("inf"), 0
            while not self._cmd_stop.is_set():
                self._c.servoJ(q_final.tolist(), 0.0, 0.0, dt, rc.UR_SERVO_LOOKAHEAD, rc.UR_SETTLE_GAIN)
                time.sleep(dt)
                err = float(np.max(np.abs(np.asarray(self._r.getActualQ()) - q_final)))
                if err < best - rc.UR_SETTLE_EPS_RAD:
                    best, stalls = err, 0
                else:
                    stalls += 1
                if stalls >= rc.UR_SETTLE_STALL_TICKS or time.monotonic() > deadline:
                    break
        try:
            self._c.servoStop(rc.UR_SERVO_STOP_DECEL)
        except Exception:
            pass
```

Then create the factory so `make_controller` works. Replace `lib/control/__init__.py` with:

```python
"""RobotController core: one motion implementation behind the teleop scripts and the API."""
from .base import Busy, RobotController, Unsupported
from .state import RobotState

__all__ = ["RobotController", "RobotState", "Busy", "Unsupported", "make_controller"]


def make_controller(robot: str) -> RobotController:
    if robot == "ur15":
        from .ur import URController
        return URController()
    if robot == "gofa":
        from .gofa import GoFaController
        return GoFaController()
    raise ValueError(f"unknown robot {robot!r} (expected 'ur15' or 'gofa')")
```

- [ ] **Step 4: Run it, verify it PASSES**

Run: `./robot_control/bin/python scripts/control_smoketest.py`
Expected: `PASS test_state_dataclass`, `PASS test_ur_connect_state`, `PASS test_ur_move`, `ALL CONTROL SMOKE TESTS PASSED`, exit 0. (First run pays the JAX/IK JIT — give it ~10-20 s.)

Note: `make_controller("gofa")` will fail until Task 5 because `control.gofa` doesn't exist; that's fine — the factory imports it lazily only when asked.

- [ ] **Step 5: Commit**

```bash
git add lib/control/ur.py lib/control/__init__.py scripts/control_smoketest.py
git commit -m "feat(control): URController — connect, state, profiled move + settle, stop"
```

---

## Task 4: `URController` — `play` with gripper-on-change + `set_gripper`

**Files:**
- Modify: `lib/control/ur.py` (gripper actuation in `_run_play`; `_gripper_blocking`)
- Modify: `scripts/control_smoketest.py` (add `test_ur_play_gripper`)

- [ ] **Step 1: Write the failing test**

In `scripts/control_smoketest.py`, add above `main()`:

```python
def test_ur_play_gripper():
    robot_sim.install("ur15")
    from control import make_controller
    c = make_controller("ur15")
    c.connect()
    try:
        cid = c.play("_sample_ur15", speed=5.0)
        assert c.wait(cid, timeout=40.0) == "done", "play did not complete"
        # final waypoint joints of _sample_ur15 should be reached
        import robot_common as rc
        wps = rc.load_trajectory("_sample_ur15")["waypoints"]
        q_final = wps[-1]["q"]
        st = c.get_state()
        assert max(abs(a - b) for a, b in zip(st.q, q_final)) < 1e-6, "play did not reach final waypoint"
        # gripper command updates state
        gid = c.set_gripper(0.5)
        assert c.wait(gid, timeout=10.0) == "done"
        assert abs(c.get_state().gripper_frac - 0.5) < 1e-9, "set_gripper did not update state"
    finally:
        c.close()
    print("PASS test_ur_play_gripper")
```

And wire it into `main()` after `test_ur_move()`.

- [ ] **Step 2: Run it, verify it FAILS**

Run: `./robot_control/bin/python scripts/control_smoketest.py`
Expected: fails at `test_ur_play_gripper` — `set_gripper` raises `Unsupported` (base default) and/or the gripper fraction never updates (play ignores grip).

- [ ] **Step 3: Implement gripper handling**

In `lib/control/ur.py`, **(a)** add gripper-on-change to `_run_play` by lifting the gripper block from `scripts/teleop_ur15.py:566-596`. Insert it inside the per-segment loop, AFTER the `while alpha < 1.0` motion loop and BEFORE the inter-segment dwell, with these substitutions: `grip` is the third tuple element `segments[seg_idx][2]`; track a running `cur_grip` initialised to `self._grip_frac` before the segment loop; `gui_*`/`viser`/`_update_gripper_viz` lines are dropped; `gripper.move(grip)` stays as `self._gripper.move(grip)` guarded by `self._gripper is not None`; after actuating set `self._grip_frac = grip`. Keep the `GRIP_PREDELAY_S` settle-before-actuate (streaming `q_goal` via `self._c.servoJ`) and the `abs(grip - cur_grip) > rc.GRIP_EPS` gate. The viz finger tween is NOT needed (no viser here) — just sleep `GRIPPER`-equivalent time or skip the tween loop; actuate then continue.

**(b)** Implement `_gripper_blocking`:

```python
    def _gripper_blocking(self, frac, progress_cb) -> None:
        frac = max(0.0, min(1.0, float(frac)))
        if self._gripper is not None:
            self._gripper.move(frac)
            time.sleep(0.8)   # let the fingers move (matches play_trajectory.py)
        self._grip_frac = frac
        progress_cb(1.0)
```

Concretely, the per-segment gripper block in `_run_play` should read (adapt to your `cur_grip` variable placement):

```python
            grip = segments[seg_idx][2]
            if not self._cmd_stop.is_set() and grip is not None and abs(grip - cur_grip) > rc.GRIP_EPS:
                for _ in range(int(rc.GRIP_PREDELAY_S * rc.UR_STREAM_HZ)):   # settle before actuating
                    if self._cmd_stop.is_set():
                        break
                    self._c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, rc.UR_SERVO_LOOKAHEAD, rc.UR_SERVO_GAIN)
                    time.sleep(dt)
                if self._gripper is not None:
                    self._gripper.move(grip)
                    time.sleep(0.8)
                self._grip_frac = grip
                cur_grip = grip
```

Initialise `cur_grip = self._grip_frac` immediately before the `for seg_idx, ...` loop.

- [ ] **Step 4: Run it, verify it PASSES**

Run: `./robot_control/bin/python scripts/control_smoketest.py`
Expected: all four `PASS` lines incl. `PASS test_ur_play_gripper`, then `ALL CONTROL SMOKE TESTS PASSED`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add lib/control/ur.py scripts/control_smoketest.py
git commit -m "feat(control): URController play gripper-on-change + set_gripper"
```

---

## Task 5: `GoFaController` — connect + state + move/play + stop (EGM)

**Files:**
- Create: `lib/control/gofa.py`
- Modify: `scripts/control_smoketest.py` (add `test_gofa_*`)

- [ ] **Step 1: Write the failing tests**

In `scripts/control_smoketest.py`, add above `main()`:

```python
def test_gofa_connect_state():
    robot_sim.install("gofa")
    from control import make_controller
    c = make_controller("gofa")
    c.connect()
    try:
        st = c.get_state()
        assert st.robot == "gofa"
        assert st.q == robot_sim.GOFA_HOME, "state q should be the seeded GoFa home"
        assert st.gripper_frac is None, "GoFa has no gripper"
        assert len(st.pose["pos"]) == 3
    finally:
        c.close()
    print("PASS test_gofa_connect_state")


def test_gofa_move_play():
    robot_sim.install("gofa")
    from control import make_controller
    c = make_controller("gofa")
    c.connect()
    try:
        target = [0.0, 0.1, 0.0, 0.0, 1.5708, 0.0]
        cid = c.move_to_joints(target, speed=5.0)
        assert c.wait(cid, timeout=30.0) == "done", "gofa move did not complete"
        st = c.get_state()
        assert max(abs(a - b) for a, b in zip(st.q, target)) < 1e-6, "gofa did not reach target"
        pid = c.play("_sample_gofa", speed=5.0)
        assert c.wait(pid, timeout=40.0) == "done", "gofa play did not complete"
    finally:
        c.close()
    print("PASS test_gofa_move_play")
```

Wire both into `main()` after `test_ur_play_gripper()`.

- [ ] **Step 2: Run it, verify it FAILS**

Run: `./robot_control/bin/python scripts/control_smoketest.py`
Expected: fails at `test_gofa_connect_state` — `control.gofa` doesn't exist.

- [ ] **Step 3: Implement `lib/control/gofa.py`**

Create `lib/control/gofa.py` by **lifting from `scripts/teleop_gofa_egm.py`** and `scripts/play_trajectory.py:play_gofa`. Implement the same primitive set as UR, plus EGM session management. Key structure:

```python
"""GoFaController — ABB GoFa over EGM (UDP) + RWS, behind RobotController."""
from __future__ import annotations

import time

import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy

import abb_egm
import abb_rws
import pyroki_snippets as pks
import robot_common as rc

from .base import RobotController, Unsupported


class GoFaController(RobotController):
    robot_name = "gofa"
    POLL_HZ = 10.0     # RWS polling rate (matches teleop_gofa_egm)

    def __init__(self) -> None:
        super().__init__()
        self._urdf = yourdfpy.URDF.load(
            rc.GOFA_URDF_PATH, filename_handler=rc.make_mesh_resolver(rc.GOFA_MESH_DIR_PREFIX))
        self._robot = pk.Robot.from_urdf(self._urdf)
        self._tcp = self._robot.links.names.index(rc.TARGET_LINK)
        self._rws = None
        self._egm = None

    def _connect(self) -> None:
        self._rws = abb_rws.RWSClient(host=rc.GOFA_ROBOT_IP, user=rc.GOFA_RWS_USER,
                                      password=rc.GOFA_RWS_PASSWORD)
        try:
            self._rws.request_mastership()
        except Exception as e:
            print(f"WARNING: could not acquire mastership: {e}")
        for flag in (rc.GOFA_RAPID_GO_FLAG, rc.GOFA_RAPID_LEAD_FLAG):
            try:
                self._rws.set_rapid_bool(flag, False, module=rc.GOFA_RAPID_MODULE)
            except Exception:
                pass
        self._egm = abb_egm.EGMSession(local_port=rc.GOFA_EGM_LOCAL_PORT)
        self._egm.start()

    def _close(self) -> None:
        for fn in (lambda: self._rws.set_rapid_bool(rc.GOFA_RAPID_GO_FLAG, False, module=rc.GOFA_RAPID_MODULE),
                   self._egm.stop, self._rws.release_mastership):
            try:
                fn()
            except Exception:
                pass

    def _read_q(self):
        return np.asarray(self._rws.get_joints(), dtype=float)

    def _read_safety(self):
        try:
            st = self._rws.get_controller_state()
            ok = st not in ("guardstop", "emergencystop", "sysfail")
            health = {"egm_rx": self._egm.packets_rx, "egm_tx": self._egm.packets_tx} if self._egm else {}
            return st, st, True, health
        except Exception:
            return "UNKNOWN", "?", False, {}

    def _fk_pose(self, q):
        Ts = self._robot.forward_kinematics(cfg=jnp.array(q))
        T = jaxlie.SE3(Ts[self._tcp])
        return np.asarray(T.translation()), np.asarray(T.rotation().wxyz)

    def _ik(self, pos, wxyz, q_seed):
        return np.asarray(pks.solve_ik_seeded(
            robot=self._robot, target_link_name=rc.TARGET_LINK,
            target_position=np.asarray(pos), target_wxyz=np.asarray(wxyz),
            q_seed=q_seed, rest_weight=2.0))

    def _gripper_frac(self):
        return None

    def _gripper_blocking(self, frac, progress_cb):
        raise Unsupported("GoFa has no gripper")

    def _graceful_stop(self) -> None:
        try:
            self._rws.set_rapid_bool(rc.GOFA_RAPID_GO_FLAG, False, module=rc.GOFA_RAPID_MODULE)
        except Exception:
            pass

    def _hard_stop(self) -> None:
        for fn in (lambda: self._rws.set_rapid_bool(rc.GOFA_RAPID_GO_FLAG, False, module=rc.GOFA_RAPID_MODULE),
                   self._rws.stop_program):
            try:
                fn()
            except Exception:
                pass
```

Then implement `_run_play` by **lifting the EGM streaming loop from `scripts/play_trajectory.py:play_gofa` (lines 284-325)** plus `_cap_seg_duration` (from `teleop_gofa_egm.py:202-219`) and the `start_egm` handshake (`play_gofa` lines 286-298). Substitutions: the local `start_egm()` becomes a helper that sets `egm_go=TRUE` and waits for `self._egm.is_fresh(0.1)` (return False on timeout); `egm.set_target_rad(q)` stays as `self._egm.set_target_rad(q)`; gate each tick on `self._cmd_stop`; after the last segment hold the final target for `GOFA_HOLD_AFTER_PLAY_S` then set `egm_go=FALSE`; call `progress_cb((seg_idx+1)/n)`. Add a `_cap_seg_duration(self, q_start, delta, seg_duration, dt)` method (lift verbatim, FK via `self._fk_pose(q)[0]`). The settle/gripper UR logic does NOT apply here. Sketch:

```python
    def _start_egm(self) -> bool:
        q_now = self._read_q_copy()
        self._egm.set_target_rad(q_now.tolist())
        try:
            self._rws.set_rapid_bool(rc.GOFA_RAPID_GO_FLAG, True, module=rc.GOFA_RAPID_MODULE)
        except Exception:
            return False
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if self._egm.is_fresh(0.1):
                return True
            time.sleep(0.05)
        return False

    def _cap_seg_duration(self, q_start, delta, seg_duration, dt):
        alpha, prev_p, peak = 0.0, self._fk_pose(q_start)[0], 0.0
        while alpha < 1.0:
            alpha = min(1.0, alpha + dt / seg_duration)
            p = self._fk_pose(q_start + delta * rc.alpha_to_s(alpha))[0]
            peak = max(peak, float(np.linalg.norm(p - prev_p)) / dt)
            prev_p = p
        if peak > rc.GOFA_MAX_TCP_SPEED:
            seg_duration *= peak / rc.GOFA_MAX_TCP_SPEED
        return seg_duration

    def _run_play(self, segments, speed, progress_cb) -> None:
        dt = 1.0 / rc.GOFA_STREAM_HZ
        if not self._start_egm():
            raise RuntimeError("EGM did not start (no packets in 3s)")
        n = len(segments)
        try:
            for seg_idx, (q_start, q_goal, _grip) in enumerate(segments):
                if self._cmd_stop.is_set():
                    break
                delta = q_goal - q_start
                seg_dur = max(rc.MIN_SEG_DURATION_S, float(np.max(np.abs(delta))) / rc.GOFA_MAX_JOINT_SPEED)
                seg_dur = self._cap_seg_duration(q_start, delta, seg_dur, dt)
                alpha = 0.0
                while alpha < 1.0:
                    if self._cmd_stop.is_set():
                        break
                    q = q_start + delta * rc.alpha_to_s(alpha)
                    self._egm.set_target_rad(q.tolist())
                    time.sleep(dt)
                    alpha = min(1.0, alpha + dt * speed / seg_dur)
                if not self._cmd_stop.is_set() and seg_idx < n - 1:
                    for _ in range(int(max(0.0, rc.DWELL_S / max(0.1, speed)) * rc.GOFA_STREAM_HZ)):
                        if self._cmd_stop.is_set():
                            break
                        self._egm.set_target_rad(q_goal.tolist())
                        time.sleep(dt)
                progress_cb((seg_idx + 1) / n)
            if not self._cmd_stop.is_set():
                hold = segments[-1][1]
                for _ in range(int(rc.GOFA_HOLD_AFTER_PLAY_S * rc.GOFA_STREAM_HZ)):
                    self._egm.set_target_rad(hold.tolist())
                    time.sleep(dt)
        finally:
            try:
                self._rws.set_rapid_bool(rc.GOFA_RAPID_GO_FLAG, False, module=rc.GOFA_RAPID_MODULE)
            except Exception:
                pass
```

- [ ] **Step 4: Run it, verify it PASSES**

Run: `./robot_control/bin/python scripts/control_smoketest.py`
Expected: all `PASS` lines incl. `PASS test_gofa_connect_state` and `PASS test_gofa_move_play`, then `ALL CONTROL SMOKE TESTS PASSED`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add lib/control/gofa.py scripts/control_smoketest.py
git commit -m "feat(control): GoFaController — connect, state, EGM move/play, stop"
```

---

## Task 6: Migrate `play_trajectory.py` onto the controller

**Files:**
- Modify: `scripts/play_trajectory.py`

Goal: `play_ur15`/`play_gofa` stop carrying their own motion loops and instead build a controller and call `controller.play(...)`. Keep the CLI, the dry-run, chaining, the confirm prompt, and `print_plan`/`estimate_duration` (those don't touch hardware). The first segment "current pose → waypoint 1" is now produced by the controller's `_build_segments` (which reads the live pose), so the combined waypoint list is passed straight to `controller.play(waypoints)`.

- [ ] **Step 1: Rewrite `play_ur15` and `play_gofa` to delegate**

Replace the bodies of `play_ur15(data, speed, no_confirm)` and `play_gofa(data, speed, no_confirm)` (lines 147-240 and 243-325) with a single shared helper. Add near the top (after the existing imports):

```python
from control import make_controller
```

Replace both functions with:

```python
def play_on_controller(robot: str, data, speed, no_confirm):
    c = make_controller(robot)
    print(f"Connecting to {robot} ...")
    c.connect()
    try:
        segments = build_segments(c.get_state().q, data["waypoints"])
        print_plan(robot, segments, speed)
        if not no_confirm and not confirm(f"Execute on the real {robot}? [y/N] "):
            print("Aborted."); return
        cid = c.play(data["waypoints"], speed=speed)
        status = c.wait(cid, timeout=600.0)
        st = c.command_status(cid)
        if status != "done":
            print(f"Play ended: {status}" + (f" ({st['error']})" if st and st.get("error") else ""))
        else:
            print("Done.")
    finally:
        c.close()
```

And change the dispatch in `main()` (lines 141-144) from:

```python
    if robot == "ur15":
        play_ur15(combined, args.speed, args.no_confirm)
    else:
        play_gofa(combined, args.speed, args.no_confirm)
```

to:

```python
    play_on_controller(robot, combined, args.speed, args.no_confirm)
```

Delete the now-unused `play_ur15`/`play_gofa` functions and any imports they alone used (e.g. the in-function `from rtde_control import ...`, `import abb_egm`, the GoFa pyroki FK block) — but KEEP `build_segments`, `estimate_duration`, `print_plan`, `confirm`, `load_trajectory`, and the top-level argparse/main. (`build_segments` is still used for `print_plan`; the controller has its own copy for actual motion.)

- [ ] **Step 2: Verify the headless e2e still passes (UR + GoFa)**

Run: `./robot_control/bin/python scripts/sim.py play _sample_ur15 --no-confirm --speed 5`
Expected: prints the plan + `Done.`, exit 0 (now driven by `URController`).

Run: `./robot_control/bin/python scripts/sim.py play _sample_gofa --no-confirm --speed 5`
Expected: prints the plan + `Done.`, exit 0 (now driven by `GoFaController`).

Run: `./robot_control/bin/python scripts/play_trajectory.py _sample_ur15 --dry-run`
Expected: prints the plan and exits (no robot needed), exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/play_trajectory.py
git commit -m "refactor(play): drive play_trajectory.py through RobotController"
```

---

## Task 7: Migrate `teleop.py` (headless recorder) onto the controller

**Files:**
- Modify: `scripts/teleop.py`

Goal: `URBackend`/`GoFaBackend` stop owning their own hardware clients and instead wrap a `RobotController` for the reads the recorder needs (`read_joints`, `grasp_pose`, gripper adjust) and free-drive. The recorder's free-drive/keypress/dashboard loop is unchanged. Free-drive is hardware-specific (UR `teachMode`, GoFa lead-through) and is NOT a controller motion command — expose it via small controller methods.

- [ ] **Step 1: Add free-drive + grasp-pose helpers to the controllers**

In `lib/control/base.py`, add to the public API (near the other commands):

```python
    def grasp_pose(self, q):
        """FK grasp/EE pose for q -> (pos, wxyz). Used by the recorder dashboard."""
        return self._fk_pose(np.asarray(q, dtype=float))

    def start_freedrive(self) -> None:
        self._start_freedrive()

    def stop_freedrive(self) -> None:
        self._stop_freedrive()

    def _start_freedrive(self) -> None: raise NotImplementedError
    def _stop_freedrive(self) -> None: raise NotImplementedError
```

In `lib/control/ur.py` add:

```python
    def _start_freedrive(self) -> None:
        self._c.teachMode()

    def _stop_freedrive(self) -> None:
        try:
            self._c.endTeachMode()
        except Exception:
            pass

    def adjust_grip(self, delta):
        if self._gripper is None:
            return None
        self._grip_frac = max(0.0, min(1.0, self._grip_frac + delta))
        try:
            self._gripper.move(self._grip_frac)
        except Exception as e:
            print(f"  gripper cmd failed: {e}")
        return self._grip_frac
```

In `lib/control/gofa.py` add:

```python
    def _start_freedrive(self) -> None:
        self._rws.set_rapid_bool(rc.GOFA_RAPID_GO_FLAG, False, module=rc.GOFA_RAPID_MODULE)
        self._rws.set_rapid_bool(rc.GOFA_RAPID_LEAD_FLAG, True, module=rc.GOFA_RAPID_MODULE)

    def _stop_freedrive(self) -> None:
        try:
            self._rws.set_rapid_bool(rc.GOFA_RAPID_LEAD_FLAG, False, module=rc.GOFA_RAPID_MODULE)
        except Exception:
            pass
```

GoFa needs no `adjust_grip` override — add the no-op default to the base instead, so any gripper-less robot inherits it:

```python
    # in lib/control/base.py, alongside the other public methods
    def adjust_grip(self, delta):
        return None
```

- [ ] **Step 2: Rewrite the recorder backends to wrap a controller**

In `scripts/teleop.py`, replace the `URBackend` (lines 105-221) and `GoFaBackend` (lines 224-308) classes with one controller-backed backend (the recorder only needs: `robot_name`, `grasp_pose`, `read_joints`, `make_waypoint`, `grip_text`, `adjust_grip`, `start_freedrive`, `stop_freedrive`, `hard_stop`, `close`). Replace both classes and `make_backend` with:

```python
from control import make_controller   # add near the top imports


class Backend:
    def __init__(self, choice: str):
        self.robot_name = "ur15" if choice == "ur" else "gofa"
        print(f"Connecting to {self.robot_name} ...")
        self._c = make_controller(self.robot_name)
        self._c.connect()

    def read_joints(self):
        return np.asarray(self._c.get_state().q, dtype=np.float64)

    def grasp_pose(self, q):
        pos, wxyz = self._c.grasp_pose(q)
        return np.asarray(pos), np.asarray(wxyz)

    def make_waypoint(self, q):
        pos, wxyz = self.grasp_pose(q)
        wp = {"q": q.tolist(), "pos": pos.tolist(), "wxyz": wxyz.tolist()}
        frac = self._c.get_state().gripper_frac
        if frac is not None:
            wp["grip"] = frac
        return wp

    def grip_text(self):
        frac = self._c.get_state().gripper_frac
        if frac is None:
            return "n/a (no gripper)"
        return f"{int(round(frac * 100))}% closed"

    def adjust_grip(self, delta):
        return self._c.adjust_grip(delta)

    def start_freedrive(self):
        self._c.start_freedrive()

    def stop_freedrive(self):
        self._c.stop_freedrive()

    def hard_stop(self):
        self.stop_freedrive()
        self._c.estop()
        print("  *** hard stop issued ***")

    def close(self):
        self.stop_freedrive()
        self._c.close()


def make_backend(choice: str):
    return Backend(choice)
```

Delete the now-unused imports that only the old backends used (`from rtde_control import ...` lived inside the old classes, so nothing at module scope to remove; remove `import abb_rws` if it was module-scope — verify it wasn't). Keep everything else (the dashboard, `key_loop`, `record_session`, `main`).

Note: the recorder's old UR backend warmed FK and waited for gripper calibration; the controller's `connect()` already opens the gripper, and FK warms on first `grasp_pose`. That is acceptable for the recorder.

- [ ] **Step 3: Verify it boots and runs against the sim**

Free-drive capture in sim is a documented no-op (no hand to move the arm), but the recorder must still start, read state, and exit cleanly. Verify it boots without traceback:

```bash
printf '' | timeout 25 ./robot_control/bin/python scripts/sim.py teleop _tmp_rec --robot ur 2>&1 | head -20
```

Expected: it connects (`Connecting to ur15 ...`), the controller starts, and the dashboard appears (or it waits on stdin). No traceback / no ImportError. (It needs a TTY for the full key loop; this check only confirms it boots against the controller+fakes.)

Also re-run the controller smoke test to confirm the base/ur/gofa additions didn't regress:

Run: `./robot_control/bin/python scripts/control_smoketest.py`
Expected: `ALL CONTROL SMOKE TESTS PASSED`, exit 0.

- [ ] **Step 4: Commit**

```bash
git add lib/control/base.py lib/control/ur.py lib/control/gofa.py scripts/teleop.py
git commit -m "refactor(teleop): drive the headless recorder through RobotController"
```

---

## Task 8: Update docs for the controller core

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Document `lib/control/` in the layout tree + a short section**

Read `CLAUDE.md` to get exact current text, then:

(a) In the `lib/` block of the project-layout tree, after the `dispatch.py` entry, add (fix connectors so the last entry uses `└──`):

```
│   ├── dispatch.py             #   shared target->script map + dispatch() for real.py / sim.py
│   └── control/                #   RobotController core (state.py, base.py, ur.py, gofa.py) — one motion impl
```

(b) Immediately before the `# UR15` heading (or after the Simulation section), add:

```
## RobotController core — `lib/control/`

One thread-safe motion implementation behind every surface. `make_controller("ur15"|"gofa")`
returns a controller that owns the hardware client and exposes async high-level commands —
`move_to_joints` / `move_to_pose` / `play` / `set_gripper` / `stop` / `estop` (each returns a
command id; `wait(id)` blocks for the result) — plus `get_state() -> RobotState` (joints, FK
pose, gripper, safety, activity) from a background state-poll thread. The headless players
(`play_trajectory.py`, `teleop.py`) run on it; the viser teleops and the remote API are
migrating onto it. Because it uses the same hardware clients the sim fakes shadow, the whole
core runs offline — `./robot_control/bin/python scripts/control_smoketest.py` exercises it
(move/play/stop/gripper/state) against `lib/robot_sim.py` with no robot.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document the RobotController core (lib/control)"
```

---

## Self-Review notes (for the implementer)

- **No pytest:** verification is `scripts/control_smoketest.py` (stdlib asserts) + the existing `sim.py play` e2e. Don't add pytest.
- **Behavior preservation:** Tasks 3-5 lift tuned loops. The acceptance bar for the migrations (Tasks 6-7) is that `sim.py play _sample_ur15` / `_sample_gofa` still end `Done.` — i.e. identical observable behavior to before. If they don't, the lift drifted from the original; re-check the source lines.
- **Async commands:** every `move_*`/`play`/`set_gripper` returns a command id and runs on a worker thread; tests use `wait(cid)`. `stop()`/`estop()` set `_cmd_stop`, which every loop checks.
- **Sim ordering:** the smoke test calls `robot_sim.install(robot)` BEFORE `make_controller(robot)` so the controller's lazy `from rtde_control import …` / `import abb_rws` resolve to the fakes.
- **Out of scope here:** lease, heartbeat watchdog, auth, FastAPI, the `api` dispatcher target, and the viser teleop migrations — those are later plans (Phase 2 + the viser-migration plan).
