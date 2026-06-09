# Offline Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let every teleop entry point run offline (no UR15, no GoFa, no network) by injecting fake robot-transport modules into `sys.modules` and `runpy`-ing the real, unmodified scripts — so the simulator reuses all real primitives and tracks features automatically.

**Architecture:** A perfect-tracking kinematic sim (`lib/robot_sim.py`) provides fake versions of the ~5 hardware client classes backed by one shared joint vector. A shared dispatcher (`lib/dispatch.py`) maps a target name to its script; `scripts/real.py` and `scripts/sim.py` are thin verbs over it that differ only in whether the sim shim is installed first. The four implementation scripts are untouched.

**Tech Stack:** Python 3.13, stdlib only for the sim/dispatch core (`sys`, `threading`, `time`, `types`, `runpy`). Tests are a stdlib-`assert` smoke script run with the venv python (no pytest in this project). The sim runs the *real* jax/pyroki/yourdfpy/viser stack — only the robot transport is faked.

**Spec:** `docs/superpowers/specs/2026-06-08-offline-simulator-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `lib/robot_sim.py` (create) | `SimWorld` singleton + fake client classes (UR + GoFa) + fake RAPID supervisor + `install()` |
| `lib/dispatch.py` (create) | `TARGETS` map + `dispatch(prog, argv, sim)` — runpy + argv plumbing |
| `scripts/real.py` (create) | `real.py <target> [args]` → `dispatch(..., sim=False)` |
| `scripts/sim.py` (create) | `sim.py <target> [args]` → `dispatch(..., sim=True)` |
| `scripts/sim_smoketest.py` (create) | stdlib-`assert` smoke test for the fakes + GoFa handshake |
| `CLAUDE.md`, `README.md` (modify) | document `real.py`/`sim.py`, the layout, the free-drive limitation |

No changes to `teleop_ur15.py`, `teleop_gofa_egm.py`, `play_trajectory.py`, `teleop.py`.

Fixtures already exist: `trajectories/_sample_ur15.json` (3 wp, all carry `q`) and `trajectories/_sample_gofa.json` (2 wp, all carry `q`).

---

## Task 1: SimWorld + UR fakes + install (UR subset)

**Files:**
- Create: `lib/robot_sim.py`
- Create: `scripts/sim_smoketest.py`

- [ ] **Step 1: Write the failing test (UR round-trip)**

Create `scripts/sim_smoketest.py`:

```python
"""Offline-sim smoke test: drive the fake transports directly and assert the
kinematic round-trips + (Task 2) the GoFa EGM handshake.

  ./robot_control/bin/python scripts/sim_smoketest.py

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


def test_ur_roundtrip():
    robot_sim.install("ur15")
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    import hande_gripper

    c = RTDEControlInterface("192.168.0.1")
    r = RTDEReceiveInterface("192.168.0.1")
    assert r.getActualQ() == robot_sim.UR_HOME, "home pose not seeded"
    assert r.getSafetyMode() == 1, "safety mode should be NORMAL (1)"

    target = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    c.servoJ(target, 0.0, 0.0, 1.0 / 50, 0.1, 300)
    assert r.getActualQ() == target, "servoJ did not update the joints"

    assert hande_gripper.DEFAULT_PORT == 63352, "DEFAULT_PORT not exposed"
    g = hande_gripper.HandEGripper("192.168.0.1", hande_gripper.DEFAULT_PORT)
    g.connect()
    g.activate()
    g.move(0.5)  # must not raise
    print("PASS test_ur_roundtrip")


def main():
    test_ur_roundtrip()
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./robot_control/bin/python scripts/sim_smoketest.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'robot_sim'`

- [ ] **Step 3: Write the minimal implementation**

Create `lib/robot_sim.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./robot_control/bin/python scripts/sim_smoketest.py`
Expected: prints `PASS test_ur_roundtrip` then `ALL SMOKE TESTS PASSED`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add lib/robot_sim.py scripts/sim_smoketest.py
git commit -m "feat(sim): SimWorld + UR fake transports + install()"
```

---

## Task 2: GoFa fakes (RWS + EGM + fake RAPID supervisor)

**Files:**
- Modify: `lib/robot_sim.py` (add GoFa fakes; extend `install()`)
- Modify: `scripts/sim_smoketest.py` (add the handshake test)

- [ ] **Step 1: Write the failing test (GoFa handshake)**

In `scripts/sim_smoketest.py`, add this function above `main()`:

```python
def test_gofa_handshake():
    robot_sim.install("gofa")
    import abb_rws
    import abb_egm

    rws = abb_rws.RWSClient(host="192.168.0.1")
    egm = abb_egm.EGMSession(local_port=6510)
    egm.start()
    try:
        assert rws.get_controller_state() == "motoron"
        assert rws.get_joints() == robot_sim.GOFA_HOME, "GoFa home not seeded"

        # Arm EGM: preload a target, then flip egm_go TRUE (as the scripts do).
        egm.set_target_rad(robot_sim.GOFA_HOME)
        rws.set_rapid_bool("egm_go", True, module="PyEgm")

        deadline = time.time() + 2.0
        while time.time() < deadline and not egm.is_fresh(0.1):
            time.sleep(0.02)
        assert egm.is_fresh(0.1), "EGM never went fresh after egm_go=TRUE"

        # A moving target keeps egm_go TRUE and is applied to the joints.
        tgt = [0.0, 0.1, 0.0, 0.0, 1.5708, 0.0]
        egm.set_target_rad(tgt)
        time.sleep(0.1)
        assert rws.get_joints() == tgt, "EGM target not applied to joints"
        assert rws.get_rapid_data("egm_go", module="PyEgm") == "TRUE"

        # Holding the target steady > COND_TIME clears egm_go (EGMRunJoint converged).
        deadline = time.time() + robot_sim.COND_TIME + 1.0
        while time.time() < deadline and \
                rws.get_rapid_data("egm_go", module="PyEgm") == "TRUE":
            egm.set_target_rad(tgt)  # same value held
            time.sleep(0.05)
        assert rws.get_rapid_data("egm_go", module="PyEgm") == "FALSE", \
            "egm_go never auto-cleared (CondTime mimic broken)"
    finally:
        egm.stop()
    print("PASS test_gofa_handshake")
```

And change `main()` to call it:

```python
def main():
    test_ur_roundtrip()
    test_gofa_handshake()
    print("ALL SMOKE TESTS PASSED")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./robot_control/bin/python scripts/sim_smoketest.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'abb_rws'` (the GoFa shims aren't injected yet).

- [ ] **Step 3: Write the minimal implementation**

In `lib/robot_sim.py`, add the GoFa fakes immediately **before** the `# ---------- install ----------` section:

```python
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
```

Then extend `install()` to inject the two GoFa shims. Add these two lines inside `install()`, right after the `hande_gripper` line and before the `home = ...` line:

```python
    sys.modules["abb_rws"] = _module("abb_rws", RWSClient=FakeRWS)
    sys.modules["abb_egm"] = _module("abb_egm", EGMSession=FakeEGM)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./robot_control/bin/python scripts/sim_smoketest.py`
Expected: prints `PASS test_ur_roundtrip`, `PASS test_gofa_handshake`, `ALL SMOKE TESTS PASSED`, exit 0. (The handshake test takes ~1.2 s for the CondTime wait.)

- [ ] **Step 5: Commit**

```bash
git add lib/robot_sim.py scripts/sim_smoketest.py
git commit -m "feat(sim): GoFa RWS + EGM fakes with fake RAPID supervisor"
```

---

## Task 3: Shared dispatcher + real.py / sim.py + e2e verification

**Files:**
- Create: `lib/dispatch.py`
- Create: `scripts/real.py`
- Create: `scripts/sim.py`

- [ ] **Step 1: Write `lib/dispatch.py`**

```python
"""Shared dispatcher for the real and simulated entry points.

real.py / sim.py both call dispatch(): identical target->script map and argv
plumbing, differing only in whether the offline sim shim is installed first.
Single source of truth so real and sim never drift in what they can launch.
"""
from __future__ import annotations

import os
import runpy
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")

TARGETS = {
    "ur15": "teleop_ur15.py",
    "gofa": "teleop_gofa_egm.py",
    "play": "play_trajectory.py",
    "teleop": "teleop.py",
}


def dispatch(prog: str, argv: list[str], sim: bool = False) -> None:
    """argv = sys.argv[1:]. Resolve the target, optionally install the sim shim,
    then runpy the real script as __main__ with the remaining args."""
    if not argv or argv[0] not in TARGETS:
        print(f"usage: {prog} <{'|'.join(TARGETS)}> [args...]")
        raise SystemExit(2)
    target, rest = argv[0], argv[1:]
    if sim:
        import robot_sim  # lazy: the real path never imports sim machinery
        robot_sim.install(target)
        print(f"[sim] offline simulator active — no robot, no network ({target})")
    script = os.path.join(_SCRIPTS, TARGETS[target])
    sys.argv = [TARGETS[target], *rest]   # so play/teleop argparse sees the right argv
    runpy.run_path(script, run_name="__main__")
```

- [ ] **Step 2: Write `scripts/real.py`**

```python
#!/usr/bin/env python
"""real.py <ur15|gofa|play|teleop> [args] — run a teleop entry point on real hardware.

Thin verb over lib/dispatch.py; the offline twin is sim.py.
  ./robot_control/bin/python scripts/real.py ur15
  ./robot_control/bin/python scripts/real.py play traj1 --speed 0.5
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dispatch import dispatch  # noqa: E402

dispatch("real.py", sys.argv[1:], sim=False)
```

- [ ] **Step 3: Write `scripts/sim.py`**

```python
#!/usr/bin/env python
"""sim.py <ur15|gofa|play|teleop> [args] — run a teleop entry point OFFLINE (no robot).

Injects fake robot transports (lib/robot_sim.py) into sys.modules, then runs the
real, unmodified script. The real twin is real.py.
  ./robot_control/bin/python scripts/sim.py ur15
  ./robot_control/bin/python scripts/sim.py play _sample_ur15 --no-confirm
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dispatch import dispatch  # noqa: E402

dispatch("sim.py", sys.argv[1:], sim=True)
```

- [ ] **Step 4: Verify usage handling (no robot, no sim machinery touched)**

Run: `./robot_control/bin/python scripts/sim.py`
Expected: prints `usage: sim.py <ur15|gofa|play|teleop> [args...]`, exit 2.

Run: `./robot_control/bin/python scripts/real.py bogus`
Expected: prints `usage: real.py <ur15|gofa|play|teleop> [args...]`, exit 2.

- [ ] **Step 5: Verify the UR play e2e (real motion logic, no robot)**

Run: `./robot_control/bin/python scripts/sim.py play _sample_ur15 --no-confirm --speed 5`
Expected: prints `[sim] offline simulator active ...`, the plan (`Robot: ur15`, segments), `Segment 1/3` … `Segment 3/3`, a `[settle] final joint error …` line, then `Done.` No traceback, exit 0.

- [ ] **Step 6: Verify the GoFa play e2e (EGM handshake, no robot)**

Run: `./robot_control/bin/python scripts/sim.py play _sample_gofa --no-confirm --speed 5`
Expected: prints the plan (`Robot: gofa`), `Segment 1/2`, `Segment 2/2`, then `Done.` No traceback, exit 0. (If it hangs at "EGM did not start", the supervisor/handshake is broken — revisit Task 2.)

- [ ] **Step 7: Re-run the smoke test (guard against regressions)**

Run: `./robot_control/bin/python scripts/sim_smoketest.py`
Expected: `ALL SMOKE TESTS PASSED`, exit 0.

- [ ] **Step 8: Commit**

```bash
git add lib/dispatch.py scripts/real.py scripts/sim.py
git commit -m "feat(sim): symmetric real.py/sim.py dispatchers over shared dispatch.py"
```

---

## Task 4: Manual viser verification (UR + GoFa)

**Files:** none (verification only).

> This step needs a human + browser; it cannot be fully automated (the viser scripts start a blocking server). The viser teleops load the real URDFs — `sim.py ur15` needs `ur15_description` already cached by `robot_descriptions` (it is, if the real UR teleop has ever run on this machine; otherwise the first run fetches it once).

- [ ] **Step 1: UR15 viser sim**

Run: `./robot_control/bin/python scripts/sim.py ur15`
Then open the printed `http://localhost:8080`. Confirm:
- console prints `[sim] offline simulator active … (ur15)`, then `Warming up IK solver …`, then `viser running.`
- the UR15 + Hand-E render; drag the gizmo → click **Plan** → **Play**: the URDF animates along the trajectory.
- tick **Live (drive robot)**: the arm follows the gizmo. Move the **Gripper close %** slider: the fingers animate.
- Ctrl-C exits cleanly.

- [ ] **Step 2: GoFa viser sim**

Run: `./robot_control/bin/python scripts/sim.py gofa`
Open `http://localhost:8080`. Confirm the GoFa renders, **Plan** → **Play** animates, tick **Execute on robot (EGM stream)** then **Play** completes (status reaches `Done`, `EGM` box shows activity then `idle`), and Ctrl-C exits cleanly.

- [ ] **Step 3: Commit (if any notes/tweaks were needed)**

No code changes expected. If a home-pose tweak was needed in `lib/robot_sim.py`, commit it:

```bash
git add lib/robot_sim.py
git commit -m "chore(sim): tune home pose after viser check"
```

---

## Task 5: Documentation

**Files:**
- Modify: `README.md:12-16`
- Modify: `CLAUDE.md` (run command near top; project-layout tree; a new sim subsection)

- [ ] **Step 1: Update `README.md` run block**

Replace the run block at `README.md:12-16`:

```
```bash
./robot_control/bin/python scripts/teleop_ur15.py   # or scripts/teleop_gofa_egm.py
```

Then open the printed `http://localhost:8080`.
```

with:

```
```bash
# real hardware:
./robot_control/bin/python scripts/real.py ur15      # or: gofa | play <name> | teleop
# offline simulation (no robot, no network):
./robot_control/bin/python scripts/sim.py  ur15      # same targets — runs the real scripts vs a fake arm
```

Then open the printed `http://localhost:8080`. The four scripts (`teleop_ur15.py`, `teleop_gofa_egm.py`, `play_trajectory.py`, `teleop.py`) still run directly too.
```

- [ ] **Step 2: Update the run command near the top of `CLAUDE.md`**

In `CLAUDE.md`, replace the existing run line:

```
```bash
./robot_control/bin/python scripts/teleop_ur15.py   # or scripts/teleop_gofa_egm.py
```
```

with:

```
```bash
./robot_control/bin/python scripts/real.py ur15   # real hardware  (targets: ur15 | gofa | play | teleop)
./robot_control/bin/python scripts/sim.py  ur15   # offline sim    (same targets, no robot/network)
```
```

- [ ] **Step 3: Add the new files to the project-layout tree in `CLAUDE.md`**

In the `scripts/` block of the layout tree, add these lines after the `verify_hande.py` line:

```
│   ├── real.py                 #   dispatcher: real.py <ur15|gofa|play|teleop> [args] on real hardware
│   ├── sim.py                  #   dispatcher: sim.py  <ur15|gofa|play|teleop> [args] offline (fake arm)
│   └── sim_smoketest.py        #   stdlib-assert smoke test for the sim fakes + EGM handshake
```

In the `lib/` block, add these lines after the `hande_gripper.py` line:

```
│   ├── robot_sim.py            #   offline sim: SimWorld + fake transports + install() (sys.modules shim)
│   └── dispatch.py             #   shared target->script map + dispatch() for real.py / sim.py
```

- [ ] **Step 4: Add a "Simulation (offline testing)" subsection to `CLAUDE.md`**

Insert this section immediately before the `# UR15` heading:

```
## Simulation — `scripts/sim.py` (offline, no robot)

`sim.py <target>` runs the **real, unmodified** teleop scripts against a
perfect-tracking kinematic sim, so trajectories / IK / the viser UI / play
profiles can be exercised on the dev machine with no UR15, GoFa, or network. It
works by injecting fake transport modules (`rtde_control`, `rtde_receive`,
`hande_gripper`, `abb_rws`, `abb_egm`) into `sys.modules`, then `runpy`-ing the
target script — see `lib/robot_sim.py`. Because it literally runs `teleop_ur15.py` (etc.), every
feature works in sim automatically and never drifts. `real.py` and `sim.py` are
the same dispatcher (`lib/dispatch.py`) and differ only in that `sim.py`
installs the shim first.

```bash
./robot_control/bin/python scripts/sim.py ur15                         # UR15 teleop, simulated
./robot_control/bin/python scripts/sim.py gofa                         # GoFa teleop, simulated
./robot_control/bin/python scripts/sim.py play _sample_ur15 --no-confirm
./robot_control/bin/python scripts/sim_smoketest.py                    # fast fakes + handshake check
```

The sim still needs the `robot_control/` venv (it runs the real jax/pyroki/viser
stack and the real URDFs) — "offline" means no *robot*, not no Python deps.

**One limitation — free-drive/teach can't be hand-guided** (no physical arm to
move): `teachMode`/lead-through are no-ops in sim, so "Capture" during free-drive
just re-grabs the current pose. Offline authoring is via the **gizmo + Add
waypoint**, **Live**, or **Plan**. The headless `teleop.py` recorder runs (UI,
keys, save all work) but captured points stay at the home pose — a plumbing test,
not realistic authoring. Home poses (`UR_HOME` / `GOFA_HOME` / `NEUTRAL_HOME`)
are tunable constants in `lib/robot_sim.py`.
```

- [ ] **Step 5: Verify the docs render and commit**

Run: `git diff --stat`
Expected: `README.md` and `CLAUDE.md` modified.

```bash
git add README.md CLAUDE.md
git commit -m "docs: document offline sim (real.py/sim.py) + layout + free-drive limitation"
```

---

## Self-Review notes (for the implementer)

- **Why no pytest:** this project has no test suite or pytest in the venv; the smoke test is a stdlib-`assert` script run directly, matching the scaffold's style. Don't add pytest.
- **Ordering matters:** `install()` must run *before* the target imports any transport module — `dispatch()` guarantees this (install → then `runpy`). In the smoke test, call `install()` first in each test.
- **`sys.modules` precedence:** `ur_rtde` (real `rtde_control`/`rtde_receive`) *is* installed in the venv; injecting the fake into `sys.modules` first shadows it, so `import rtde_control` returns the fake without touching the real package. Same for the real `abb_rws`/`abb_egm` in `lib/`.
- **Perfect tracking + settle loops:** because `getActualQ()`/`get_joints()` return the last commanded `q` exactly, the UR settle loop converges via its stall-tick path (~10 ticks) — it terminates; don't "fix" it.
- **GoFa CondTime:** `HOLD_AFTER_PLAY_S` (1.5 s) > `COND_TIME` (1.0 s), so a held final target auto-clears `egm_go` and `_wait_egm_clear()` succeeds. The convergence clock keys off the target *value* changing, not the call count.
```
