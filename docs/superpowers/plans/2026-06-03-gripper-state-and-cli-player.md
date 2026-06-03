# Gripper State Tracking + Headless Trajectory Player — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the UR15 gripper state a single tracked value (saved into trajectories, mirrored in the sim, settled 0.5 s before actuation), and add a headless CLI to replay saved trajectories on either arm without viser.

**Architecture:** Edit `teleop_ur15.py` in place for the gripper-state model and on-change replay. Add a new self-contained `play_trajectory.py` that reuses the already-modular transport classes (`hande_gripper`, `abb_rws`, `abb_egm`) and duplicates only the small trapezoidal-profile helpers. The two teleop scripts are not refactored for the CLI (D1).

**Tech Stack:** Python 3.13 (`robot_control/` venv), numpy, ur_rtde, viser, pyroki/jax (GoFa FK only), `hande_gripper.py`, `abb_rws.py`, `abb_egm.py`.

**Testing reality:** This codebase has no pytest suite and the teleop scripts connect to hardware at import time, so they can't be imported on a robot-less machine. Verification here is: `py_compile` for syntax, `grep` for removed-symbol consistency, and the CLI's `--dry-run` (which never touches transports) run against a sample trajectory on the Mac. Steps that move a real robot are **manual** — run on the Oracle PC that reaches the arms. Each such step is marked **[MANUAL — robot]**.

---

### Task 1: UR15 — gripper-state model + startup `open` (spec A)

**Files:**
- Modify: `teleop_ur15.py` (constants ~line 64; shared state ~line 119; gripper connect ~line 137)

- [ ] **Step 1: Add the `GRIP_PREDELAY_S` constant**

In `teleop_ur15.py`, right after the `GRIPPER_COG` line (line 72), add:

```python
GRIP_PREDELAY_S = 0.5           # hold/settle the arm this long before actuating the gripper
```

- [ ] **Step 2: Add the `gripper_state` global**

In the `# ---------- shared state ----------` block, replace this line (line 119):

```python
gripper_finger = GRIPPER_FINGER_OPEN   # displayed per-side finger opening (m); start open
```

with:

```python
gripper_finger = GRIPPER_FINGER_OPEN   # displayed per-side finger opening (m); rendered value
gripper_state = "open"                 # single source of truth: "open" | "close". Known because
                                       # we command open at startup and only we change it after.
```

- [ ] **Step 3: Command `open` at startup so the tracked state is real**

Replace the gripper connect block (lines 137-147):

```python
# ---------- Hand-E gripper (best-effort: viz still works if it's unreachable) ----------
try:
    gripper: hande_gripper.HandEGripper | None = hande_gripper.HandEGripper(
        GRIPPER_HOST, GRIPPER_PORT
    )
    gripper.connect()
    gripper.activate()
    print("Hand-E gripper connected + activated.")
except Exception as e:
    gripper = None
    print(f"Hand-E gripper unavailable ({e}); running viz-only gripper.")
```

with (adds the startup `open()` so `gripper_state == "open"` is guaranteed true):

```python
# ---------- Hand-E gripper (best-effort: viz still works if it's unreachable) ----------
try:
    gripper: hande_gripper.HandEGripper | None = hande_gripper.HandEGripper(
        GRIPPER_HOST, GRIPPER_PORT
    )
    gripper.connect()
    gripper.activate()
    gripper.open()   # known initial state: open => gripper_state ("open") matches reality
    print("Hand-E gripper connected + activated + opened.")
except Exception as e:
    gripper = None
    print(f"Hand-E gripper unavailable ({e}); running viz-only gripper.")
```

- [ ] **Step 4: Verify syntax**

Run: `./robot_control/bin/python -m py_compile teleop_ur15.py && echo OK`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add teleop_ur15.py
git commit -m "UR15: track gripper_state as one value, command open at startup"
```

---

### Task 2: UR15 — remove the grip dropdown; record state on capture/add (spec A)

**Files:**
- Modify: `teleop_ur15.py` (GUI defs ~line 250; `_grip_choice` ~line 274; add/capture handlers ~line 293; plan fallback ~line 345; gripper buttons ~line 407)

- [ ] **Step 1: Remove the dropdown GUI element**

Delete this line (line 250):

```python
gui_grip_action = server.gui.add_dropdown("Gripper @ waypoint", ("none", "open", "close"))
```

- [ ] **Step 2: Remove `_grip_choice()` and read `gripper_state` directly**

Delete this function (lines 274-275):

```python
def _grip_choice() -> str | None:
    return None if gui_grip_action.value == "none" else gui_grip_action.value
```

- [ ] **Step 3: Record `gripper_state` in the gizmo-add handler**

In the `@gui_add_wp.on_click` handler (lines 293-301), replace `"grip": _grip_choice(),` with `"grip": gripper_state,`. The handler becomes:

```python
@gui_add_wp.on_click
def _(_):
    _add_waypoint({
        "q": None,  # filled by IK at Plan
        "pos": np.asarray(gizmo.position).tolist(),
        "wxyz": np.asarray(gizmo.wxyz).tolist(),
        "grip": gripper_state,
    })
    gui_status.value = f"Added waypoint {len(waypoints)} (gizmo)"
```

- [ ] **Step 4: Record `gripper_state` in the capture handler**

In the `@gui_capture.on_click` handler (lines 304-315), replace `"grip": _grip_choice(),` with `"grip": gripper_state,`. The handler becomes:

```python
@gui_capture.on_click
def _(_):
    with state_lock:
        q = current_q.copy()
    T = grasp_pose(q)
    _add_waypoint({
        "q": q.tolist(),  # taught joints — replayed exactly, no IK
        "pos": np.asarray(T.translation()).tolist(),
        "wxyz": np.asarray(T.rotation().wxyz).tolist(),
        "grip": gripper_state,
    })
    gui_status.value = f"Captured waypoint {len(waypoints)} (joints)"
```

- [ ] **Step 5: Fix the no-waypoints Plan fallback**

In the `@gui_plan.on_click` handler, the single-target fallback (lines 344-346) sets `"grip": None`. Replace `"grip": None}]` with `"grip": gripper_state}]` so the fallback target carries the current state. The fallback line becomes:

```python
        targets = [{"q": None, "pos": np.asarray(gizmo.position).tolist(),
                    "wxyz": np.asarray(gizmo.wxyz).tolist(), "grip": gripper_state}]
```

- [ ] **Step 6: Make the Open/Close buttons update `gripper_state`**

Replace the `_actuate_gripper` function and its two button handlers (lines 386-418):

```python
def _actuate_gripper(target_finger: float, action: str) -> None:
    """Command the real gripper (if connected), then tween the viz fingers to match."""
    global gripper_finger
    if gripper is not None:
        try:
            getattr(gripper, action)()
        except Exception as e:
            gui_grip_status.value = f"cmd failed: {e}"
            return
    start = gripper_finger
    steps = max(1, int(GRIPPER_TWEEN_S * PLAY_HZ))
    for i in range(1, steps + 1):
        gripper_finger = start + (target_finger - start) * i / steps
        time.sleep(1.0 / PLAY_HZ)
    gripper_finger = target_finger
    if gripper is not None:
        gui_grip_status.value = "object grasped" if gripper.status()["object"] else (
            "open" if target_finger > 0 else "closed"
        )


@gui_grip_open.on_click
def _(_):
    global gripper_state
    gripper_state = "open"
    threading.Thread(
        target=_actuate_gripper, args=(GRIPPER_FINGER_OPEN, "open"), daemon=True
    ).start()


@gui_grip_close.on_click
def _(_):
    global gripper_state
    gripper_state = "close"
    threading.Thread(
        target=_actuate_gripper, args=(0.0, "close_gripper"), daemon=True
    ).start()
```

- [ ] **Step 7: Verify no orphaned references remain**

Run: `grep -n "gui_grip_action\|_grip_choice" teleop_ur15.py`
Expected: no output (both symbols fully removed).

Run: `./robot_control/bin/python -m py_compile teleop_ur15.py && echo OK`
Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add teleop_ur15.py
git commit -m "UR15: drop grip dropdown; capture records live gripper_state"
```

---

### Task 3: UR15 — replay actuates on change, with a 0.5 s settle (spec B)

**Files:**
- Modify: `teleop_ur15.py` (`_play`, lines 439-498)

- [ ] **Step 1: Track `cur_grip` and gate actuation on a state change**

In `_play`, the function already starts (line 440) with `global gripper_finger`. Replace that line with:

```python
    global gripper_finger, gripper_state
```

Then, just before the segment loop `for seg_idx, (q_start, q_goal, grip) in enumerate(plan_segments):` (line 456), add a line initialising the running grip state from the live value:

```python
        cur_grip = gripper_state   # running gripper state; only actuate when a waypoint differs
```

(Place it inside the `try:` at line 455, before the `for` on line 456, at the same indentation as the `for`.)

- [ ] **Step 2: Replace the per-waypoint grip block with on-change + 0.5 s settle**

Replace the grip block (lines 477-498), which currently fires whenever `grip in ("open","close")`:

```python
            # gripper action recorded at this waypoint (hold the arm with servoJ
            # while the fingers tween; real command only on an executed play)
            if not stop_flag.is_set() and grip in ("open", "close"):
                gui_status.value = f"Gripper: {grip}"
                target_f = GRIPPER_FINGER_OPEN if grip == "open" else 0.0
                if execute and gripper is not None:
                    try:
                        getattr(gripper, "open" if grip == "open" else "close_gripper")()
                    except Exception as e:
                        gui_grip_status.value = f"cmd failed: {e}"
                start_f = gripper_finger
                steps = max(1, int(GRIPPER_TWEEN_S * STREAM_HZ))
                for k in range(1, steps + 1):
                    if stop_flag.is_set():
                        break
                    gripper_finger = start_f + (target_f - start_f) * k / steps
                    if execute:
                        rtde_c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, SERVO_LOOKAHEAD, SERVO_GAIN)
                    viser_urdf.update_cfg(q_goal)
                    _update_gripper_viz(q_goal)
                    time.sleep(dt)
                gripper_finger = target_f
```

with (actuate only on a change; settle GRIP_PREDELAY_S first, holding the pose):

```python
            # gripper action: only when this waypoint's state differs from the
            # running state. Settle the arm GRIP_PREDELAY_S first (keep streaming
            # the pose), then actuate. Real command only on an executed play.
            if not stop_flag.is_set() and grip in ("open", "close") and grip != cur_grip:
                gui_status.value = f"Settling before gripper: {grip}"
                for _ in range(int(GRIP_PREDELAY_S * STREAM_HZ)):
                    if stop_flag.is_set():
                        break
                    if execute:
                        rtde_c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, SERVO_LOOKAHEAD, SERVO_GAIN)
                    time.sleep(dt)

                gui_status.value = f"Gripper: {grip}"
                target_f = GRIPPER_FINGER_OPEN if grip == "open" else 0.0
                if execute and gripper is not None:
                    try:
                        getattr(gripper, "open" if grip == "open" else "close_gripper")()
                    except Exception as e:
                        gui_grip_status.value = f"cmd failed: {e}"
                start_f = gripper_finger
                steps = max(1, int(GRIPPER_TWEEN_S * STREAM_HZ))
                for k in range(1, steps + 1):
                    if stop_flag.is_set():
                        break
                    gripper_finger = start_f + (target_f - start_f) * k / steps
                    if execute:
                        rtde_c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, SERVO_LOOKAHEAD, SERVO_GAIN)
                    viser_urdf.update_cfg(q_goal)
                    _update_gripper_viz(q_goal)
                    time.sleep(dt)
                gripper_finger = target_f
                gripper_state = grip   # keep the global in sync with what we just commanded
                cur_grip = grip
```

- [ ] **Step 3: Verify syntax**

Run: `./robot_control/bin/python -m py_compile teleop_ur15.py && echo OK`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add teleop_ur15.py
git commit -m "UR15: replay actuates gripper only on change, with 0.5s settle"
```

- [ ] **Step 5: [MANUAL — robot] Teach + replay smoke test**

On the Oracle PC: run `./robot_control/bin/python teleop_ur15.py`. Free-drive to a pose, click **Open gripper**, **Capture**; move, click **Close gripper**, **Capture**; **Save trajectory** as `griptest`. Confirm `trajectories/griptest.json` has `"grip": "open"` then `"grip": "close"` on the two waypoints. **Load**, **Plan**, **Play** with Execute on; verify the arm settles ~0.5 s at the second waypoint before the gripper closes, and that the gripper does not re-fire at waypoints whose state is unchanged.

---

### Task 4: CLI — argument parsing, JSON load/validate, plan + dry-run (spec D)

**Files:**
- Create: `play_trajectory.py`
- Create (test fixture): `trajectories/_sample_ur15.json`, `trajectories/_sample_gofa.json`

- [ ] **Step 1: Write the CLI core (no transports yet)**

Create `play_trajectory.py`:

```python
"""Headless replay of a saved teach trajectory on the UR15 or GoFa.

Reads trajectories/<name>.json, auto-detects the robot from its "robot" field,
and replays the stored joint waypoints on the real arm (after a confirm prompt)
-- no viser. IK-solver-free: every waypoint must already carry "q" (from Capture
or Plan-and-save in viser). The GoFa path imports pyroki for forward kinematics
only, to enforce the MAX_TCP_SPEED collaborative cap.

  ./robot_control/bin/python play_trajectory.py <name> [--speed S] [--dry-run] [--no-confirm]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ_DIR = os.path.join(_HERE, "trajectories")

# ---- profile (mirrors the teleop scripts) ----
RAMP_FRAC = 0.25
MIN_SEG_DURATION_S = 0.5
DWELL_S = 0.2
GRIP_PREDELAY_S = 0.5

# ---- UR15 ----
UR_ROBOT_IP = "192.168.125.2"
UR_MAX_JOINT_SPEED = 1.0
UR_STREAM_HZ = 50
UR_SERVO_LOOKAHEAD = 0.1
UR_SERVO_GAIN = 300
UR_SERVO_STOP_DECEL = 2.0
UR_SETTLE_GAIN = 600
UR_SETTLE_EPS_RAD = 0.00002
UR_SETTLE_STALL_TICKS = 10
UR_SETTLE_MAX_S = 3.0
UR_GRIPPER_FINGER_OPEN = 0.025
UR_GRIPPER_MASS = 1.0
UR_GRIPPER_COG = (0.0, 0.0, 0.06)

# ---- GoFa ----
GOFA_ROBOT_IP = "192.168.125.1"
GOFA_RWS_USER = "Default User"
GOFA_RWS_PASSWORD = "robotics"
GOFA_RAPID_MODULE = "PyEgm"
GOFA_RAPID_GO_FLAG = "egm_go"
GOFA_EGM_LOCAL_PORT = 6510
GOFA_MAX_JOINT_SPEED = 1.0
GOFA_MAX_TCP_SPEED = 0.25
GOFA_STREAM_HZ = 100
GOFA_HOLD_AFTER_PLAY_S = 1.5
GOFA_URDF_PATH = os.path.join(_HERE, "crb15000_5_95.urdf")
GOFA_MESH_DIR_PREFIX = os.path.join(_HERE, "abb_desc")


def alpha_to_s(alpha: float, r: float = RAMP_FRAC) -> float:
    """Trapezoidal velocity profile: alpha in [0,1] -> traversed fraction in [0,1]."""
    v_peak = 1.0 / (1.0 - r)
    if alpha < r:
        return 0.5 * v_peak * alpha * alpha / r
    if alpha < 1.0 - r:
        return 0.5 * v_peak * r + v_peak * (alpha - r)
    return 1.0 - 0.5 * v_peak * (1.0 - alpha) ** 2 / r


def load_trajectory(name: str) -> dict:
    path = os.path.join(TRAJ_DIR, f"{name}.json")
    with open(path) as f:
        data = json.load(f)
    wps = data.get("waypoints", [])
    if not wps:
        raise SystemExit(f"{name}.json has no waypoints.")
    for i, wp in enumerate(wps):
        if wp.get("q") is None:
            raise SystemExit(
                f"waypoint {i} in {name}.json has no joints -- open '{name}' in viser, "
                f"Plan, and re-save (the CLI replays stored joints, it does not run IK)."
            )
    return data


def build_segments(q_start: np.ndarray, waypoints: list[dict]):
    """First segment goes from the robot's current joints to waypoint 1, then
    waypoint-to-waypoint. Each segment is (q_start, q_goal, grip)."""
    segments = []
    q = np.asarray(q_start, dtype=np.float64)
    for wp in waypoints:
        q_next = np.asarray(wp["q"], dtype=np.float64)
        segments.append((q.copy(), q_next, wp.get("grip")))
        q = q_next
    return segments


def estimate_duration(segments, max_joint_speed: float, speed: float) -> float:
    total = 0.0
    for q_start, q_goal, grip in segments:
        delta = q_goal - q_start
        seg = max(MIN_SEG_DURATION_S, float(np.max(np.abs(delta))) / max_joint_speed)
        total += seg / max(0.1, speed) + DWELL_S
        if grip in ("open", "close"):
            total += GRIP_PREDELAY_S
    return total


def print_plan(robot: str, segments, speed: float) -> None:
    mjs = UR_MAX_JOINT_SPEED if robot == "ur15" else GOFA_MAX_JOINT_SPEED
    print(f"Robot: {robot}")
    print(f"Segments: {len(segments)} (first = current pose -> waypoint 1)")
    for i, (a, b, grip) in enumerate(segments):
        dmax = float(np.max(np.abs(b - a)))
        tag = f"  grip->{grip}" if grip in ("open", "close") else ""
        print(f"  seg {i + 1}: max|dq|={np.degrees(dmax):6.1f} deg{tag}")
    print(f"Estimated duration: {estimate_duration(segments, mjs, speed):.1f} s "
          f"(speed={speed})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Headless trajectory replay (UR15 / GoFa).")
    ap.add_argument("name", help="trajectory name (trajectories/<name>.json)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed scale (default 1.0)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit, no motion")
    ap.add_argument("--no-confirm", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    data = load_trajectory(args.name)
    robot = data.get("robot")
    if robot not in ("ur15", "gofa"):
        raise SystemExit(f"unknown robot {robot!r} in {args.name}.json")

    if args.dry_run:
        # current pose unknown without a robot connection: show segments between
        # the stored waypoints (waypoint1->2->...), which is the bulk of the plan.
        wps = data["waypoints"]
        segs = build_segments(np.asarray(wps[0]["q"], dtype=np.float64), wps[1:])
        print("[dry-run] (move-to-start segment omitted; needs a live robot pose)")
        print_plan(robot, segs, args.speed)
        return

    if robot == "ur15":
        play_ur15(data, args.speed, args.no_confirm)
    else:
        play_gofa(data, args.speed, args.no_confirm)


def play_ur15(data, speed, no_confirm):
    raise SystemExit("play_ur15 not implemented yet")  # Task 5


def play_gofa(data, speed, no_confirm):
    raise SystemExit("play_gofa not implemented yet")  # Task 6


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create the sample fixtures**

Create `trajectories/_sample_ur15.json`:

```json
{
  "robot": "ur15",
  "created": "2026-06-03T00:00:00",
  "waypoints": [
    {"q": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0], "pos": [0.4, 0.0, 0.3], "wxyz": [1, 0, 0, 0], "grip": "open"},
    {"q": [0.3, -1.40, 1.40, -1.57, -1.57, 0.0], "pos": [0.5, 0.1, 0.3], "wxyz": [1, 0, 0, 0], "grip": "close"},
    {"q": [0.3, -1.40, 1.40, -1.57, -1.57, 0.0], "pos": [0.5, 0.1, 0.4], "wxyz": [1, 0, 0, 0], "grip": "close"}
  ]
}
```

Create `trajectories/_sample_gofa.json`:

```json
{
  "robot": "gofa",
  "created": "2026-06-03T00:00:00",
  "waypoints": [
    {"q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "pos": [0.4, 0.0, 0.4], "wxyz": [1, 0, 0, 0]},
    {"q": [0.2, 0.1, -0.1, 0.0, 0.2, 0.0], "pos": [0.45, 0.05, 0.4], "wxyz": [1, 0, 0, 0]}
  ]
}
```

- [ ] **Step 3: Dry-run the UR15 sample**

Run: `./robot_control/bin/python play_trajectory.py _sample_ur15 --dry-run`
Expected: prints `Robot: ur15`, a 2-segment plan (waypoint1->2->3) with `grip->close` on seg 1, and an estimated duration line. No errors.

- [ ] **Step 4: Dry-run the GoFa sample**

Run: `./robot_control/bin/python play_trajectory.py _sample_gofa --dry-run`
Expected: prints `Robot: gofa`, a 1-segment plan, estimated duration. No errors.

- [ ] **Step 5: Verify the q-missing guard**

Run: `./robot_control/bin/python -c "import json,os; d={'robot':'ur15','waypoints':[{'q':None,'pos':[0,0,0],'wxyz':[1,0,0,0],'grip':'open'}]}; open('trajectories/_sample_bad.json','w').write(json.dumps(d))"`
Run: `./robot_control/bin/python play_trajectory.py _sample_bad --dry-run; echo "exit=$?"`
Expected: error message containing `has no joints` and `exit=1`.
Then: `rm trajectories/_sample_bad.json`

- [ ] **Step 6: Commit**

```bash
git add play_trajectory.py trajectories/_sample_ur15.json trajectories/_sample_gofa.json
git commit -m "CLI: play_trajectory.py core (load, validate, plan, dry-run)"
```

---

### Task 5: CLI — UR15 execution path (spec D)

**Files:**
- Modify: `play_trajectory.py` (`play_ur15`)

- [ ] **Step 1: Implement `play_ur15`**

Replace the `play_ur15` stub:

```python
def play_ur15(data, speed, no_confirm):
    raise SystemExit("play_ur15 not implemented yet")  # Task 5
```

with:

```python
def play_ur15(data, speed, no_confirm):
    import hande_gripper
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    rtde_r = RTDEReceiveInterface(UR_ROBOT_IP)
    rtde_c = RTDEControlInterface(UR_ROBOT_IP)
    try:
        rtde_c.setPayload(UR_GRIPPER_MASS, list(UR_GRIPPER_COG))
    except Exception as e:
        print(f"setPayload failed ({e}); end-of-play droop may be larger.")

    # gripper best-effort + known-open startup state (mirrors teleop_ur15.py)
    try:
        gripper = hande_gripper.HandEGripper(UR_ROBOT_IP, hande_gripper.DEFAULT_PORT)
        gripper.connect()
        gripper.activate()
        gripper.open()
        print("Hand-E connected + opened.")
    except Exception as e:
        gripper = None
        print(f"Hand-E unavailable ({e}); motion-only (grip actions skipped).")
    cur_grip = "open"

    q_now = np.array(rtde_r.getActualQ(), dtype=np.float64)
    segments = build_segments(q_now, data["waypoints"])
    print_plan("ur15", segments, speed)
    if not no_confirm and input("Execute on the real UR15? [y/N] ").strip().lower() != "y":
        print("Aborted."); return

    dt = 1.0 / UR_STREAM_HZ
    try:
        for seg_idx, (q_start, q_goal, grip) in enumerate(segments):
            delta = q_goal - q_start
            seg_duration = max(MIN_SEG_DURATION_S, float(np.max(np.abs(delta))) / UR_MAX_JOINT_SPEED)
            print(f"Segment {seg_idx + 1}/{len(segments)}")
            alpha = 0.0
            while alpha < 1.0:
                q = q_start + delta * alpha_to_s(alpha)
                rtde_c.servoJ(q.tolist(), 0.0, 0.0, dt, UR_SERVO_LOOKAHEAD, UR_SERVO_GAIN)
                time.sleep(dt)
                alpha = min(1.0, alpha + dt * speed / seg_duration)

            if grip in ("open", "close") and grip != cur_grip:
                for _ in range(int(GRIP_PREDELAY_S * UR_STREAM_HZ)):  # settle before actuating
                    rtde_c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, UR_SERVO_LOOKAHEAD, UR_SERVO_GAIN)
                    time.sleep(dt)
                if gripper is not None:
                    print(f"Gripper: {grip}")
                    getattr(gripper, "open" if grip == "open" else "close_gripper")()
                    time.sleep(0.8)  # let the fingers move
                cur_grip = grip

            if seg_idx < len(segments) - 1:  # inter-waypoint dwell
                for _ in range(int(max(0.0, DWELL_S / max(0.1, speed)) * UR_STREAM_HZ)):
                    rtde_c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, UR_SERVO_LOOKAHEAD, UR_SERVO_GAIN)
                    time.sleep(dt)

        # final settle: hold the last target until the measured joints arrive
        q_final = segments[-1][1]
        deadline = time.monotonic() + UR_SETTLE_MAX_S
        best, stalls = float("inf"), 0
        while True:
            rtde_c.servoJ(q_final.tolist(), 0.0, 0.0, dt, UR_SERVO_LOOKAHEAD, UR_SETTLE_GAIN)
            time.sleep(dt)
            err = float(np.max(np.abs(np.array(rtde_r.getActualQ()) - q_final)))
            if err < best - UR_SETTLE_EPS_RAD:
                best, stalls = err, 0
            else:
                stalls += 1
            if stalls >= UR_SETTLE_STALL_TICKS or time.monotonic() > deadline:
                print(f"[settle] final joint error {np.degrees(err):.3f} deg")
                break
    finally:
        rtde_c.servoStop(UR_SERVO_STOP_DECEL)
        rtde_c.stopScript()
    print("Done.")
```

- [ ] **Step 2: Verify syntax**

Run: `./robot_control/bin/python -m py_compile play_trajectory.py && echo OK`
Expected: `OK`

- [ ] **Step 3: Confirm dry-run still works (no regression)**

Run: `./robot_control/bin/python play_trajectory.py _sample_ur15 --dry-run`
Expected: same 2-segment plan as Task 4 Step 3 (the new code only runs on a non-dry-run UR15 play).

- [ ] **Step 4: Commit**

```bash
git add play_trajectory.py
git commit -m "CLI: UR15 execution path (servoJ + settle + gripper on change)"
```

- [ ] **Step 5: [MANUAL — robot] Replay the taught UR15 trajectory headless**

On the Oracle PC, with the UR15 in Remote Control: `./robot_control/bin/python play_trajectory.py griptest` (the trajectory taught in Task 3 Step 5). Confirm the prompt, then verify: arm moves from its current pose to waypoint 1, runs the segments, settles ~0.5 s before the gripper closes, and stops cleanly at the end.

---

### Task 6: CLI — GoFa execution path (spec D)

**Files:**
- Modify: `play_trajectory.py` (`play_gofa`)

- [ ] **Step 1: Implement `play_gofa` (imports pyroki for the TCP-speed cap)**

Replace the `play_gofa` stub:

```python
def play_gofa(data, speed, no_confirm):
    raise SystemExit("play_gofa not implemented yet")  # Task 6
```

with:

```python
def play_gofa(data, speed, no_confirm):
    import jax.numpy as jnp
    import jaxlie
    import pyroki as pk
    import yourdfpy

    import abb_egm
    import abb_rws

    def _resolve_mesh(fname):
        if fname.startswith("package://"):
            pkg, rest = fname[len("package://"):].split("/", 1)
            return os.path.join(GOFA_MESH_DIR_PREFIX, pkg, rest)
        return fname

    urdf = yourdfpy.URDF.load(GOFA_URDF_PATH, filename_handler=_resolve_mesh)
    robot = pk.Robot.from_urdf(urdf)
    tcp_idx = robot.links.names.index("tool0")

    def tcp_xyz(q):
        Ts = robot.forward_kinematics(cfg=jnp.array(q))
        return np.asarray(jaxlie.SE3(Ts[tcp_idx]).translation())

    def cap_seg_duration(q_start, delta, seg_duration, dt):
        """Stretch seg_duration so peak TCP speed stays <= GOFA_MAX_TCP_SPEED."""
        alpha, prev_p, peak = 0.0, tcp_xyz(q_start), 0.0
        while alpha < 1.0:
            alpha = min(1.0, alpha + dt / seg_duration)
            p = tcp_xyz(q_start + delta * alpha_to_s(alpha))
            peak = max(peak, float(np.linalg.norm(p - prev_p)) / dt)
            prev_p = p
        if peak > GOFA_MAX_TCP_SPEED:
            seg_duration *= peak / GOFA_MAX_TCP_SPEED
        return seg_duration

    rws = abb_rws.RWSClient(host=GOFA_ROBOT_IP, user=GOFA_RWS_USER, password=GOFA_RWS_PASSWORD)
    rws.request_mastership()
    rws.set_rapid_bool(GOFA_RAPID_GO_FLAG, False, module=GOFA_RAPID_MODULE)
    egm = abb_egm.EGMSession(local_port=GOFA_EGM_LOCAL_PORT)
    egm.start()

    q_now = np.array(rws.get_joints(), dtype=np.float64)
    segments = build_segments(q_now, data["waypoints"])
    print_plan("gofa", segments, speed)
    if not no_confirm and input("Execute on the real GoFa? [y/N] ").strip().lower() != "y":
        print("Aborted."); return

    dt = 1.0 / GOFA_STREAM_HZ

    def start_egm():
        egm.set_target_rad(q_now.tolist())
        rws.set_rapid_bool(GOFA_RAPID_GO_FLAG, True, module=GOFA_RAPID_MODULE)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if egm.is_fresh(max_age_s=0.1):
                return True
            time.sleep(0.05)
        return False

    if not start_egm():
        print("EGM did not start (no packets in 3s). Is PyEgm parked at WaitUntil egm_go?")
        return
    try:
        for seg_idx, (q_start, q_goal, _grip) in enumerate(segments):
            delta = q_goal - q_start
            seg_duration = max(MIN_SEG_DURATION_S, float(np.max(np.abs(delta))) / GOFA_MAX_JOINT_SPEED)
            seg_duration = cap_seg_duration(q_start, delta, seg_duration, dt)
            print(f"Segment {seg_idx + 1}/{len(segments)}")
            alpha = 0.0
            while alpha < 1.0:
                q = q_start + delta * alpha_to_s(alpha)
                egm.set_target_rad(q.tolist())
                time.sleep(dt)
                alpha = min(1.0, alpha + dt * speed / seg_duration)
            if seg_idx < len(segments) - 1:
                for _ in range(int(max(0.0, DWELL_S / max(0.1, speed)) * GOFA_STREAM_HZ)):
                    egm.set_target_rad(q_goal.tolist())
                    time.sleep(dt)

        hold = segments[-1][1]
        for _ in range(int(GOFA_HOLD_AFTER_PLAY_S * GOFA_STREAM_HZ)):  # let \CondTime fire
            egm.set_target_rad(hold.tolist())
            time.sleep(dt)
    finally:
        try:
            rws.set_rapid_bool(GOFA_RAPID_GO_FLAG, False, module=GOFA_RAPID_MODULE)
        except Exception:
            pass
    print("Done.")
```

- [ ] **Step 2: Verify syntax**

Run: `./robot_control/bin/python -m py_compile play_trajectory.py && echo OK`
Expected: `OK`

- [ ] **Step 3: Confirm both dry-runs still work**

Run: `./robot_control/bin/python play_trajectory.py _sample_gofa --dry-run && ./robot_control/bin/python play_trajectory.py _sample_ur15 --dry-run`
Expected: both plans print without error.

- [ ] **Step 4: Commit**

```bash
git add play_trajectory.py
git commit -m "CLI: GoFa execution path (EGM stream + TCP-speed cap via FK)"
```

- [ ] **Step 5: [MANUAL — robot] Replay a taught GoFa trajectory headless**

On the Oracle PC, GoFa in Auto with PyEgm parked at `WaitUntil egm_go`: teach + save a trajectory in `teleop_gofa_egm.py` (e.g. `gofatest`), Ctrl+C it (free the mastership), then `./robot_control/bin/python play_trajectory.py gofatest`. Confirm the prompt; verify it moves to waypoint 1, runs the segments under the TCP cap, holds at the end, and the session closes cleanly.

---

### Task 7: Documentation + sample-fixture cleanup decision

**Files:**
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: Document the gripper-state model in CLAUDE.md**

In `CLAUDE.md`, in the Hand-E "Control path" / tunables area, replace the description of the per-waypoint dropdown with the tracked-state model. Add this paragraph after the "URCap socket protocol" subsection (search for the Hand-E tunables table):

```markdown
**Gripper state is one tracked value.** `gripper_state` (`"open"`/`"close"`) is the
single source of truth; the viz fingers always render it. At startup the script sends
the real gripper an `open` command, so `gripper_state` is known without polling — only
the Open/Close buttons and trajectory replay change it after that. **Capture/Add
records the current `gripper_state` into the waypoint automatically** (the old "Gripper
@ waypoint" dropdown is gone). On replay, the gripper actuates only when a waypoint's
state differs from the running state, and the arm settles `GRIP_PREDELAY_S` (0.5 s)
before each actuation.
```

Add a `GRIP_PREDELAY_S` row to the Hand-E tunables table:

```markdown
| `GRIP_PREDELAY_S` | `0.5` s | Settle hold before the gripper actuates at a waypoint |
```

- [ ] **Step 2: Document the CLI player in CLAUDE.md**

Add a new section after the "Free-drive teach & saved trajectories" section:

```markdown
## Headless replay — `play_trajectory.py`

Replay a saved trajectory on either arm without viser:

\`\`\`bash
./robot_control/bin/python play_trajectory.py <name> [--speed S] [--dry-run] [--no-confirm]
\`\`\`

Reads `trajectories/<name>.json`, auto-detects the robot from its `"robot"` field, and
executes on the real arm after a `[y/N]` confirm (`--no-confirm` to skip). `--dry-run`
prints the plan (segments + estimated duration) and exits without touching hardware.
It is **IK-solver-free**: every waypoint must already carry `"q"` (from Capture, or
Plan-and-save in viser) — a `q`-less waypoint aborts with a "Plan + re-save" message.
The first segment moves the arm from its current pose to waypoint 1 (same as viser).
The UR15 path mirrors `teleop_ur15.py` (servoJ + settle + gripper-on-change with the
0.5 s pre-delay, gripper opened at start). The GoFa path imports pyroki for forward
kinematics **only** to enforce the `MAX_TCP_SPEED` collaborative cap, then streams over
the existing EGM supervisor (PyEgm must be parked at `WaitUntil egm_go`).
```

- [ ] **Step 3: Add a README bullet**

In `README.md`, near the run instructions, add:

```markdown
- **`play_trajectory.py <name>`** — headless replay of a saved trajectory (UR15 or GoFa),
  no viser. `--dry-run` to preview, `--no-confirm` to skip the prompt.
```

- [ ] **Step 4: Decide the sample fixtures' fate**

The `trajectories/_sample_*.json` files were test fixtures. Keep them (underscore-prefixed, harmless, useful as `--dry-run` examples) — leave as committed. No action needed; this step documents the decision.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "Docs: gripper-state model + headless play_trajectory.py"
```

- [ ] **Step 6: Push**

```bash
git push
```

---

## Self-review notes

- **Spec coverage:** A → Tasks 1-2; B → Task 3; C → no code (already exists), noted in Task 6 manual prereq; D core → Task 4, D UR15 → Task 5, D GoFa → Task 6 (FK cap preserved per the spec correction); docs → Task 7.
- **Naming consistency:** `gripper_state`, `cur_grip`, `GRIP_PREDELAY_S`, `alpha_to_s`, `build_segments`, `print_plan`, `estimate_duration`, `cap_seg_duration`, `play_ur15`, `play_gofa` used consistently across tasks.
- **Profile parity:** CLI `alpha_to_s` and the seg-duration rule mirror the teleop constants exactly (UR_/GOFA_ prefixes); drift risk noted in the spec (D1).
- **No silent caps:** the GoFa TCP cap is preserved in the CLI (Task 6), not dropped.
