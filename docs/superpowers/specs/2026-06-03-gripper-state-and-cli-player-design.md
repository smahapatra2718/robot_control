# Gripper state tracking + headless trajectory player — design

**Date:** 2026-06-03
**Goal:** Make the UR15 gripper state a single tracked source of truth (saved into
trajectories, reflected in the sim, settled before actuation), and add a headless
CLI to replay saved trajectories on either arm without spinning up viser.

## Decisions (from brainstorming)

- **Gripper state is one tracked value**, not a per-waypoint dropdown. Removed the
  "Gripper @ waypoint" dropdown.
- **Known initial state via an `open` command at startup** — no polling of the
  hardware; we track state from our own commands thereafter.
- **0.5 s settle hold before every actual gripper actuation.**
- **CLI player executes on real hardware by default** (with a confirm prompt),
  auto-detects the robot from the JSON, and is **IK-free** (replays stored joints).
- **D1:** CLI is self-contained; the two working teleop scripts are not refactored.

## A. Gripper state = single source of truth (`teleop_ur15.py`)

Replace the implicit viz-only `gripper_finger` flow + the `"grip"` dropdown with one
tracked discrete state plus its viz mapping:

- `gripper_state: str` — `"open"` or `"close"`. The existing `gripper_finger` (metres)
  stays as the *rendered* value; it is always derived from / tweened toward
  `gripper_state` (`open` -> `GRIPPER_FINGER_OPEN`, `close` -> `0.0`).
- **Startup:** after `gripper.connect()` + `activate()`, send `gripper.open()`, set
  `gripper_state = "open"`, `gripper_finger = GRIPPER_FINGER_OPEN`. If the gripper is
  unavailable (viz-only), still initialise `gripper_state = "open"`.
- **Open/Close buttons:** set `gripper_state` and tween `gripper_finger` (real command
  only if connected) — unchanged behaviour, but now they update the tracked state.
- **Capture / Add waypoint:** record `wp["grip"] = gripper_state` automatically. The
  `gui_grip_action` dropdown and `_grip_choice()` are **removed**.
- The viz fingers are driven by `gripper_state` everywhere (`_update_gripper_viz`
  already renders `gripper_finger`), so the sim always reflects the gripper.

### Waypoint model

```
Waypoint = { "q": [j1..j6] | None, "pos": [x,y,z], "wxyz": [w,x,y,z], "grip": "open"|"close" }
```

`grip` is now always present and absolute (the gripper state *at* that waypoint), no
longer `None`. No saved trajectories exist yet, so no migration is required; a loader
that meets a legacy `null`/missing `grip` treats it as `"open"` (no actuation when it
already matches the running state).

## B. Replay — actuate on change, with a 0.5 s settle

`plan_segments` stays `list[(q_start, q_goal, grip)]`. In `_play` and the CLI:

- Track a running `cur_grip` initialised to the startup state (`"open"`).
- At each waypoint, after the motion alpha-loop completes, **actuate only if
  `wp.grip != cur_grip`**. This avoids re-firing (and re-delaying) on identical
  consecutive states.
- When actuating: hold the arm at `q_goal` for `GRIP_PREDELAY_S = 0.5 s` (keep
  streaming the pose — `servoJ`/EGM hold — so the arm is settled), **then** command the
  fingers and tween the viz (`GRIPPER_TWEEN_S`). Update `cur_grip`.
- Real gripper command fires only on an executed play; viz tweens either way.

New constant: `GRIP_PREDELAY_S = 0.5` (seconds).

## C. GoFa

No change. Capture + Save/Load + taught-joint exact replay already exist
(`teleop_gofa_egm.py`, commit `9ccc8bf`). No gripper, so A/B do not apply. The CLI (D)
covers GoFa replay.

## D. Headless CLI player — `play_trajectory.py`

```
./robot_control/bin/python play_trajectory.py <name> [--speed S] [--dry-run] [--no-confirm]
```

- Loads `trajectories/<name>.json`; reads the `"robot"` field (`"ur15"` | `"gofa"`).
- **IK-solver-free:** replays stored joint configs only. If any waypoint has
  `q is None`, abort with: *"waypoint N has no joints — open <name> in viser, Plan,
  and re-save."* The UR15 path needs no robot model at all. The **GoFa path imports
  pyroki for forward kinematics only** — its `MAX_TCP_SPEED` collaborative cap is
  enforced by walking the path through FK (`_cap_seg_duration`), so dropping it would
  weaken the safety guarantee. No IK solve and no viser either way.
- **Plan:** same shape as the teleop scripts — first segment is `(current measured
  joints -> waypoint 1)`, then waypoint-to-waypoint. So it moves the arm to the start
  of the trajectory from wherever it currently is, at capped speed.
- **Profile:** self-contained copies of `alpha_to_s` (trapezoidal, `RAMP_FRAC`) and the
  per-segment duration rule (`max(MIN_SEG_DURATION_S, max|Δq|/MAX_JOINT_SPEED)`), scaled
  by `--speed` (default `1.0`). Values mirror the teleop constants.
- **Confirmation:** before moving, print robot, #waypoints, and estimated duration, then
  prompt `Execute on the real <robot>? [y/N]`. `--no-confirm` skips it. `--dry-run`
  prints the plan and exits without connecting to motion / moving.

### UR15 path

- `RTDEControlInterface` + `RTDEReceiveInterface`; `setPayload(GRIPPER_MASS, GRIPPER_COG)`.
- Gripper best-effort: `hande_gripper.HandEGripper(...).connect()/activate()`, then
  `open()` at start (mirrors A; sets `cur_grip = "open"`). If unavailable, log and
  continue motion-only (grip actions are skipped).
- Stream `servoJ` per tick along the profile; final settle loop (same plateau-detector
  logic / constants as `_play`); gripper actions per B; `servoStop` on exit.

### GoFa path

- `abb_rws.RWSClient` for mastership + `egm_go`; `abb_egm` UDP client.
- `egm_go = TRUE`, wait for first feedback, stream targets along the profile with the
  `MAX_TCP_SPEED` cap (reuse the teleop `_cap_seg_duration` rule), hold the final target
  `HOLD_AFTER_PLAY_S` so `\CondTime` closes the session. No gripper.

### Reuse vs duplication (D1)

Imports the already-modular transports directly: `hande_gripper`, `abb_rws`, `abb_egm`.
Duplicates only the small pure profile helpers and the relevant constants. The two
teleop scripts are **not** modified by the CLI work (they are by A/B). Risk: the
profile/constants could drift from the teleop scripts over time — acceptable for the
~30 lines involved; noted here so a future change keeps them in sync.

## Files touched

- `teleop_ur15.py` — A (gripper state, startup open, remove dropdown, capture records
  state) + B (on-change actuation + `GRIP_PREDELAY_S`).
- `play_trajectory.py` — new, headless player for both arms (D).
- `CLAUDE.md` / `README.md` — document the gripper-state model and the CLI.

## Out of scope / deferred

- Polling the real gripper position (explicitly rejected — startup `open` makes state
  deterministic without it).
- GoFa software lead-through and GoFa gripper (no hardware).
- CLI IK for gizmo-only (q-less) waypoints — Plan + re-save in viser instead.
