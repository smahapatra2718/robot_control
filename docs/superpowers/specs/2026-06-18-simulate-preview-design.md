# Simulate / preview in the 3D view — design

**Date:** 2026-06-18
**Status:** approved, pre-implementation

## Goal

Add a **Simulate** button next to the console's Teleop *Send* and Play *▶* buttons
that previews the motion in the 3D viewer **without commanding the robot**. The
preview is a translucent "ghost" arm that animates the exact path the real motion
would follow, shown alongside the solid live arm.

The preview must be **faithful** for all three motions — move/joints, move/pose,
and play — which means the joint path is computed **server-side** (the browser has
no IK; pose/play interpolate in Cartesian space and re-solve seeded IK each step).

## Decisions (locked during brainstorming)

- **Preview style:** translucent ghost overlay — a semi-transparent clone animates
  the path while the solid live arm stays at its real (telemetry) pose.
- **Faithfulness:** server computes the joint path by reusing the real motion
  planner (same `_build_segments` / `_ik` / `_cartesian_q` / `alpha_to_s`). Only
  playback *timing* is approximated in the browser.
- **No lease:** preview endpoints are auth-only and never move hardware.
- **Idle-only:** preview `409`s while a motion command is running (avoids a
  mid-motion start pose and concurrent IK).
- **Scope:** Simulate buttons on Teleop (Joints/Pose target) and Play. Ghost is
  arm-only — the Hand-E gripper is not ghosted.

## Architecture

### 1. Controller — `simulate()` (base, hardware-free)

Add to `lib/control/base.py`:

```python
def simulate(self, segments, interp="cartesian", step_rad=0.05, max_per_seg=120):
    """Compute-only twin of _run_play: sample the joint path the motion would
    follow, without touching hardware. Reuses _cartesian_q (straight tool line +
    seeded IK) / joint lerp and the alpha_to_s easing — identical geometry to the
    real motion. Returns a list of joint vectors (lists of floats)."""
```

For each `(q_start, q_goal, _grip)` segment it builds the same per-tick generator
`at` as `_run_play` (`q_start + delta*s` for `interp="joint"`, else
`self._cartesian_q(q_start, q_goal)`), chooses a sample count `N` from joint travel
(`max(2, min(max_per_seg, ceil(max|Δq| / step_rad)))`), and appends
`at(alpha_to_s(i/N))` for `i in 0..N`. Sample count affects only smoothness, not
geometry, so this is robot-agnostic (no `seg_duration` / TCP-cap needed — those set
timing, which the viewer approximates).

This lives entirely in the base: it is hardware-free and reuses base helpers, so no
subclass (`ur.py` / `gofa.py`) changes.

### 2. API — three lease-free preview endpoints

Add to `lib/robot_api.py`, mirroring the command endpoints:

- `POST /preview/move/joints` — body `{q:[6]}`; validates with `_check_joints`;
  segments `[(read_q, q, None)]`, `interp="joint"`.
- `POST /preview/move/pose` — body `{pos:[3], wxyz:[4]}`; validates with
  `_check_pose`; `q_goal = controller._ik(pos, wxyz, read_q)`; segment
  `[(read_q, q_goal, None)]`, `interp="cartesian"`.
- `POST /preview/play` — body `{name | names | waypoints}` (same shapes as `/play`,
  validated the same way); `segments = controller._build_segments(load_waypoints(...))`.

Each calls `controller.simulate(segments, interp)` and returns
`200 {"path": [[6 floats], …]}`. **Auth required, no `X-Lease`.** Returns `409` when
`controller.get_state().activity in ("moving", "playing")` — i.e. a motion is in
progress (free-drive/idle are fine). `422` on a bad vector/pose; `400` on a
bad/missing trajectory name (same mapping as the real endpoints).

Building segments reuses the same private methods the real commands call
(`_read_q`, `_ik`, `_build_segments`, `_load_waypoints`), so the start pose is the
live `_read_q()` — identical to what the real command would plan from.

### 3. Viewer — translucent ghost + playback (`web/dashboard.html`)

Extend the `window.viewer3d` module:

- **Ghost model:** lazily, on the first `simulate()` for the current arm, load a
  second URDF instance via the existing `makeLoader()` (the `.glb`s are
  browser-cached, so the second load is cheap and reliable — a fresh URDFRobot with
  a working `setJointValue`). Traverse it: clone each mesh material, set
  `transparent=true, opacity=0.35, depthWrite=false`, tint toward the accent color.
  Add it to the scene at the same base frame, `visible=false`.
- **`viewer3d.simulate(path)`:** if no renderer/model, no-op. Show the ghost, set a
  `simulating` flag, and step an index through `path` with `requestAnimationFrame`,
  calling `ghost.setJointValue(name, path[i][j])` for the arm's joint names. Total
  duration is a smooth few seconds, scaled by the Speed slider value. On completion
  (or a new simulate call superseding it), hide the ghost and clear the flag.
- A **"SIMULATING"** badge in the 3D panel while the flag is set (reuse the existing
  `viewer3dNote` overlay, shown non-blocking).
- The solid live arm keeps tracking telemetry the whole time (the ghost is separate),
  so real vs. simulated show together.

### 4. Frontend — buttons + handlers

- **Teleop:** a `Simulate` button beside `#teleopSend`. Builds the same body as
  `teleopSend()` for the active mode (joints → `{q}`, pose → `{pos,wxyz}`) and
  POSTs to `/preview/move/joints` or `/preview/move/pose`, then
  `viewer3d.simulate(resp.path)`.
- **Play:** a `Simulate` button beside `#playBtn`. Builds the same body as `play()`
  (`{names}` for a chain, else `{name}`, else inline `{waypoints}`) and POSTs to
  `/preview/play`, then animates the returned path.
- Both use a new `preview(path, body)` helper calling `api(path, {method:"POST",
  body})` — **no `lease:true`** (read-only). Available without a lease.
- The console logs the preview (e.g. "preview: N samples") to the command log.

### 5. Error handling

Strictly non-blocking. A failed preview (409 busy, 422/400 validation, IK error, or
a viewer that never initialized) logs to the console and does nothing else — no arm
motion is ever involved, and the rest of the console is unaffected.

## Testing

- **Python** (`scripts/api_smoketest.py`, in-process TestClient):
  - `POST /preview/move/joints` with a valid `q` (no lease) → `200`, `path` non-empty,
    each sample length 6, **last sample ≈ the goal `q`**.
  - `POST /preview/move/pose` with a valid pose → `200`, `path` non-empty, last
    sample's FK ≈ the requested pose (or last sample ≈ the IK of the pose).
  - `POST /preview/play` by `name` → `200`, `path` non-empty.
  - Bad vector/pose → `422`; bad trajectory name → `400`; no auth → `401`.
  - While a long move is running → `/preview/*` returns `409`.
- **JS:** `node --check` the extracted viewer module.
- **Docs:** CLAUDE.md (Remote API endpoint list + console description), API.md
  (the three `/preview/*` endpoints, auth-only/no-lease, 409-when-busy), usage.html.
- The in-browser ghost animation is the user's visual confirmation (cannot be
  verified headlessly).

## Out of scope

- Ghosting the Hand-E gripper (arm-only preview).
- Previewing while a motion is in progress (idle-only by design).
- Exact timing fidelity (playback rate is a smooth approximation, not the real
  `seg_duration`/TCP-cap timing).
- Scrubbing / pausing the preview timeline (plays once, then hides).
