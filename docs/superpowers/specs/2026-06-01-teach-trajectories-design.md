# Free-drive teach + saved trajectories — design

**Date:** 2026-06-01
**Goal:** Hand-guide the arm to poses, capture them as waypoints, and save/load
trajectories to disk for later replay. (a.k.a. teach-by-demonstration.)

## Decisions (from brainstorming)

- **Waypoint stores both** joint angles and Cartesian EE pose.
- **Free-drive engaged by software toggle where possible:** UR15 via `ur_rtde`
  `teachMode()`; GoFa via its physical lead-through button for v1 (RWS has no
  clean lead-through toggle — programmatic would need a RAPID-supervisor
  addition + installer re-run; deferred).
- **Gripper actions recorded** per waypoint (UR15 only; GoFa has no gripper).
- Save/Load: trajectory-name text field + Save/Load buttons → `trajectories/<name>.json`.
- Editing: append / remove-last / clear only (no reorder/insert in v1).

## Data model

Replace the current `waypoints: list[(pos, wxyz)]` with dicts:

```
Waypoint = { "q": [j1..j6] | None, "pos": [x,y,z], "wxyz": [w,x,y,z], "grip": "open"|"close"|None }
```

- **Free-drive capture:** `q` = live joints, `pos/wxyz` = FK (grasp pose on UR15), `grip` = dropdown.
- **Gizmo add (existing flow):** `pos/wxyz` from gizmo, `q` = None; filled by IK at Plan and backfilled into the waypoint (so a saved-after-plan trajectory carries both).

## Playback

`plan_segments` becomes `list[(q_start, q_goal, grip)]`. In `_play`, at each
waypoint: if the waypoint has `q`, use it directly (faithful taught replay, no
IK); else IK from `pos/wxyz` (today's sequential-seeded behavior). After
reaching a waypoint, fire its `grip` action (UR15).

## UI additions

- **Free-drive** toggle — UR15: `teachMode()`/`endTeachMode()`. Mutually
  exclusive with Plan/Play/Live; Stop exits it. GoFa: button-driven (no toggle).
- **Capture waypoint** — snapshot current pose into the list + scene frame.
- **Gripper @ waypoint** dropdown (UR15): `none / open / close`, tagged onto each
  captured/added waypoint.
- **Trajectory name** text + **Save** / **Load** buttons.

## File format

`trajectories/<name>.json`:
```json
{ "robot": "ur15", "created": "<iso8601>", "waypoints": [ {q, pos, wxyz, grip}, ... ] }
```

## Implementation order

1. UR15 — full version (teachMode + gripper actions + save/load). *(testable)*
2. GoFa — capture + save/load; free-drive via hardware lead-through button; no gripper.

## Risks / deferred

- GoFa programmatic lead-through (RAPID-supervisor toggle) — deferred; hardware
  button for now.
- `teachMode` must be exited (`endTeachMode`) before any servoJ/Plan/Play, or the
  control mode conflicts. Stop and the toggle-off both call it.
