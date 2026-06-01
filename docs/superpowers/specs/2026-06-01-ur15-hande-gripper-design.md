# UR15 + Robotiq Hand-E gripper — design

**Date:** 2026-06-01
**Goal:** Mount a Robotiq Hand-E parallel gripper on the UR15 in `teleop_ur15.py`:
render the official mesh on the wrist, and add real open/close control.

## Decisions (from brainstorming)

- **Scope:** real gripper actuation **and** viz animation.
- **Wiring:** gripper on the UR wrist tool connector (RS-485 + 24 V through the
  flange); Grippers URCap currently installed.
- **Control path: A — RS-485 → TCP socket bridge.** Swap the Grippers URCap for
  Robotiq's RS485 URCap / Tool Communication Interface, exposing the tool RS-485
  as a background-daemon TCP socket. Python speaks Modbus RTU over it,
  independent of ur_rtde's resident control script. (Pendant gripper buttons are
  given up — accepted.)
  - **Update (implementation):** superseded by a simpler channel that needs no
    swap — the *Grippers* URCap already runs a daemon socket at
    `<robot_ip>:63352` (ASCII `SET`/`GET`), independent of the active program, so
    it coexists with ur_rtde and keeps the pendant buttons. `hande_gripper.py`
    uses that. Falls back to the RS485 bridge or a USB-RS485 adapter if 63352
    isn't reachable on PolyScope X.
- **TCP frame: grasp point.** The gizmo/IK target represents the gripper's grasp
  point, not the bare flange.

## Why not the obvious alternatives

- ur_rtde keeps its control script resident for the whole session, so a separate
  URCap gripper program can't run concurrently → control must be
  program-independent (hence the socket bridge).
- Merging Hand-E into the UR URDF for IK would put finger joints into the
  optimizer and risk the tuned seeded-IK. Avoided (see rendering below).

## Architecture

### Mesh & model
- Vendor `macmacal/robotiq_hande_description` (Apache-2.0; meshes from Robotiq's
  official STEP files) into the repo, like `abb_desc/`.
- Generate a standalone `hande.urdf` via `xacrodoc` (the existing GoFa toolchain),
  rewriting mesh paths to `package://` resolved by a `filename_handler`.

### Rendering — two viser models (IK untouched)
- **Arm:** existing `ViserUrdf(ur15, root="/world/base")`, unchanged.
- **Gripper:** second `ViserUrdf(hande, root="/world/gripper")`. Each viz/play
  tick, set the `/world/gripper` frame pose to the arm's current `tool0` pose
  (from FK) and `update_cfg` the two finger joints. Both live under `/world`, so
  the 30° display yaw is inherited. The UR15-only IK model is unchanged.

### Grasp-point TCP (fixed offset)
- The gripper is rigid, so the grasp point is a fixed `tool0_T_grasp` derived once
  from the gripper URDF (`hande_end` relative to the mount).
- Gizmo init/Reset: `FK(tool0) ∘ tool0_T_grasp`.
- Plan: convert gizmo pose → `tool0` target via `gizmo ∘ tool0_T_grasp⁻¹`, then
  run the **existing** seeded IK. ~3-line change; play loop unchanged.

### Gripper control — `HandEGripper` (isolated)
- One class wraps Modbus RTU over the tool-RS-485 socket; only place that knows
  the wire protocol.
- Registers (standard Robotiq): activate (`rACT`), then `rGTO`+`rPR` for position
  0 (open) … 255 (closed); `rSP`/`rFR` speed/force as constants; read
  `gOBJ`/`gPO` for status / object-detected.
- Transport: `pyserial` `socket://host:port` + a tiny RTU/CRC helper, behind the
  class so it can be swapped.
- Startup: verify socket, run activation once, expose a status flag.

### UI
- **Open** and **Close** buttons + a status text (activated / object-detected).
- Each click commands the real gripper in a worker thread and tweens the viz
  fingers to match. Manual only — not sequenced into waypoints (future work).

## Implementation order

1. Vendor `robotiq_hande_description`; generate `hande.urdf`; verify it loads in
   yourdfpy. *(no hardware)*
2. Two-model rendering + finger animation + grasp-point gizmo in
   `teleop_ur15.py`; verify in browser. *(no hardware)*
3. `HandEGripper` class + standalone verification probe — swap to the RS485/Tool-
   Comm socket and confirm activate + move from the Mac. **Gates the rest.**
4. Wire Open/Close buttons into `teleop_ur15.py`.

## Risks

- **PolyScope X tool-RS-485 socket exposure** differs from PolyScope 5 — resolved
  by the step-3 probe; the `HandEGripper` transport absorbs any difference.
- **Mount/coupler transform** in the macal repo targets UR e-series — confirm the
  ISO 50 mm flange offset matches the UR15 (likely identical).

## Source

- Gripper description: https://github.com/macmacal/robotiq_hande_description
