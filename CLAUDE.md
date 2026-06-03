# abb_foga — robot teleop scaffold

Browser-based teleop for two arms, sharing the same viser + pyroki stack:

- **`teleop_ur15.py`** — Universal Robots UR15 over RTDE (`ur_rtde`). Speed slider is unified: the same `q` is sent to both the viser preview and `rtde_c.servoJ()` each tick, so viz and robot move in lockstep.
- **`teleop_gofa_egm.py`** — ABB GoFa CRB 15000 (variant `crb15000_5_95`) over Externally Guided Motion (EGM), with RWS (`abb_rws.py`) for mastership and the start/stop flag. Joint targets stream over UDP at `STREAM_HZ` to a RAPID supervisor (`PyEgm.mod`) running `EGMRunJoint`, so the slider is unified like the UR15 — the same `q` drives both the viser preview and the robot. Real tool speed is held under `MAX_TCP_SPEED` regardless of slider/pose. EGM is a licensed controller option.

Both scripts share: the same UI (viser scene + gizmo + waypoints), the same IK (`pyroki_snippets._solve_ik_seeded`), the same trapezoidal-time alpha profile in the play loop, and the same auto-cleanup after a successful executed Play.

Run:

```bash
./robot_control/bin/python teleop_ur15.py   # or teleop_gofa_egm.py
```

Open the printed `http://localhost:8080`. Each script connects to its controller at startup and aborts if it can't reach.

## Project layout

```
abb_foga/
├── robot_control/             # Python venv (3.13)
├── pyroki_src/                # git clone of chungmin99/pyroki, installed -e
├── pyroki_snippets/           # copied from pyroki_src/examples/ + our _solve_ik_seeded.py
├── abb_desc/                  # clone of ros-industrial/abb (GoFa URDF + meshes)
├── crb15000_5_95.urdf         # generated from abb_desc/.../crb15000_5_95.xacro
├── robotiq_hande_description/ # vendored macmacal/robotiq_hande_description (Hand-E URDF + meshes)
├── hande.urdf                 # generated from robotiq_hande_description/.../robotiq_hande_gripper.urdf.xacro
├── hande_gripper.py           # minimal Hand-E Modbus-RTU-over-socket client
├── verify_hande.py            # one-shot gripper comms probe (run before trusting control)
├── teleop_ur15.py             # UR15 script (RTDE / servoJ) + Hand-E gripper
├── teleop_gofa_egm.py         # GoFa script (EGM joint streaming)
├── abb_rws.py                 # minimal RWS client for OmniCore (RobotWare 7+)
├── abb_egm.py                 # minimal EGM UDP client (protobuf wire format)
├── egm.proto / egm_pb2.py     # EGM protobuf schema + generated bindings
├── EGM_COMM.cfg / EGM_MOC.cfg # controller EGM config (uploaded by installer)
├── install_gofa_egm.py        # one-shot: uploads PyEgm.mod + .cfg, gets it running
└── CLAUDE.md                  # this file
```

## Dependencies (already installed in `robot_control/`)

Python: numpy, viser, yourdfpy, jaxlie, jax, jaxlib, robot_descriptions, xacrodoc, pyroki (editable), ur_rtde 1.6.3, requests, urllib3.

System (brew): cmake, **boost@1.85** (keg-only, for ur_rtde build only).

URDFs:
- **UR15**: loaded at runtime via `robot_descriptions.loaders.yourdfpy.load_robot_description("ur15_description")`. No local file.
- **Hand-E**: local `hande.urdf`, generated via `xacrodoc` from the vendored `robotiq_hande_description` (`xacrodoc.packages.look_in(["."]); XacroDoc.from_file(".../robotiq_hande_gripper.urdf.xacro").to_urdf_file("hande.urdf")`). Same `file://` → `package://` strip as the GoFa; resolved by a `filename_handler` in `teleop_ur15.py` with prefix `_HERE`. The `<ros2_control>` block (irrelevant to us) was stripped after generation. Rendered as a *second* `ViserUrdf` whose root frame is slaved to the live `tool0` pose — the UR IK model is untouched.
- **GoFa**: local `crb15000_5_95.urdf` (the 5 kg / 0.95 m reach variant — match this to your actual hardware nameplate), generated via `xacrodoc` from the ros-industrial repo: `xacrodoc.packages.look_in(["abb_desc"]); XacroDoc.from_file(".../crb15000_5_95.xacro").to_urdf_file(...)`. Mesh paths are rewritten to `package://abb_crb15000_support/...`; resolved via `URDF_MESH_DIR_PREFIX = os.path.join(_HERE, "abb_desc")` and a custom `filename_handler` in `teleop_gofa_egm.py`. xacrodoc emits absolute `file://` URIs (yourdfpy can't resolve those); strip the `file://.../abb_desc/` prefix down to `package://`.

---

# UR15

## Controller setup (one-time)

UR15 ships with Polyscope X, which firewalls external services by default. On the pendant:

1. **Settings → Security → Services**: enable Dashboard (29999), Primary (30001), RTDE (30004). Ping works without this but every TCP port times out.
2. **Top-right toggle: Remote Control mode.** `RTDEControlInterface` refuses to attach in Local mode.
3. Robot must be in Normal state — base ring **green**.

Quick diagnostic from your Mac:

```bash
for p in 29999 30001 30002 30004; do nc -zv -G 2 192.168.0.100 $p; done
```

All four should report "succeeded".

## Architecture of `teleop_ur15.py`

Threads:
- **`poll_loop`** — `rtde_r.getActualQ()` at 30 Hz into `current_q` under `state_lock`.
- **`viz_loop`** — when not playing, writes `current_q` into the `ViserUrdf`.
- **`_play`** — spawned per Play click; owns the URDF viz and (when Execute is on) the `servoJ` stream.

State machine:
1. **Idle:** browser mirrors real arm. User drags a 6-DoF gizmo.
2. **Add waypoint:** snapshots `(pos, wxyz)` into `waypoints[]` and adds a frame to the scene. Remove-last / Clear reverse it.
3. **Plan:** seeded IK at each waypoint in order (each segment's solution seeds the next). Stores `plan_segments: list[(q_start, q_goal)]`. If no waypoints, falls back to the gizmo's current pose.
4. **Play:** iterates segments. Per segment, advances `alpha ∈ [0,1]` at `dt * speed_slider / seg_duration` per tick, with `seg_duration = max(MIN_SEG_DURATION_S, max(|Δq|) / MAX_JOINT_SPEED)`. Maps alpha through `_alpha_to_s` (trapezoidal: 25% parabolic ramp / 50% linear cruise / 25% parabolic ramp) and writes `q = q_start + delta * eased` to the URDF and (if Execute on) `rtde_c.servoJ(q, 0, 0, dt, lookahead, gain)`.
5. **Auto-cleanup after a successful executed Play:** clears waypoints + their frames, resets the gizmo to the new EE pose, drops `plan_segments`, unchecks Execute, disables Play. Stopped or preview plays leave state untouched.

Safety:
- Execute auto-unchecks the instant a Play with Execute=on begins.
- Stop button → `stop_flag` → loop breaks → `finally` calls `rtde_c.servoStop(SERVO_STOP_DECEL=2.0)` (default 10 felt jerky).

## Tunables — `teleop_ur15.py`

| Constant | Default | Effect |
|---|---|---|
| `ROBOT_IP` | `192.168.0.100` | UR15 controller address |
| `MAX_JOINT_SPEED` | `1.0` rad/s | Peak per-joint speed at slider=1.0 |
| `MAX_JOINT_ACCEL` | `8.0` rad/s² | Accel limit for Live following (lower = smoother, laggier) |
| `MIN_SEG_DURATION_S` | `0.5` s | Floor on per-segment time |
| `RAMP_FRAC` | `0.25` | Trapezoidal ramp fraction. `0.5` = triangle, `0.1` = sharper |
| `DWELL_S` | `0.2` s | Pause at each intermediate waypoint |
| `SERVO_STOP_DECEL` | `2.0` rad/s² | Final settle deceleration |
| `STREAM_HZ` | `50` | servoJ + viz frame rate |
| `LIVE_HZ` | `125` | Live gizmo-follow IK + servoJ rate (fast steady cadence via initPeriod/waitPeriod keeps servoJ smooth) |
| `rest_weight` (in Plan handler) | `2.0` | IK pull toward current joints |

## Hand-E gripper

A Robotiq Hand-E parallel gripper is mounted on the UR15 wrist (RS-485 + 24 V through the tool flange connector).

**Rendering & TCP.** `hande.urdf` is rendered as a second `ViserUrdf` rooted at `/world/gripper`, whose frame is slaved to the live `tool0` pose every viz/play tick (`_update_gripper_viz`). The gripper is rigid, so the grasp point is a *fixed* `tool0`→`hande_end` offset (`TOOL0_T_GRASP`, ~0.1565 m along tool0 +z, read straight from the URDF). The gizmo/waypoints live at the grasp point; `_grasp_to_tool0()` maps them back to a `tool0` target so **the UR IK model and the seeded-IK call are unchanged**. Only the one actuated finger joint animates (the other `mimic`s it); 0 m = closed, 0.025 m = open.

**Control path — the URCap socket.** ur_rtde keeps its control script resident for the whole session, so a Robotiq gripper *program* can't run concurrently. But the **Grippers URCap also runs a background daemon** that owns the wrist RS-485 and serves a socket at **`<robot_ip>:63352`** — independent of whatever program is playing, so it coexists with ur_rtde, and the pendant buttons keep working (both route through the same daemon). No URCap swap. `hande_gripper.py` speaks the URCap's newline-terminated ASCII protocol over that socket (`SET POS 255 SPE 255 FOR 150 GTO 1` → `ack`; `GET STA` → `STA 3`); stdlib `socket` only, no deps. This is the same channel the standalone `robotiq_gripper.py` driver uses.

⚠️ **PolyScope X firewall.** Like RTDE/Dashboard, port 63352 is likely blocked by the Services firewall by default — allow it under **Settings → Security → Services**. **Run `verify_hande.py` first** (connect → activate → open/close standalone); only once it passes is gripper control in `teleop_ur15.py` trustworthy. If 63352 isn't reachable on X even after opening it, the fallbacks are a USB-RS485 adapter (talk Modbus straight to the gripper) or the RS485 URCap socket bridge — both absorbed by swapping `HandEGripper`'s transport. The gripper connect in `teleop_ur15.py` is best-effort: if the socket is absent it logs and runs viz-only (the slider still animates the meshes).

**URCap socket protocol** (port 63352): `SET <VAR> <val> …` → `ack`, `GET <VAR>` → `<VAR> <val>`. Vars: `ACT` activate, `GTO` go-to, `POS` request (0 open … 255 closed), `SPE` speed, `FOR` force (all 0–255), `STA` status (==3 ⇒ activated), `OBJ` object detection (1/2 ⇒ stopped on an object), `FLT` fault.

**Gripper position is one tracked value.** `gripper_frac` (a float, `0.0`=open … `1.0`=fully closed) is the single source of truth; the viz fingers always render it (`_frac_to_finger`). The **"Gripper close %" slider** drives it live — its `on_update` calls `_command_grip(frac)`, which commands the real gripper (`HandEGripper.move(frac)` → Robotiq `POS 0..255`) and snaps the viz. At startup the script sends an `open` command, so `gripper_frac` (0.0) is known without polling. **Capture/Add records the current `gripper_frac` into the waypoint automatically.** On replay the gripper actuates only when a waypoint's fraction differs from the running one by more than `GRIP_EPS` (2%), and the arm settles `GRIP_PREDELAY_S` (0.5 s) before each actuation. Saved trajectories persist the per-waypoint fraction (it's just a field in the waypoint JSON); legacy `"open"`/`"close"` strings are normalized on Plan via `_norm_grip`.

### Tunables — Hand-E (`teleop_ur15.py` / `hande_gripper.py`)

| Constant | Default | Effect |
|---|---|---|
| `GRIPPER_HOST` | `ROBOT_IP` | Grippers URCap socket host (the UR controller) |
| `GRIPPER_PORT` | `63352` | Robotiq URCap socket server port (open it in the X firewall) |
| `GRIPPER_FINGER_OPEN` | `0.025` m | Per-side finger travel at "open" (URDF upper limit) |
| `GRIPPER_TWEEN_S` | `0.8` s | Viz finger animation duration (match the real move) |
| `GRIPPER_MASS` / `GRIPPER_COG` | `1.0` kg / `(0,0,0.06)` m | Payload told to the UR via `setPayload` so it compensates gravity at the loaded wrist (bump if you add a workpiece) |
| `DEFAULT_SPEED` / `DEFAULT_FORCE` | `255` / `150` | Robotiq `SPE` / `FOR` (0–255); force kept collaborative |
| `GRIP_PREDELAY_S` | `0.5` s | Settle hold before the gripper actuates at a waypoint |

**End-of-play precision & the payload.** With the ~1 kg gripper on the wrist, an *undeclared* payload makes `servoJ` hold the loaded joints slightly below target (gravity droop) — which is why end-of-play undershoot felt worse after mounting it. `setPayload(GRIPPER_MASS, GRIPPER_COG)` at startup fixes that. Separately, the gizmo now targets the grasp point (~156 mm past `tool0`), so the *same* joint error shows up as a larger Cartesian shift — geometric, not a regression. The end-of-play settle (`SETTLE_*`) drives joint error to the `servoJ` floor regardless.

## Live gizmo-follow mode

Both scripts have a **"Live (drive robot)"** checkbox: a single toggle that makes the real arm chase the gizmo in real time (no Plan/Play). The `_live_loop` thread, at `LIVE_HZ` (30): reads the gizmo pose → seeded IK (seeded from the *last commanded* `q`, not measured, to avoid feedback jitter) → clamps the step → commands the arm. Mutually exclusive with Plan/Play; **Stop** or unticking ends it.

Safety, shared:
- **Snap-on-enable:** the gizmo jumps to the current EE first, so the arm never lurches toward a stale gizmo pose.
- **Acceleration-limited follower (UR):** the live loop tracks the IK target through a per-joint motion profile that bounds **both** speed (`MAX_JOINT_SPEED`) and the rate speed can change (`MAX_JOINT_ACCEL`), with the desired speed tapered to `√(2·a·|err|)` so it decelerates to rest at the target without overshoot and a one-tick `|err|/dt` cap that kills dither at rest. This bounds jerk (smooth stops / IK branch-flips) and low-passes IK jitter — the UR has no controller-side filter like the GoFa's EGM `LpFilter`. Runs at `LIVE_HZ` (125) on an exact `initPeriod`/`waitPeriod` cadence (`servoJ` is jitter-sensitive). The GoFa live loop still uses the simpler per-tick step clamp since EGM filters controller-side.

Per-arm:
- **UR15:** `servoJ` each tick; `servoStop` on exit.
- **GoFa:** streams the target over the *existing* EGM session (no supervisor change). Because `PyEgm.mod` uses `\CondTime := 1`, a >1 s pause lets the robot converge and RAPID drops the session; `_live_loop` detects the stale feed (`egm.is_fresh`) and **re-arms** (`_start_egm_session`) on the next motion — so expect a brief hitch after a long pause. It also applies the `MAX_TCP_SPEED` collaborative cap by scaling the per-tick step. On exit it holds the last pose for `HOLD_AFTER_PLAY_S` so `\CondTime` cleanly closes the session. (A smoother-through-pauses version would need a larger `\CondTime` in the supervisor + installer re-run.)

## Free-drive teach & saved trajectories

Teach-by-demonstration: hand-guide the arm, capture poses, save/replay. UR15 has the full version (software free-drive + gripper actions); the GoFa has capture + save/load with hand-guiding via its hardware button.

- **Free-drive** checkbox → `ur_rtde` `teachMode()` (zero-gravity hand-guiding); unticking / Stop calls `endTeachMode()`. Mutually exclusive with Plan/Play/Live. **`teachMode` must be ended before any `servoJ`** or the control mode conflicts — every exit path does this.
- **Capture waypoint** snapshots the live joints + FK grasp pose; **Add waypoint** still captures the gizmo pose. Both record the current tracked `gripper_state` (`open`/`close`) automatically — there is no dropdown (see the Hand-E "Gripper state is one tracked value" note).
- **Waypoint model:** `{"q": [6]|None, "pos", "wxyz", "grip"}` where `grip` is the absolute gripper state at that waypoint. Capture fills `q` (taught joints); gizmo-add leaves `q=None` and Plan backfills it from IK. At Plan, a waypoint **with `q` replays those joints exactly** (no IK); without `q` it IKs from the Cartesian pose (sequential seed, as before). `plan_segments` is `(q_start, q_goal, grip)`; `_play` actuates the gripper at a waypoint **only when its state differs** from the running state — settling `GRIP_PREDELAY_S` (0.5 s) first, holding the arm with `servoJ` (real gripper only on an executed play, viz tweens either way).
- **Save/Load:** `trajectories/<name>.json` = `{robot, created, waypoints}`. Load clears + repopulates the waypoint list and frames; then Plan to replay. (Saved trajectories are tracked, not gitignored — they sync across machines via the repo.)
- **GoFa:** same Capture + Save/Load (waypoints store joints+Cartesian; taught joints replay exactly). **Software free-drive toggle:** the **"Free-drive (lead-through)"** checkbox flips a `lead_go` flag over RWS; `PyEgm.mod` then calls `SetLeadThrough \On` (RAPID hand-guiding) and holds the arm compliant until you untick it (`SetLeadThrough \Off`). Mutually exclusive with Plan/Play/Live; Stop and untick both release it; the controller also auto-clears lead-through on motors-off. The GoFa's **physical lead-through button** still works too. Either way, Capture reads the hand-moved joints via RWS polling. No gripper actions. **Requires re-running `install_gofa_egm.py`** to push the updated supervisor (see the PyEgm.mod section + its lead-through caveats).

## Headless replay — `play_trajectory.py`

Replay a saved trajectory on either arm without viser:

```bash
./robot_control/bin/python play_trajectory.py <name> [--speed S] [--dry-run] [--no-confirm]
```

Reads `trajectories/<name>.json`, auto-detects the robot from its `"robot"` field, and executes on the real arm after a `[y/N]` confirm (`--no-confirm` to skip). `--dry-run` prints the plan (segments + estimated duration) and exits without touching hardware. It is **IK-solver-free**: every waypoint must already carry `"q"` (from Capture, or Plan-and-save in viser) — a `q`-less waypoint aborts with a "Plan + re-save" message. The first segment moves the arm from its current pose to waypoint 1 (same as viser). The UR15 path mirrors `teleop_ur15.py` (servoJ + settle + gripper-on-change with the 0.5 s pre-delay, gripper opened at start). The GoFa path imports pyroki for forward kinematics **only** to enforce the `MAX_TCP_SPEED` collaborative cap, then streams over the existing EGM supervisor (PyEgm must be parked at `WaitUntil egm_go`). The profile constants are mirrored from the teleop scripts (the `UR_`/`GOFA_` prefixed block at the top of `play_trajectory.py`) — keep them in sync if you retune a teleop script.

---

# GoFa CRB 15000

## Controller setup (one-time, ~20 minutes)

This is more involved than the UR because ABB controllers always run a RAPID program for motion — Python doesn't talk to the motion executor directly; it pokes RAPID variables and a RAPID `WHILE TRUE` loop does the work.

### Hardware

**1. Safety jumpers.** The OmniCore C30 ships with the safeguard chain expecting an external safety device. For a standalone benchtop GoFa, you need physical jumpers on the **X14** terminal block:

| Pair | Function | Status (lab install) |
|---|---|---|
| ES1 (pins 1–2) | Emergency stop ch 1 | jumpered |
| ES2 (pins 3–4) | Emergency stop ch 2 | jumpered |
| AS1 (pins 5–6) | Auto stop / safeguard ch 1 | **must be jumpered** |
| AS2 (pins 7–8) | Auto stop / safeguard ch 2 | **must be jumpered** |

Symptom of missing AS jumpers: Manual mode works (the pendant enabling grip switch bypasses AS), Auto mode immediately fires a "guard stop / protective stop circuit open". Look at the **specific** event log entry — if it says AS1/AS2 or "protective stop circuit", jumper them.

Safety note: jumpering AS bypasses external safeguarding inputs. The GoFa's built-in cobot collision detection still protects against collisions, but **don't** run high-speed Auto motion with people in the workspace.

### Network

OmniCore C30 has three logical networks; pick one for RWS access:

| Logical network | Physical port | What lives there |
|---|---|---|
| Private / MGMT | **MGMT** (the rightmost of the three at bottom-right) | RWS at `192.168.125.1`, FlexPendant, RobotStudio direct |
| Public / WAN | **WAN** (next to MGMT) | RWS on plant subnet, IP configurable from pendant |
| I/O Network | **LAN** + X1–X5 ETHERNET SWITCH | EtherNet/IP, Profinet, fieldbus only — **not RWS** |

For lab use, the simplest path is Mac → MGMT direct via Ethernet cable, with the Mac's Ethernet interface set to a static `192.168.125.x` (anything except `.1`).

⚠️ **VPN gotcha.** If your Mac is on a VPN that claims the `192.168.125.0/24` subnet (e.g., a corporate VPN), packets to the GoFa will be routed into the VPN tunnel instead of out the Ethernet cable. Symptom: connect-refused or timeouts on every port. Either disconnect the VPN while working with the GoFa, or use the WAN port on a non-conflicting subnet (e.g., `192.168.0.102` if your lab network is `192.168.0.0/24` — same subnet as the UR15).

Diagnostic from Mac:

```bash
ping -c 2 192.168.125.1               # should reply, ~0.5 ms
nc -zv -G 2 192.168.125.1 443         # should be "succeeded" (OmniCore is HTTPS)
route get 192.168.125.1 | grep interface   # should be en* (Ethernet), NOT utun* (VPN)
```

### Pendant: enter Auto mode

OmniCore C30 has **no physical Auto/Manual key switch** (unlike larger ABB controllers). Mode selection is on the FlexPendant touchscreen — usually a small mode-icon in the top status bar. Tap it → Automatic → confirm. There may be a separate white motors-on button on the front of the controller cabinet that needs to be pressed once.

### Push the EGM supervisor

EGM is a **licensed** controller option — confirm it's enabled (pendant: Settings → System → installed options) before this will work.

Run the installer:

```bash
./robot_control/bin/python install_gofa_egm.py
```

What it does:
1. Connects to RWS over HTTPS at `ROBOT_IP` (default `192.168.125.1`).
2. Probes controller state, opmode, exec state.
3. Grabs RAPID mastership and stops any running program.
4. **Unloads `MainModule`** — it ships with a `PROC main()` that collides with ours.
5. Uploads `EGM_COMM.cfg` + `EGM_MOC.cfg` and `PyEgm.mod` to `$HOME/` on the controller.
6. Loads `PyEgm.mod` into task `T_ROB1` and turns motors on (if off).
7. Tries `resetpp` + `start_program` over RWS; when that fails (see [OmniCore RWS gotchas](#omnicore-rws-gotchas)) it prompts you to tap **PP to Main** + green **Play** on the pendant.

⚠️ The `.cfg` files define the EGM UDP peer (`UCdevice` → your PC). `EGM_COMM.cfg` `RemoteAddress` must equal your PC's IP (default assumes `192.168.125.50`; edit it and `PC_IP` in the installer to match) and `RemotePortNumber` must equal `EGM_LOCAL_PORT` (6510) in `teleop_gofa_egm.py`. Applying the `.cfg` may need a controller restart from the pendant.

After this, `PyEgm` is parked at `WaitUntil egm_go = TRUE OR lead_go = TRUE`. Setting `egm_go = TRUE` (which `teleop_gofa_egm.py` does via RWS) makes it enter `EGMRunJoint` and follow the UDP joint stream until convergence, then clear `egm_go` and re-park. Setting `lead_go = TRUE` instead makes it call `SetLeadThrough \On` (hand-guiding) and hold until `lead_go` clears, then `SetLeadThrough \Off`.

### PyEgm.mod — the supervisor

```RAPID
MODULE PyEgm
  PERS bool egm_go := FALSE;
  PERS bool lead_go := FALSE;          ! TRUE = software lead-through (hand-guide)
  CONST string EGM_EXT_NAME := "default";
  CONST string EGM_UC_NAME  := "UCdevice";
  VAR egmident egm_id;

  PROC main()
    AccSet 50, 50;
    WHILE TRUE DO
      WaitUntil egm_go = TRUE OR lead_go = TRUE;
      IF lead_go = TRUE THEN
        SetLeadThrough \On;            ! default StopMove -> arm compliant
        WaitUntil lead_go = FALSE;
        SetLeadThrough \Off;           ! default ClearPath + StartMove -> resume
      ELSE
        EGMReset egm_id;
        EGMGetId egm_id;
        EGMSetupUC ROB_1, egm_id, EGM_EXT_NAME, EGM_UC_NAME \Joint;
        EGMActJoint egm_id \LpFilter := 20 \MaxSpeedDeviation := 20;
        EGMRunJoint egm_id, EGM_STOP_HOLD \J1 \J2 \J3 \J4 \J5 \J6
          \CondTime := 1 \RampInTime := 0.1 \RampOutTime := 0.2;
        EGMReset egm_id;
        egm_go := FALSE;
      ENDIF
    ENDWHILE
  ENDPROC
ENDMODULE
```

Knobs (edit in `install_gofa_egm.py`, then rerun the installer — Ctrl+C any running teleop first so mastership is free):

- `\MaxSpeedDeviation := 20` — controller-side per-joint speed cap (deg/s). Backstop to the Python `MAX_TCP_SPEED` cap; raise both together if you raise speed.
- `\LpFilter := 20` — low-pass cutoff (Hz); lower = smoother but laggier following.
- `\CondTime := 1` — seconds of convergence before `EGMRunJoint` returns (how the session ends after the final target is held).

**Lead-through (`SetLeadThrough`) caveats — verify on hardware.** `SetLeadThrough \On`/`\Off` is the RAPID hand-guiding instruction (3HAC050917-001 / RW7 3HAC065038). Two unknowns to confirm on the actual GoFa: (1) the **RW6** manual documents it as YuMi-only — RW7/OmniCore reportedly extends it to GoFa, but if the controller rejects it, the build error shows at `/rw/rapid/tasks/T_ROB1/program/builderror` (check it after the installer loads `PyEgm`); (2) whether it engages in **Auto** (EGM needs Auto) or requires Manual + the enabling device — the physical lead-through button working in your current Auto setup is a good sign. If lead-through can't run in Auto, teach in Manual and switch to Auto to replay. On failure, the physical button path is unaffected (no regression).

## Architecture of `teleop_gofa_egm.py`

Same shape as `teleop_ur15.py`, with EGM in place of servoJ:

- **State polling** uses `rws.get_joints()` at ~10 Hz (idle viz only). During an Execute play, the play loop drives the URDF directly from the streamed target.
- **Execute mode** streams joint targets over EGM (UDP) at `STREAM_HZ`. The same `q` from the trapezoidal alpha profile goes to BOTH viser and the EGM stream every tick, so viz and robot move in lockstep — like UR15 servoJ. A play:
  1. Sets `egm_go = TRUE` (via RWS) and waits for the first EGM feedback packet (controller has entered `EGMRunJoint`).
  2. Streams targets segment by segment, holding each waypoint for a short dwell between segments.
  3. Holds the final target for `HOLD_AFTER_PLAY_S` so RAPID's `\CondTime` convergence fires and `EGMRunJoint` exits, clearing `egm_go`.

**Speed is unified and TCP-capped.** The slider scales playback, but `_cap_seg_duration()` stretches each segment so the real TCP speed never exceeds `MAX_TCP_SPEED` (measured against the URDF kinematics per segment). The slider can only scale *below* that cap.

Startup safety: the script sets `egm_go = FALSE` on connect so a stray TRUE doesn't fire EGM.

## Tunables — `teleop_gofa_egm.py`

| Constant | Default | Effect |
|---|---|---|
| `ROBOT_IP` | `192.168.125.1` | GoFa MGMT IP (direct cable). For WAN/lab use, set to the Public IP you assigned. |
| `RWS_USER` / `RWS_PASSWORD` | `Default User` / `robotics` | OmniCore defaults |
| `EGM_LOCAL_PORT` | `6510` | UDP port; must match `RemotePortNumber` in `EGM_COMM.cfg` |
| `MAX_TCP_SPEED` | `0.25` m/s | Hard cap on real tool speed (collaborative limit) |
| `MAX_JOINT_SPEED` | `1.0` rad/s | Per-joint pacing before the TCP cap is applied |
| `MIN_SEG_DURATION_S` | `0.5` s | Floor on per-segment time |
| `RAMP_FRAC` | `0.25` | Trapezoidal profile ramp fraction |
| `DWELL_S` | `0.2` s | Pause between segments |
| `HOLD_AFTER_PLAY_S` | `1.5` s | Hold final target so RAPID's `\CondTime` fires |
| `POLL_HZ` | `10` | RWS state polling rate |
| `STREAM_HZ` | `100` | EGM target stream + viz frame rate |
| `LIVE_HZ` | `30` | Live gizmo-follow IK + EGM target update rate |
| `rest_weight` (Plan handler) | `2.0` | IK pull toward current joints |

## OmniCore RWS gotchas

These bit us during the GoFa bring-up. Captured here so future-us doesn't relearn them:

- **HTTPS only, port 443.** Port 80 is disabled by default. Use `https=True` (the default in `abb_rws.RWSClient`). The cert is self-signed; we set `verify=False` and suppress `InsecureRequestWarning`.

- **HTTP Basic auth, NOT Digest.** OmniCore RWS 2.0 advertises `WWW-Authenticate: Basic`. `abb_robot_client` (the popular Python lib) uses Digest, which is for IRC5/RW6 — it'll silently 401 against OmniCore. `abb_rws.RWSClient` uses Basic by default; pass `auth_scheme="digest"` for legacy IRC5.

- **Every POST/PUT must have `Content-Type: application/x-www-form-urlencoded;v=2.0`** — note the `;v=2.0` suffix. Without it you get `406 Not Acceptable`. `_post()` sets this automatically.

- **Mastership URL has action in the path, not the body.** OmniCore: `POST /rw/mastership/edit/request`. IRC5 was: `POST /rw/mastership/edit` body `action=request`. Many other endpoints still use action-in-body — there's no consistent convention.

- **RAPID symbol data URL needs the module name in the path.** OmniCore: `POST /rw/rapid/symbol/RAPID/{task}/{module}/{var}/data` body `value=...`. IRC5 was: `POST /rw/rapid/symbol/data/RAPID/{task}/{var}?action=set`.

- **Mastership orphans easily.** If a process holding mastership crashes, the controller still thinks it's held until the session times out. Symptom: subsequent `request` calls 403. Fix: reboot the controller from the pendant, or wait ~30s for session timeout. `abb_rws.RWSClient.__post_init__` registers an `atexit` hook to release on clean exit.

- **`MainModule` collision.** The GoFa ships with a `MainModule.mod` that has its own `PROC main()`. RAPID won't compile two `main`s — when we load `PyEgm` the controller logs "Errors in RAPID program" event 40160. The installer unloads `MainModule` before loading `PyEgm` via `POST /rw/rapid/tasks/T_ROB1/unloadmod` body `module=MainModule`.

- **Build errors are visible at `/rw/rapid/tasks/T_ROB1/program/builderror`.** Useful for diagnosing semantic errors after a module load — it returns module name, row, column, error type. Way more useful than the generic "errors in RAPID program" event log entry.

- **`resetpp` (PP-to-Main) and `start_program` via RWS** — we couldn't find the right shape on OmniCore. Every variation returned 400 "semantic error" or 403. The installer falls back to telling the user to press PP-to-Main + Play on the pendant. If you crack the right URL, it would tighten the install flow.

- **OmniCore C30 has no physical Auto/Manual key switch.** Mode is selected on the FlexPendant touchscreen (top status bar icon). Larger ABB controllers do have a key switch — don't assume.

---

# Custom IK: `pyroki_snippets/_solve_ik_seeded.py`

PyRoki's stock `solve_ik` snippet has no seed and no posture cost, so it returns *a* valid IK solution that often lives in a distant null-space branch (wrist flipped 180°, elbow on the wrong side, etc.). `solve_ik_seeded` adds:
- `pk.costs.rest_cost(joint_var, q_seed, weight=rest_weight)` — pulls the solution toward `q_seed` (defaults to `rest_weight=2.0`).
- `initial_vals=VarValues.make([joint_var.with_value(q_seed)])` — starts the optimizer at the seed instead of at zeros.

Trade-off: at `rest_weight=2.0` the pose error is ~0.5 mm. Raise to 5–10 if the IK still picks wrong branches; the gizmo target then drifts more but stays in the same kinematic family.

Used by both `teleop_ur15.py` and `teleop_gofa_egm.py`.

---

# Other gotchas

- **Boost 1.90 breaks ur_rtde on macOS.** Boost made `Boost.System` header-only in 1.87+, so homebrew Boost 1.90 ships no `boost_system-*-Config.cmake` and ur_rtde's `find_package(boost_system CONFIG)` dies. Fix: `brew install boost@1.85` (keg-only, doesn't shadow 1.90), then build ur_rtde with `BOOST_ROOT=/opt/homebrew/opt/boost@1.85 CMAKE_PREFIX_PATH=/opt/homebrew/opt/boost@1.85 pip install ur_rtde`. Already done in `robot_control/`.

- **`pip install pyroki` doesn't exist.** Install from source: `git clone https://github.com/chungmin99/pyroki.git pyroki_src && ./robot_control/bin/pip install -e ./pyroki_src`. Already done.

- **PyRoki's `solve_ik` lives in `examples/pyroki_snippets/`, not the package itself.** It's NOT installed by `pip install -e .`. Workaround: the `pyroki_snippets/` directory in the project root is a copy of `pyroki_src/examples/pyroki_snippets/` plus our `_solve_ik_seeded.py`, and both teleop scripts add the project root to `sys.path` so `import pyroki_snippets` works.

- **First IK call takes ~800 ms (JAX JIT compile); subsequent calls are milliseconds.** Both teleop scripts call `_warmup_ik()` at launch (a no-op IK at the current pose) to pay this cost during startup instead of on the first Plan click — so launch prints "Warming up IK solver…" and takes ~800 ms longer, but the first real Plan is fast.

- **UR15 model is brand new.** It postdates a lot of training data — if Claude says "there's no UR15", point at https://www.universal-robots.com/products/ur15/. The official ROS2 description repo supports `ur_type:=ur15`.

- **yourdfpy + `file://` URIs.** When `xacrodoc` processes a xacro to URDF, it resolves `package://` URIs into absolute `file://...` paths. yourdfpy's `filename_handler` callback only fires for unresolved URIs (`package://`, plain paths) — `file://` is passed straight to trimesh which can't open them. Fix: strip `file://` prefix from the URDF after xacro processing. The GoFa URDF was generated this way.

- **ABB `abb-robot-client` (Python) only supports IRC5.** Despite being the most-starred ABB python lib, it doesn't work on OmniCore. We wrote `abb_rws.py` ourselves; see [OmniCore RWS gotchas](#omnicore-rws-gotchas).
