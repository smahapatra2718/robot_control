# abb_foga — robot teleop scaffold

Browser-based teleop for two arms on a shared viser + pyroki stack:

- **`teleop_ur15.py`** — Universal Robots UR15 over RTDE (`ur_rtde`), `servoJ` per tick.
- **`teleop_gofa_egm.py`** — ABB GoFa CRB 15000 (variant `crb15000_5_95`) over Externally Guided Motion (EGM, a licensed option), with RWS (`abb_rws.py`) for mastership + the start/stop flag. Joint targets stream over UDP at `STREAM_HZ` to a RAPID supervisor (`PyEgm.mod`) running `EGMRunJoint`. Real tool speed is held under `MAX_TCP_SPEED` regardless of slider/pose.

Both unify the speed slider — the same `q` drives the viser preview and the robot each tick, so viz and arm move in lockstep. Both share the UI (viser scene + gizmo + waypoints), the IK (`pyroki_snippets._solve_ik_seeded`), the trapezoidal alpha play profile, and auto-cleanup after a successful executed Play. All four entry points (`teleop_ur15.py`, `teleop_gofa_egm.py`, `play_trajectory.py`, `teleop.py`) pull shared config + pure helpers from **`robot_common.py`** (single source of truth).

```bash
./robot_control/bin/python scripts/real.py ur15   # real hardware  (targets: ur15 | gofa | play | teleop)
./robot_control/bin/python scripts/sim.py  ur15   # offline sim    (same targets, no robot/network)
```

Open the printed `http://localhost:8080`. Each script connects to its controller at startup and aborts if it can't reach.

## Project layout

Runnable scripts in `scripts/`, importable modules in `lib/`, assets/vendored trees in their own folders. Each `scripts/` script bootstraps the repo root (for `pyroki_snippets`) and `lib/` onto `sys.path`, so it imports `robot_common` / `abb_rws` / … by bare name from anywhere. Asset paths resolve from `robot_common._ROOT` (= parent of `lib/`), so scripts work regardless of cwd.

```
abb_foga/
├── scripts/                    # entry points — run with ./robot_control/bin/python scripts/<script>
│   ├── teleop_ur15.py          #   UR15 teleop: viser + RTDE/servoJ + Hand-E gripper
│   ├── teleop_gofa_egm.py      #   GoFa teleop: viser + EGM joint streaming
│   ├── teleop.py               #   headless CLI trajectory recorder (free-drive + keypress capture)
│   ├── play_trajectory.py      #   headless replay of a saved trajectory (UR15 / GoFa)
│   ├── install_gofa_egm.py     #   one-shot GoFa bring-up: generates+loads PyEgm.mod, sets the UDP peer
│   ├── verify_hande.py         #   one-shot Hand-E gripper comms probe (run before trusting control)
│   ├── real.py                 #   dispatcher: real.py <ur15|gofa|play|teleop> [args] on real hardware
│   ├── sim.py                  #   dispatcher: sim.py  <ur15|gofa|play|teleop> [args] offline (fake arm)
│   └── sim_smoketest.py        #   stdlib-assert smoke test for the sim fakes + EGM handshake
│
├── lib/                        # importable modules (on sys.path via the scripts/ bootstrap)
│   ├── robot_common.py         #   shared config (UR_*/GOFA_* constants) + pure helpers
│   ├── abb_rws.py              #   minimal RWS client for OmniCore (RobotWare 7+)
│   ├── abb_egm.py              #   minimal EGM UDP client (protobuf wire format)
│   ├── egm_pb2.py / egm.proto  #   EGM protobuf bindings + the schema they're generated from
│   ├── hande_gripper.py        #   minimal Hand-E URCap-socket client
│   ├── robot_sim.py            #   offline sim: SimWorld + fake transports + install() (sys.modules shim)
│   ├── dispatch.py             #   shared target->script map + dispatch() for real.py / sim.py
│   └── control/                #   RobotController core (state.py, base.py, ur.py, gofa.py) — one motion impl
│
│   # ── assets ──
├── urdf/                       # generated robot models (crb15000_5_95.urdf, hande.urdf)
├── egm/                        # GoFa controller EGM config (EGM_COMM.cfg, EGM_MOC.cfg) — reference
├── trajectories/               # saved teach trajectories (<name>.json)
│
│   # ── vendored third-party (see README "Vendored third-party sources") ──
├── pyroki_src/                 # git clone of chungmin99/pyroki, installed -e (don't move: editable install)
├── pyroki_snippets/            # copy of pyroki_src/examples/ + our _solve_ik_seeded.py (imported by bare name)
├── abb_desc/                   # clone of ros-industrial/abb (GoFa meshes; package://abb_crb15000_support)
├── robotiq_hande_description/  # vendored macmacal/robotiq_hande_description (Hand-E meshes)
│
│   # ── runtime / docs ──
├── robot_control/              # Python venv (3.13), gitignored
├── docs/                       # design specs + plans
├── README.md
└── CLAUDE.md                   # this file
```

## Dependencies (already installed in `robot_control/`)

Python: numpy, viser, yourdfpy, jaxlie, jax, jaxlib, robot_descriptions, xacrodoc, pyroki (editable), ur_rtde 1.6.3, requests, urllib3. System (brew): cmake, **boost@1.85** (keg-only, for the ur_rtde build only).

URDFs — all generated via `xacrodoc`, then mesh URIs rewritten to `package://` and resolved through `robot_common.make_mesh_resolver(<prefix>)`. xacrodoc emits absolute `file://` URIs that yourdfpy can't resolve, so the `file://…` prefix is stripped back to `package://` after generation (see the yourdfpy gotcha below).
- **UR15**: no local file — loaded at runtime via `robot_descriptions.loaders.yourdfpy.load_robot_description("ur15_description")`.
- **Hand-E**: local `urdf/hande.urdf` from the vendored `robotiq_hande_description` (prefix = project root). `<ros2_control>` block stripped post-gen. Rendered as a *second* `ViserUrdf` slaved to the live `tool0` pose — the UR IK model is untouched.
- **GoFa**: local `urdf/crb15000_5_95.urdf` (5 kg / 0.95 m reach variant — match to your hardware nameplate) from the ros-industrial repo (prefix = `abb_desc/`, meshes under `package://abb_crb15000_support`).

## Shared config — `robot_common.py`

One stdlib-only module holds what the four entry points used to duplicate. Kept stdlib-only on purpose so importing it never drags in the heavy jax stack.

- **Config constants:** `UR_*` (IP, servoJ/settle params, Hand-E geometry + payload) and `GOFA_*` (IP, RWS creds, RAPID module/flags, EGM port, TCP-speed cap, hold time), plus shared trapezoidal-profile knobs (`RAMP_FRAC`, `MIN_SEG_DURATION_S`, `DWELL_S`, `GRIP_PREDELAY_S`, `GRIP_EPS`) and `TRAJ_DIR` / `TARGET_LINK`.
- **Pure helpers:** `alpha_to_s` (trapezoidal velocity profile), `norm_grip` (legacy `"open"`/`"close"` → fraction), `make_mesh_resolver` (`package://` → local-path `filename_handler` factory), `load_trajectory` / `save_trajectory` (the `trajectories/<name>.json` read/write).

Teleop scripts bind these to short local names (`ROBOT_IP = rc.UR_ROBOT_IP`, …) so the large hardware-loop bodies are unchanged; `play_trajectory.py` / `teleop.py` `from robot_common import` the `UR_*`/`GOFA_*` names directly. Script-specific tunables stay local (e.g. UR-only `MAX_JOINT_ACCEL`, `LIVE_HZ`, `POLL_HZ`).

**Forward kinematics is deliberately NOT shared** — it needs jax/jaxlie/pyroki, which `teleop.py` / `play_trajectory.py` import *lazily* to defer the ~800 ms JIT cost. The few `ee_pose` / `grasp_pose` / `_grasp_to_tool0` blocks (~3 lines each) stay per-script.

---

## Simulation — `scripts/sim.py` (offline, no robot)

`sim.py <target>` runs the **real, unmodified** teleop scripts against a
perfect-tracking kinematic sim, so trajectories / IK / the viser UI / play
profiles can be exercised on the dev machine with no UR15, GoFa, or network. It
works by injecting fake transport modules (`rtde_control`, `rtde_receive`,
`hande_gripper`, `abb_rws`, `abb_egm`) into `sys.modules`, then `runpy`-ing the
target script — see `lib/robot_sim.py`. Because it literally runs
`teleop_ur15.py` (etc.), every feature works in sim automatically and never
drifts. `real.py` and `sim.py` are the same dispatcher (`lib/dispatch.py`) and
differ only in that `sim.py` installs the shim first.

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

## RobotController core — `lib/control/`

One thread-safe motion implementation behind every surface. `make_controller("ur15"|"gofa")`
returns a controller that owns the hardware client and exposes async high-level commands —
`move_to_joints` / `move_to_pose` / `play` / `set_gripper` / `stop` / `estop` (each returns a
command id; `wait(id)` blocks for the result) — plus `get_state() -> RobotState` (joints, FK
pose, gripper, safety, activity) from a background state-poll thread, and `grasp_pose` /
`start_freedrive` / `stop_freedrive` / `adjust_grip` for the recorder. The headless players
(`play_trajectory.py`, `teleop.py`) run on it; the viser teleops and the remote API are next.
Because it uses the same hardware clients the sim fakes shadow, the whole core runs offline —
`./robot_control/bin/python scripts/control_smoketest.py` exercises it (move/play/stop/gripper/
state/free-drive) against `lib/robot_sim.py` with no robot.

Subclasses (`URController`, `GoFaController`) implement the hardware primitives (`_read_q`,
`_servo`-style `_run_play`, `_ik`, `_graceful_stop`/`_hard_stop`, …); the base owns the command
executor (one motion at a time; a submit while busy raises `Busy`; `stop`/`estop` preempt), the
state loop, and segment building. The motion loops are lifted verbatim from the tuned teleop
scripts, so behavior is identical.

# UR15

## Controller setup (one-time)

UR15 ships with Polyscope X, which firewalls external services by default. On the pendant:

1. **Settings → Security → Services**: enable Dashboard (29999), Primary (30001), RTDE (30004). Ping works without this but every TCP port times out.
2. **Top-right toggle: Remote Control mode.** `RTDEControlInterface` refuses to attach in Local mode.
3. Robot must be in Normal state — base ring **green**.

Quick diagnostic from your Mac:

```bash
for p in 29999 30001 30002 30004; do nc -zv -G 2 192.168.125.2 $p; done   # all four should "succeed"
```

## Architecture of `teleop_ur15.py`

Threads: **`poll_loop`** reads `rtde_r.getActualQ()` at 30 Hz into `current_q` (under `state_lock`); **`viz_loop`** writes `current_q` into the `ViserUrdf` when not playing; **`_play`** is spawned per Play click and owns the URDF viz + (when Execute is on) the `servoJ` stream.

State machine: Idle (browser mirrors arm, user drags gizmo) → Add waypoint (snapshot `(pos, wxyz)` + scene frame) → Plan (seeded IK at each waypoint, each solution seeds the next, stored as `plan_segments: list[(q_start, q_goal)]`; no waypoints → falls back to gizmo pose) → Play (per segment advance `alpha ∈ [0,1]` at `dt·speed/seg_duration`, `seg_duration = max(MIN_SEG_DURATION_S, max(|Δq|)/MAX_JOINT_SPEED)`, map through `_alpha_to_s`, write `q` to URDF and `servoJ` if Execute on). After a successful executed Play, auto-cleanup clears waypoints/frames, resets the gizmo to the new EE pose, drops `plan_segments`, unchecks Execute. Stopped/preview plays leave state untouched.

Safety: Execute auto-unchecks the instant a Play with Execute=on begins; Stop → `stop_flag` → loop breaks → `finally` calls `rtde_c.servoStop(SERVO_STOP_DECEL=2.0)` (default 10 felt jerky).

## Tunables — `teleop_ur15.py`

Shared ones live in `robot_common.py` (`UR_*` / unprefixed); the script binds them to these local names. UR-only knobs are defined locally.

| Constant | Default | Effect |
|---|---|---|
| `ROBOT_IP` | `192.168.125.2` | UR15 controller address |
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

A Robotiq Hand-E parallel gripper on the UR15 wrist (RS-485 + 24 V through the tool flange).

**Rendering & TCP.** `hande.urdf` is a second `ViserUrdf` rooted at `/world/gripper`, slaved to the live `tool0` pose each tick (`_update_gripper_viz`). The gripper is rigid, so the grasp point is a fixed `tool0`→`hande_end` offset (`TOOL0_T_GRASP`, ~0.1565 m along tool0 +z, from the URDF). The gizmo/waypoints live at the grasp point; `_grasp_to_tool0()` maps them back to a `tool0` target so **the UR IK model and the seeded-IK call are unchanged**. Only the one actuated finger joint animates (the other `mimic`s it); 0 m = closed, 0.025 m = open.

**Control path — the URCap socket.** ur_rtde keeps its control script resident all session, so a Robotiq gripper *program* can't run concurrently. But the **Grippers URCap also runs a background daemon** that owns the wrist RS-485 and serves a socket at **`<robot_ip>:63352`** — independent of the running program, so it coexists with ur_rtde and the pendant buttons keep working (same daemon). `hande_gripper.py` speaks the URCap's newline-terminated ASCII protocol over that socket, stdlib only.

⚠️ **PolyScope X firewall.** Like RTDE/Dashboard, port 63352 is likely firewalled — allow it under **Settings → Security → Services**. **Run `scripts/verify_hande.py` first**; only once it passes is gripper control trustworthy. If 63352 is unreachable even after opening it, fallbacks are a USB-RS485 adapter (Modbus straight to the gripper) or the RS485 URCap socket bridge — both absorbed by swapping `HandEGripper`'s transport. The gripper connect is best-effort: if the socket is absent it logs and runs viz-only.

**URCap socket protocol** (port 63352): `SET <VAR> <val> …` → `ack`, `GET <VAR>` → `<VAR> <val>`. Vars: `ACT` activate, `GTO` go-to, `POS` request (0 open … 255 closed), `SPE` speed, `FOR` force (all 0–255), `STA` status (==3 ⇒ activated), `OBJ` object detection (1/2 ⇒ stopped on an object), `FLT` fault. (`SET POS 255 SPE 255 FOR 150 GTO 1` → `ack`; `GET STA` → `STA 3`.)

**Gripper position is one tracked value.** `gripper_frac` (float, `0.0`=open … `1.0`=closed) is the single source of truth; viz fingers always render it (`_frac_to_finger`). The **"Gripper close %" slider** drives it live — its `on_update` calls `_command_grip(frac)` → real gripper (`HandEGripper.move(frac)` → `POS 0..255`) + snaps viz. Startup sends `open`, so `gripper_frac` (0.0) is known without polling. **Capture/Add records `gripper_frac` into the waypoint automatically.** On replay the gripper actuates only when a waypoint's fraction differs by more than `GRIP_EPS` (2%), after the arm settles `GRIP_PREDELAY_S` (0.5 s). Saved trajectories persist the per-waypoint fraction; legacy `"open"`/`"close"` strings normalize on Plan via `_norm_grip`.

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

**Payload & end-of-play precision.** With the ~1 kg gripper, an *undeclared* payload makes `servoJ` hold the loaded joints slightly below target (gravity droop) — `setPayload(GRIPPER_MASS, GRIPPER_COG)` at startup fixes that. The gizmo targets the grasp point (~156 mm past `tool0`), so the *same* joint error shows up as a larger Cartesian shift — geometric, not a regression. The end-of-play settle drives joint error to the `servoJ` floor regardless.

## Live gizmo-follow mode

Both scripts have a **"Live (drive robot)"** checkbox: the real arm chases the gizmo in real time (no Plan/Play). The `_live_loop` thread reads the gizmo pose → seeded IK (seeded from the *last commanded* `q`, not measured, to avoid feedback jitter) → clamps the step → commands the arm. Mutually exclusive with Plan/Play; Stop or unticking ends it.

- **Snap-on-enable:** the gizmo jumps to the current EE first, so the arm never lurches toward a stale pose.
- **UR acceleration-limited follower:** tracks the IK target through a per-joint profile bounding both speed (`MAX_JOINT_SPEED`) and accel (`MAX_JOINT_ACCEL`), with desired speed tapered to `√(2·a·|err|)` (decelerates to rest without overshoot) and a one-tick `|err|/dt` cap killing rest dither. This bounds jerk and low-passes IK jitter — the UR has no controller-side filter like the GoFa's EGM `LpFilter`. Runs at `LIVE_HZ` (125) on an exact `initPeriod`/`waitPeriod` cadence (`servoJ` is jitter-sensitive). `servoStop` on exit.
- **GoFa:** simpler per-tick step clamp (EGM filters controller-side). Streams the target over the *existing* EGM session. Because `PyEgm.mod` uses `\CondTime := 1`, a >1 s pause lets the robot converge and RAPID drops the session; `_live_loop` detects the stale feed (`egm.is_fresh`) and **re-arms** (`_start_egm_session`) on the next motion — expect a brief hitch after a long pause. Applies the `MAX_TCP_SPEED` cap by scaling the per-tick step; on exit holds the last pose for `HOLD_AFTER_PLAY_S` so `\CondTime` cleanly closes the session.

## Free-drive teach & saved trajectories

Hand-guide the arm, capture poses, save/replay. UR15 has the full version (software free-drive + gripper actions); GoFa has capture + save/load.

- **Free-drive** checkbox → UR `teachMode()` (zero-g hand-guiding); untick/Stop → `endTeachMode()`. Mutually exclusive with Plan/Play/Live. **`teachMode` must be ended before any `servoJ`** or the control mode conflicts — every exit path does this.
- **Capture waypoint** snapshots live joints + FK grasp pose; **Add waypoint** captures the gizmo pose. Both record the current `gripper_frac` automatically.
- **Waypoint model:** `{"q": [6]|None, "pos", "wxyz", "grip"}`. Capture fills `q` (taught joints); gizmo-add leaves `q=None` and Plan backfills via IK. At Plan a waypoint **with `q` replays those joints exactly** (no IK); without `q` it IKs from the Cartesian pose (sequential seed). `plan_segments` is `(q_start, q_goal, grip)`; `_play` actuates the gripper only when state differs, settling `GRIP_PREDELAY_S` first (real gripper only on an executed play, viz tweens either way).
- **Save/Load:** `trajectories/<name>.json` = `{robot, created, waypoints}`. Load clears + repopulates the waypoint list and frames; then Plan to replay. (Tracked, not gitignored — they sync across machines via the repo.)
- **GoFa software free-drive:** the **"Free-drive (lead-through)"** checkbox flips a `lead_go` flag over RWS; `PyEgm.mod` calls `SetLeadThrough \On` and holds the arm compliant until you untick (`SetLeadThrough \Off`). Mutually exclusive with Plan/Play/Live; Stop/untick release it; the controller auto-clears on motors-off. The **physical lead-through button** still works too. Capture reads the hand-moved joints via RWS polling. No gripper actions. **Requires re-running `install_gofa_egm.py`** to push the updated supervisor (see PyEgm.mod lead-through caveats).

## Headless replay — `play_trajectory.py`

```bash
./robot_control/bin/python scripts/play_trajectory.py <name> [more names...] [--speed S] [--dry-run] [--no-confirm]
```

Reads `trajectories/<name>.json`, auto-detects the robot from its `"robot"` field, executes after a `[y/N]` confirm (`--no-confirm` to skip). `--dry-run` prints the plan (segments + estimated duration) and exits. **IK-solver-free**: every waypoint must already carry `"q"` (from Capture, or Plan-and-save in viser) — a `q`-less waypoint aborts with a "Plan + re-save" message. First segment moves from the current pose to waypoint 1. UR15 path mirrors `teleop_ur15.py` (servoJ + settle + gripper-on-change with the 0.5 s pre-delay, gripper opened at start). GoFa path imports pyroki for FK **only** to enforce the `MAX_TCP_SPEED` cap, then streams over the existing EGM supervisor (PyEgm parked at `WaitUntil egm_go`). Profile + connection constants come from `robot_common.py`, so retuning a teleop script updates this one too.

**Chaining.** Pass several names to play them back-to-back as **one continuous motion**: waypoint lists are concatenated, so each seam is just another move segment and gripper state carries across. One confirm, one Hand-E calibration at the start (the slow part), one final settle. All chained trajectories must target the same robot (mixing UR15 + GoFa aborts). Implemented purely in `main()` — builds a synthetic combined `{robot, waypoints}` for the unchanged single-trajectory player.

## Headless recording — `teleop.py`

The record-side counterpart to `play_trajectory.py`; same `trajectories/<name>.json` format, replays unchanged.

```bash
./robot_control/bin/python scripts/teleop.py [name] [--robot ur|gofa]
```

Missing `name`/`--robot` are prompted for. The robot connects once and is reused for the session. The arm enters free-drive (UR `teachMode()`; GoFa `lead_go=TRUE` → `SetLeadThrough`) and a raw-keypress loop runs:

| Key | Action |
|---|---|
| `c` | Capture a waypoint: live joints `q` + FK grasp `pos`/`wxyz` + current gripper `grip` (UR). |
| `o` / `p` | UR only: open / close the gripper one `GRIP_STEP` (10%), live. No-op on GoFa or if unreachable. |
| `Enter` | End free-drive, save, then prompt for the next trajectory (same robot). 0 waypoints → nothing saved. |
| `w` | **Soft stop:** end free-drive cleanly and exit. |
| `q` | **Hard stop:** UR → `triggerProtectiveStop()` + `stopScript`; GoFa → clear `lead_go`/`egm_go` + `stop_program()` (halts RAPID + drops lead-through, no software motors-off over RWS — recover via PP-to-Main + Play or re-running the installer). Then exit. |

A live terminal dashboard redraws in place at `DASH_HZ` (10 Hz): grasp pose (XYZ mm + RPY deg), six joint angles, gripper %, waypoint count, last-action status. Single `key_loop` polling `stdin` non-blocking via `select` (no arrow-key escapes — hence stop is `w` not `Esc`); cursor-up + clear-to-EOL keeps it flicker-free. Blank name + Enter exits. Terminal uses `tty.setcbreak` (raw only during the key loop; prompts run cooked), restored on every exit path. FK reuses the jaxlie + pyroki path (UR × `TOOL0_T_GRASP`; GoFa uses `tool0`), warmed at startup. Two backends (`URBackend`, `GoFaBackend`) behind one loop. GoFa lead-through carries the same caveats as `teleop_gofa_egm.py`. Design spec: `docs/superpowers/specs/2026-06-04-cli-trajectory-recorder-design.md`.

---

# GoFa CRB 15000

## Controller setup (one-time, ~20 minutes)

More involved than the UR because ABB controllers always run a RAPID program for motion — Python doesn't talk to the motion executor directly; it pokes RAPID variables and a RAPID `WHILE TRUE` loop does the work.

### Hardware — safety jumpers

The OmniCore C30 ships with the safeguard chain expecting an external safety device. For a standalone benchtop GoFa, physical jumpers on the **X14** terminal block:

| Pair | Function | Status (lab install) |
|---|---|---|
| ES1 (pins 1–2) | Emergency stop ch 1 | jumpered |
| ES2 (pins 3–4) | Emergency stop ch 2 | jumpered |
| AS1 (pins 5–6) | Auto stop / safeguard ch 1 | **must be jumpered** |
| AS2 (pins 7–8) | Auto stop / safeguard ch 2 | **must be jumpered** |

Symptom of missing AS jumpers: Manual mode works (the pendant enabling grip switch bypasses AS), Auto mode immediately fires a "guard stop / protective stop circuit open". Check the **specific** event log entry — if it says AS1/AS2 or "protective stop circuit", jumper them. Safety: jumpering AS bypasses external safeguarding inputs; built-in cobot collision detection still protects, but **don't** run high-speed Auto motion with people in the workspace.

### Network

OmniCore C30 has three logical networks; pick one for RWS access:

| Logical network | Physical port | What lives there |
|---|---|---|
| Private / MGMT | **MGMT** (rightmost of three at bottom-right) | RWS at `192.168.125.1`, FlexPendant, RobotStudio direct |
| Public / WAN | **WAN** (next to MGMT) | RWS on plant subnet, IP configurable from pendant |
| I/O Network | **LAN** + X1–X5 ETHERNET SWITCH | EtherNet/IP, Profinet, fieldbus only — **not RWS** |

Lab use: Mac → MGMT direct via Ethernet, Mac's interface set to a static `192.168.125.x` (anything except `.1`).

⚠️ **VPN gotcha.** If your Mac is on a VPN claiming `192.168.125.0/24` (e.g. a corporate VPN), packets to the GoFa route into the tunnel instead of out the Ethernet cable — connect-refused/timeouts on every port. Disconnect the VPN, or use the WAN port on a non-conflicting subnet (e.g. `192.168.0.102` if your lab network is `192.168.0.0/24`).

```bash
ping -c 2 192.168.125.1                     # should reply, ~0.5 ms
nc -zv -G 2 192.168.125.1 443               # "succeeded" (OmniCore is HTTPS)
route get 192.168.125.1 | grep interface    # should be en* (Ethernet), NOT utun* (VPN)
```

### Pendant: enter Auto mode

OmniCore C30 has **no physical Auto/Manual key switch**. Mode is on the FlexPendant touchscreen — a small mode-icon in the top status bar → Automatic → confirm. A separate white motors-on button on the cabinet front may need one press.

### Push the EGM supervisor

EGM is a **licensed** option — confirm it's enabled (pendant: Settings → System → installed options) first.

```bash
./robot_control/bin/python scripts/install_gofa_egm.py
```

What it does:
1. Connects to RWS over HTTPS at `ROBOT_IP` (default `192.168.125.1`); probes controller state, opmode, exec state.
2. Grabs RAPID mastership, stops any running program.
3. **Unloads `MainModule`** — it ships with a `PROC main()` that collides with ours.
4. Uploads `EGM_COMM.cfg` + `EGM_MOC.cfg` and `PyEgm.mod` to `$HOME/` on the controller.
5. Loads `PyEgm.mod` into task `T_ROB1`, turns motors on if off.
6. Tries `resetpp` + `start_program` over RWS; when that fails (see gotchas) it prompts you to tap **PP to Main** + green **Play** on the pendant.

⚠️ The `.cfg` files (in `egm/`) define the EGM UDP peer (`UCdevice` → your PC). `egm/EGM_COMM.cfg` `RemoteAddress` must equal your PC's IP (default assumes `192.168.125.50`; edit it and `PC_IP` in the installer to match) and `RemotePortNumber` must equal `EGM_LOCAL_PORT` (6510). Applying the `.cfg` may need a controller restart from the pendant.

After this, `PyEgm` is parked at `WaitUntil egm_go = TRUE OR lead_go = TRUE`. Setting `egm_go = TRUE` (which `teleop_gofa_egm.py` does via RWS) makes it enter `EGMRunJoint` and follow the UDP joint stream until convergence, then clear `egm_go` and re-park. Setting `lead_go = TRUE` instead calls `SetLeadThrough \On` (hand-guiding) until `lead_go` clears.

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
- `\MaxSpeedDeviation := 20` — controller-side per-joint speed cap (deg/s). Backstop to the Python `MAX_TCP_SPEED` cap; raise both together.
- `\LpFilter := 20` — low-pass cutoff (Hz); lower = smoother but laggier.
- `\CondTime := 1` — seconds of convergence before `EGMRunJoint` returns (how the session ends after the final target is held).

**Lead-through (`SetLeadThrough`) caveats — verify on hardware.** The RAPID hand-guiding instruction (3HAC050917-001 / RW7 3HAC065038). Two unknowns to confirm on the actual GoFa: (1) **RW6** documents it as YuMi-only — RW7/OmniCore reportedly extends it to GoFa, but if the controller rejects it the build error shows at `/rw/rapid/tasks/T_ROB1/program/builderror` (check after the installer loads `PyEgm`); (2) whether it engages in **Auto** (EGM needs Auto) or requires Manual + enabling device — the physical lead-through button working in your Auto setup is a good sign. If it can't run in Auto, teach in Manual and switch to Auto to replay. On failure the physical button path is unaffected.

## Architecture of `teleop_gofa_egm.py`

Same shape as `teleop_ur15.py`, with EGM in place of servoJ:
- **State polling** via `rws.get_joints()` at ~10 Hz (idle viz only). During an Execute play the loop drives the URDF from the streamed target.
- **Execute mode** streams joint targets over EGM (UDP) at `STREAM_HZ`; the same `q` from the alpha profile goes to both viser and the stream every tick. A play: (1) sets `egm_go = TRUE` and waits for the first EGM feedback packet; (2) streams targets segment by segment with a short dwell between; (3) holds the final target for `HOLD_AFTER_PLAY_S` so RAPID's `\CondTime` convergence fires and `EGMRunJoint` exits, clearing `egm_go`.
- **Speed is unified and TCP-capped.** `_cap_seg_duration()` stretches each segment so real TCP speed never exceeds `MAX_TCP_SPEED` (measured against URDF kinematics); the slider can only scale *below* that cap.
- Startup safety: sets `egm_go = FALSE` on connect so a stray TRUE doesn't fire EGM.

## Tunables — `teleop_gofa_egm.py`

Shared ones live in `robot_common.py` (`GOFA_*` / unprefixed); GoFa-only knobs are local.

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

These bit us during bring-up. Captured so future-us doesn't relearn them:

- **HTTPS only, port 443.** Port 80 is disabled by default. Use `https=True` (default in `abb_rws.RWSClient`). Cert is self-signed; we set `verify=False` and suppress `InsecureRequestWarning`.
- **HTTP Basic auth, NOT Digest.** OmniCore RWS 2.0 advertises `WWW-Authenticate: Basic`. `abb_robot_client` uses Digest (for IRC5/RW6) and silently 401s against OmniCore. `abb_rws.RWSClient` uses Basic by default; pass `auth_scheme="digest"` for legacy IRC5.
- **Every POST/PUT must have `Content-Type: application/x-www-form-urlencoded;v=2.0`** — note the `;v=2.0` suffix. Without it you get `406 Not Acceptable`. `_post()` sets this automatically.
- **Mastership URL has action in the path, not the body.** OmniCore: `POST /rw/mastership/edit/request`. IRC5 was `POST /rw/mastership/edit` body `action=request`. Many other endpoints still use action-in-body — no consistent convention.
- **RAPID symbol data URL needs the module name in the path.** OmniCore: `POST /rw/rapid/symbol/RAPID/{task}/{module}/{var}/data` body `value=...`. IRC5 was `POST /rw/rapid/symbol/data/RAPID/{task}/{var}?action=set`.
- **Mastership orphans easily.** If a process holding mastership crashes, the controller thinks it's held until session timeout. Symptom: subsequent `request` calls 403. Fix: reboot from the pendant, or wait ~30s. `abb_rws.RWSClient.__post_init__` registers an `atexit` hook to release on clean exit.
- **`MainModule` collision.** The GoFa ships with a `MainModule.mod` that has its own `PROC main()`. RAPID won't compile two `main`s — loading `PyEgm` logs "Errors in RAPID program" event 40160. The installer unloads it first via `POST /rw/rapid/tasks/T_ROB1/unloadmod` body `module=MainModule`.
- **Build errors are visible at `/rw/rapid/tasks/T_ROB1/program/builderror`** — returns module name, row, column, error type. Far more useful than the generic event-log entry.
- **`resetpp` (PP-to-Main) and `start_program` via RWS** — we couldn't find the right shape on OmniCore. Every variation returned 400 "semantic error" or 403. The installer falls back to telling the user to press PP-to-Main + Play on the pendant. Cracking the right URL would tighten the install.
- **OmniCore C30 has no physical Auto/Manual key switch.** Mode is on the FlexPendant touchscreen (top status bar icon). Larger ABB controllers do have a key switch — don't assume.

---

# Custom IK: `pyroki_snippets/_solve_ik_seeded.py`

PyRoki's stock `solve_ik` snippet has no seed and no posture cost, so it returns *a* valid IK solution often in a distant null-space branch (wrist flipped 180°, elbow on the wrong side). `solve_ik_seeded` adds:
- `pk.costs.rest_cost(joint_var, q_seed, weight=rest_weight)` — pulls the solution toward `q_seed` (default `rest_weight=2.0`).
- `initial_vals=VarValues.make([joint_var.with_value(q_seed)])` — starts the optimizer at the seed, not at zeros.

Trade-off: at `rest_weight=2.0` pose error is ~0.5 mm. Raise to 5–10 if IK still picks wrong branches; the target drifts more but stays in the same kinematic family. Used by both teleop scripts.

---

# Other gotchas

- **Boost 1.90 breaks ur_rtde on macOS.** Boost made `Boost.System` header-only in 1.87+, so homebrew Boost 1.90 ships no `boost_system-*-Config.cmake` and ur_rtde's `find_package(boost_system CONFIG)` dies. Fix: `brew install boost@1.85` (keg-only, doesn't shadow 1.90), then `BOOST_ROOT=/opt/homebrew/opt/boost@1.85 CMAKE_PREFIX_PATH=/opt/homebrew/opt/boost@1.85 pip install ur_rtde`. Already done in `robot_control/`.
- **`pip install pyroki` doesn't exist.** Install from source: `git clone https://github.com/chungmin99/pyroki.git pyroki_src && ./robot_control/bin/pip install -e ./pyroki_src`. Already done.
- **PyRoki's `solve_ik` lives in `examples/pyroki_snippets/`, not the package** — NOT installed by `pip install -e .`. The `pyroki_snippets/` dir in the project root is a copy of `pyroki_src/examples/pyroki_snippets/` plus our `_solve_ik_seeded.py`; every script's bootstrap adds the repo root to `sys.path` so `import pyroki_snippets` works.
- **First IK call takes ~800 ms (JAX JIT compile); subsequent calls are ms.** Both teleop scripts call `_warmup_ik()` at launch (a no-op IK at the current pose) to pay this during startup instead of on the first Plan click.
- **UR15 model is brand new.** It postdates a lot of training data — if Claude says "there's no UR15", point at https://www.universal-robots.com/products/ur15/. The official ROS2 description repo supports `ur_type:=ur15`.
- **yourdfpy + `file://` URIs.** When `xacrodoc` processes a xacro to URDF, it resolves `package://` into absolute `file://...` paths. yourdfpy's `filename_handler` only fires for unresolved URIs (`package://`, plain paths) — `file://` is passed straight to trimesh, which can't open it. Fix: strip the `file://` prefix from the URDF after xacro processing. Both local URDFs are generated this way.
- **ABB `abb-robot-client` (Python) only supports IRC5.** Despite being the most-starred ABB python lib, it doesn't work on OmniCore. We wrote `abb_rws.py` ourselves; see the OmniCore RWS gotchas.
