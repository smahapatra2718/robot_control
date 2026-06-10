# Remote control API over a shared RobotController

**Date:** 2026-06-09
**Status:** Approved, ready for implementation plan

## Goal

Expose the two arms (UR15, GoFa) to a **remote service** over the network for
full read/write: monitor robot state + safety continuously, and issue control
commands. Control is **high-level goals** — the remote sends commands like
"move to pose", "play trajectory", "set gripper", "stop"; the robot-side server
runs IK, the motion profile, and the tight servo loop locally. The network
carries goals + telemetry only, never the tight control loop (robust to latency
and jitter, far safer).

The work also unifies the codebase: the motion logic currently inlined in the
teleop scripts is extracted into one reusable `RobotController` core that the API
server, the UR15 viser teleop, and the GoFa viser teleop all sit on top of. One
motion implementation, zero divergence, and — because the controller uses the
same hardware clients the sim fakes shadow — the whole API server runs offline
under `sim.py` for development and testing.

## Decisions (locked during brainstorming)

- **Control level:** high-level goals (not real-time network streaming of the servo loop).
- **Code architecture:** extract a shared `RobotController` core AND migrate the
  existing entry points onto it (one implementation, no divergence).
- **Transport:** HTTP + WebSocket (FastAPI + uvicorn); REST-style commands, a WS for telemetry.
- **Process model:** one process owns the hardware and hosts BOTH the remote API
  and the local viser viewer/control.
- **Safety:** explicit `/stop` + `/estop`, plus a heartbeat watchdog (deadman).
- **Auth:** bearer token on the LAN (TLS optional, out of scope here).

## Architecture

```
                     ┌────────────────────────────┐
   remote service ──►│  FastAPI app  (HTTP + WS)   │─┐
   curl/browser  ───►│  /state /move /play /stop … │ │
                     └────────────────────────────┘ │
   local operator ──► viser scene / gizmo  ──────────┤
                                                     ▼
                                        ┌──────────────────────────┐
                                        │   RobotController (core)  │  single hardware owner
                                        │  state loop · cmd executor│
                                        │  lease · watchdog · safety│
                                        └──────────────────────────┘
                                                     │ same hardware clients
                                ┌────────────────────┴───────────────────┐
                          URController (RTDE+Hand-E)         GoFaController (EGM+RWS)
                                │                                         │
                          rtde_control/receive, hande_gripper      abb_rws, abb_egm
                                                     ▲
                                    sim.py injects fakes here → whole API runs offline
```

The `RobotController` is the single owner of the hardware client and the only
thing that runs motion. The viser teleop and the FastAPI app are **surfaces**
that call controller methods. `sim.py api ur15` runs the entire API server
offline against the fakes.

## `RobotController` interface

Base class `RobotController` with concrete `URController` / `GoFaController`
subclasses — fuller versions of the `URBackend` / `GoFaBackend` pattern already
in `scripts/teleop.py`. Thread-safe; owns a state-poll thread and a single
command-executor thread.

### Reads

`get_state() -> RobotState`, a lock-free snapshot of the latest poll:

```
RobotState:
  ts: float                       # monotonic timestamp of the snapshot
  robot: str                      # "ur15" | "gofa"
  q: list[float]                  # 6 joint angles (rad)
  pose: {pos: [x,y,z], wxyz: [w,x,y,z]}   # grasp/EE pose (FK)
  gripper_frac: float | None      # 0=open..1=closed; None if no gripper (GoFa)
  safety_state: str               # e.g. "NORMAL"/"PROTECTIVE_STOP" (UR) | "motoron"/"guardstop" (GoFa)
  controller_state: str           # robot-reported controller/exec state
  activity: str                   # "idle" | "moving" | "playing" | "live" | "freedrive" | "stopped"
  active_command: {id, kind, progress} | None
  conn_ok: bool                   # last hardware read succeeded
  health: dict                    # transport-specific (egm rx/tx + freshness, rtde mode, ...)
```

`RobotState` is a stdlib dataclass with a `to_dict()` for JSON. The controller
maintains the latest `RobotState` under a lock, written by the state-poll thread
at `POLL_HZ`; `get_state()` returns a copy.

### Commands (high-level goals)

All commands are submitted to the single command executor and **return a command
id immediately**. One motion runs at a time.

- `move_to_joints(q, speed) -> cmd_id` — profiled single-segment move (trapezoidal alpha, same as teleop).
- `move_to_pose(pos, wxyz, speed) -> cmd_id` — seeded IK to a joint target, then move.
- `play(waypoints | name, speed) -> cmd_id` — the trajectory player (segments + dwell + gripper-on-change + final settle), reusing the existing logic.
- `set_gripper(frac) -> cmd_id` — Hand-E (UR); on GoFa returns an "unsupported" result.
- `set_live_target(pos, wxyz)` — continuous gizmo/streaming follow primitive used by the viser **Live** mode. NOT exposed over the API in this scope; it exists so teleop migrates cleanly.
- `stop()` — graceful stop (servoStop / clear egm_go); preempts the active command.
- `estop()` — hardest stop available (UR `triggerProtectiveStop`; GoFa `stop_program` + clear flags); preempts.

Command lifecycle: `queued -> running -> (done | failed | stopped)`. A new motion
command while one is running is **rejected** (`busy`) unless the caller `stop()`s
first; `stop`/`estop` always preempt. Results are retrievable by id and pushed
over telemetry.

### Lifecycle

`connect()` (acquire the hardware: RTDE attach + payload + gripper for UR; RWS
mastership + EGM listen + safety-init for GoFa), `close()` (release cleanly —
the same teardown the teleop scripts do today). The controller is the **single
owner**; constructing two on the same arm is unsupported (the hardware itself is
exclusive).

### Concurrency model

- One **state-poll thread** writes `RobotState` at `POLL_HZ`.
- One **command-executor thread** runs the active motion; a `stop_flag` preempts it.
- `move_*` / `play` enqueue work; `stop` / `estop` set the flag and run the hardware stop.
- The base class holds the profile/segment/settle/play logic; subclasses implement
  the hardware-specific primitives: `_read_q()`, `_servo(q, dt)`, `_hold(q)`,
  `_graceful_stop()`, `_hard_stop()`, `_gripper(frac)`, `_read_safety()`, plus
  GoFa's EGM session arm/clear and UR's `initPeriod`/`waitPeriod` pacing.

## Module layout

A `lib/control/` package (this layer pulls jax/pyroki for IK/FK, so it is kept
separate from stdlib-only `robot_common`):

| File | Responsibility |
|---|---|
| `lib/control/__init__.py` | `make_controller(robot) -> RobotController`; re-exports |
| `lib/control/state.py` | `RobotState` dataclass (stdlib, JSON-serializable) |
| `lib/control/base.py` | `RobotController` ABC: profile, segment build, play, settle, command executor, state loop, lease + watchdog hooks |
| `lib/control/ur.py` | `URController` — RTDE servoJ + Hand-E + IK/FK |
| `lib/control/gofa.py` | `GoFaController` — EGM + RWS + IK/FK + the EGM handshake |
| `lib/viser_scene.py` | reusable viser scene/gizmo/waypoint builder shared by the teleop entry and the API server (factored out of the teleop scripts) |

The four existing entry points (`teleop_ur15.py`, `teleop_gofa_egm.py`,
`play_trajectory.py`, `teleop.py`) migrate to call the controller; their viser /
CLI UI stays.

## API server (FastAPI)

`lib/robot_api.py` builds the FastAPI app over a `RobotController`;
`scripts/api_server.py` is the entry that constructs the controller, mounts the
app (uvicorn), and embeds the viser viewer — one process.

| Method | Endpoint | Auth | Notes |
|---|---|---|---|
| `GET` | `/state` | token | current `RobotState` snapshot (JSON) |
| `GET` | `/health` | token | connection + controller health |
| `POST` | `/control/acquire` | token | grab the write lease → returns `lease_token`; 409 if held |
| `POST` | `/control/release` | token + lease | release the lease |
| `POST` | `/move/joints` | token + lease | body `{q:[6], speed}` → `202 {command_id}` |
| `POST` | `/move/pose` | token + lease | body `{pos:[3], wxyz:[4], speed}` → `202 {command_id}` |
| `POST` | `/play` | token + lease | body `{name}` or `{waypoints:[…], speed}` → `202 {command_id}` |
| `POST` | `/gripper` | token + lease | body `{frac}` → `202 {command_id}` (UR; 400 on GoFa) |
| `POST` | `/stop` | token | graceful stop; allowed without the lease |
| `POST` | `/estop` | token | hardest stop; allowed without the lease |
| `GET` | `/command/{id}` | token | `{id, kind, status, progress, error?}` |
| `WS` | `/telemetry` | token | pushes `RobotState` at `TELEM_HZ` + events; also the heartbeat channel |

- **Async commands:** `POST` validates + enqueues, returns `202` with `command_id`.
  Progress/terminal status is pushed over `/telemetry` and queryable at
  `/command/{id}`. No long-blocking HTTP requests (motions take seconds).
- **Telemetry events** (over the WS, alongside periodic state): `command_started`,
  `command_done`, `command_failed`, `safety_changed`, `lease_changed`.
- **Auth:** a bearer token (from `ROBOT_API_TOKEN` env / config) checked on every
  request and on WS connect (`Authorization: Bearer …`, or `?token=` for WS).

## Safety & concurrency

- **Control lease:** any authed client may read `/state` and subscribe to
  `/telemetry`; exactly one client holds the **write lease**. `move_*` / `play` /
  `gripper` require the lease token. `/stop` and `/estop` bypass the lease (any
  authed client may stop the arm). `/control/acquire` returns 409 if held; an
  optional `force=true` steals it (and stops any active motion first).
- **Heartbeat watchdog:** the lease holder must keep its `/telemetry` WS alive (the
  WS is the heartbeat). If the connection drops or no heartbeat arrives within
  `WATCHDOG_TIMEOUT_S` **while a motion is active**, the controller auto-`stop()`s
  and releases the lease. (Idle with no lease holder is fine — nothing is moving.)
- **Stop vs e-stop:** `stop` = graceful (`servoStop` / clear `egm_go`); `estop` =
  hardest (UR `triggerProtectiveStop`; GoFa `stop_program`). New motion commands
  are rejected unless `safety_state` is normal.
- **Safety surfacing:** the state-poll thread reads UR `getSafetyMode` / GoFa
  `get_controller_state`, includes it in `RobotState`, and emits `safety_changed`
  events on transitions.
- **One executor for all surfaces:** every write — remote API *and* the embedded
  local viser — goes through the controller's single command executor, so only one
  motion ever runs (a second source gets `busy` / must `stop` first). The **lease is
  an API-level gate on remote writers only**; the local viser console is privileged
  (no lease needed) and can always `stop` and take over. The local operator is thus
  the ultimate authority — exactly the "watch and intervene locally while the remote
  drives" intent of the one-process model.

## Reuse & sim integration

- The controller reuses `robot_common` (config constants, `alpha_to_s` profile,
  `load_trajectory` / `save_trajectory`) and the pyroki seeded IK.
- A new dispatcher target `api` (added to `lib/dispatch.py`'s `TARGETS`) maps to
  `scripts/api_server.py`, so `real.py api ur15` (hardware) and `sim.py api ur15`
  (offline) both work. The API server is a plain consumer of the hardware clients,
  so `install()` shimming makes the whole server run against the fakes — the API
  can be developed and tested with no robot.
- `api_server.py` needs to know which robot to serve; it reads the target from
  `sys.argv` (e.g. `api ur15`), consistent with how `play`/`teleop` parse argv.

## Testing

**Phase 1 — controller + migration (all against the sim fakes, no robot):**
- `scripts/control_smoketest.py` (stdlib-assert, like `sim_smoketest.py`): for each
  robot, `install()` the fakes, build the controller, then assert: `connect()`
  succeeds; `get_state()` returns the seeded home; `move_to_joints(target)` drives
  `get_state().q` to `target`; `play(_sample_<robot>)` completes and reaches the
  final waypoint; `stop()` preempts an in-flight move; `set_gripper(0.5)` updates
  state (UR); safety_state reads normal.
- Regression: existing `sim_smoketest.py`, `sim.py play _sample_ur15/_sample_gofa`,
  and a manual viser parity check on `sim.py ur15` / `sim.py gofa` all still pass.

**Phase 2 — API (against `sim.py api`, no robot):**
- `scripts/api_smoketest.py`: start the server (`sim.py api ur15`) in a subprocess;
  with `requests` + `websockets` assert: bad token → 401; `/state` returns a
  snapshot; `/control/acquire` → lease, second acquire → 409; `/move/joints` →
  202 + command_id, telemetry shows `command_started`→`command_done` and
  `get_state().q` reaches the target; `/play` by name completes; `/stop` preempts;
  watchdog — drop the WS mid-motion and confirm the controller stops; `/estop`
  reports a stopped safety state.

## Dependencies

Add to the `robot_control/` venv and document in README setup:
`fastapi`, `uvicorn[standard]` (brings `websockets`/`httptools`). The
`websockets` client lib (already installed) is reused for the API smoke test.

## Phasing (one spec, phased plan)

- **Phase 1 — controller core + migration.** Build `lib/control/` and
  `lib/viser_scene.py`; migrate the four entry points onto the controller
  (behavior-preserving). Ships a cleaner codebase with no new API; fully verified
  against the sim + saved trajectories.
- **Phase 2 — remote API.** Build `lib/robot_api.py` + `scripts/api_server.py`,
  the `api` dispatcher target, lease/watchdog/auth, and the API smoke test. Install
  FastAPI + uvicorn.

## Out of scope (this spec)

- Real-time network streaming of joint/pose setpoints (the servo loop stays local).
- TLS / certificate management (LAN token only; reverse-proxy for TLS if needed).
- Multi-arm coordination in one server (one controller = one arm per process;
  run two servers on two ports for two arms).
- A new browser control UI beyond the existing viser scene.
