# Offline simulator via transport shims

**Date:** 2026-06-08
**Status:** Approved, ready for implementation plan

## Goal

Let every teleop entry point run **offline** — no UR15, no GoFa, no network — so
trajectories, IK, the viser UI, and the play profiles can be exercised on the dev
Mac. The hard requirement: the simulator must **reuse the real source primitives**
(IK, trapezoidal profile, viser scene/gizmo/waypoints, TCP-speed cap, gripper viz,
state machine) so that any feature added to the teleop scripts later works in sim
**with no extra work and no divergence**.

"Offline" means no robot hardware or controller network is needed. It still uses
the `robot_control/` venv (jax / pyroki / yourdfpy / viser), because the sim runs
the *real* IK and renders the *real* URDFs — only the robot transport is faked.

## Approach

**Import-shim launcher (perfect-tracking kinematic sim).** The reusable logic in
the four entry points sits entirely *above* a small, stable set of hardware client
classes. We provide fake versions of those clients, inject them into `sys.modules`,
and then `runpy` the **real, unmodified** teleop scripts. Because the sim literally
runs `teleop_ur15.py` (etc.), feature parity is automatic and permanent.

Rejected alternatives:
- **Backend-injection refactor** (route every hardware call through a `RobotBackend`
  interface + `--sim` flag): invasive rewrite of large, tuned scripts; risks
  regressing real-robot behavior; every *future* hardware call must be routed
  through the interface or sim silently diverges.
- **Standalone sim script** (fresh viser app reusing only `robot_common` + IK):
  fails the core requirement — the UI and state machine live *inline* in
  `teleop_*.py`, not in shared functions, so a standalone app would duplicate them
  and drift out of sync as the codebase evolves.

**Symmetric dispatchers.** `sim` is conceptually `real` + the shim, so both go
through one shared dispatch path and differ only in whether fakes are injected.
This keeps a single source of truth for the target→script mapping (real and sim can
never drift in what they can launch) and makes "am I about to move a physical
robot?" unambiguous at the command line (`real.py` vs `sim.py`).

## Hardware client surface (what the fakes must mimic)

Enumerated from the four scripts. This is the entire coupling to robot hardware.

**UR path**
- `rtde_receive.RTDEReceiveInterface(ip)`: `getActualQ()`, `getSafetyMode()`,
  `disconnect()`
- `rtde_control.RTDEControlInterface(ip)`: `servoJ(q, a, v, dt, lookahead, gain)`,
  `servoStop(decel)`, `teachMode()`, `endTeachMode()`, `initPeriod()`,
  `waitPeriod(t)`, `setPayload(mass, cog)`, `stopScript()`,
  `triggerProtectiveStop()`, `disconnect()`
- `hande_gripper.HandEGripper(host, port)`: `connect()`, `close()`,
  `activate(timeout)`, `reset(timeout)`, `move(frac, speed, force)`, `open()`,
  `close_gripper()`, `wait_until_idle(timeout)`, `status()`; module also exposes
  `DEFAULT_PORT`.

**GoFa path**
- `abb_rws.RWSClient(host, user, password)`: `request_mastership()`,
  `release_mastership()`, `set_rapid_bool(var, val, task, module)`,
  `get_rapid_data(var, task, module)`, `get_joints(mechunit)`,
  `get_controller_state()`, plus unused-by-teleop but present: `set_motors_on()`,
  `reset_pp()`, `unload_module()`, `start_program()`, `stop_program()`,
  `get_operation_mode()`, `get_execution_state()`.
- `abb_egm.EGMSession(local_port)`: `start()`, `stop()`, `set_target_rad(joints)`,
  `get_feedback_rad()`, `has_feedback()`, `is_fresh(max_age_s)`, `stats()`, and
  attributes `packets_rx` / `packets_tx`.

`egm_pb2` is **not** needed — the fake `abb_egm` does no protobuf.

## File layout (4 new files; 0 changes to the implementation scripts)

```
lib/robot_sim.py     # SimWorld singleton + fake client classes + install()
lib/dispatch.py      # TARGETS map + dispatch(target, rest, sim=False)
scripts/real.py      # real.py  <ur15|gofa|play|teleop> [args] -> dispatch(..., sim=False)
scripts/sim.py       # sim.py   <ur15|gofa|play|teleop> [args] -> dispatch(..., sim=True)
```

`teleop_ur15.py`, `teleop_gofa_egm.py`, `play_trajectory.py`, `teleop.py` are
unchanged and stay directly runnable.

## `lib/robot_sim.py`

### `SimWorld` (singleton)

Holds the simulated robot state, shared by all fakes:
- `q`: 6-vector of joint angles (rad), the single source of truth.
- `lock`: guards `q` + flags.
- `flags`: dict with `egm_go` / `lead_go` (the RAPID PERS bools).
- gripper bookkeeping (`grip_frac`) — cosmetic only.
- EGM bookkeeping for the supervisor: latest target, last-distinct-target value +
  timestamp, feedback value + timestamp, `packets_rx`/`packets_tx`.

**Perfect tracking:** every command writes `q`; every read returns `q`.

### Fake clients (alias the real APIs exactly, no network)

| Fake | Method | Behavior |
|---|---|---|
| `FakeRTDEControl` | `__init__(ip, *a, **kw)` | store nothing networked |
| | `servoJ(q, a, v, dt, lookahead, gain)` | `SimWorld.q = q`; remember `dt` for pacing |
| | `initPeriod()` | return `time.monotonic()` |
| | `waitPeriod(t)` | sleep until `t + last_dt` (so `_live_loop` paces, no busy-spin) |
| | `setPayload`, `teachMode`, `endTeachMode`, `servoStop`, `stopScript`, `triggerProtectiveStop`, `disconnect` | no-op |
| `FakeRTDEReceive` | `getActualQ()` | return `SimWorld.q` (list) |
| | `getSafetyMode()` | return `1` (NORMAL) |
| | `disconnect()` | no-op |
| `FakeHandE` | all methods | instant no-op; `move(frac)` stores `grip_frac`; `status()` returns a sane dict; module exposes `DEFAULT_PORT` |
| `FakeRWS` | `get_joints(mechunit="ROB_1")` | return `SimWorld.q` (list, rad) |
| | `get_controller_state()` | return `"motoron"` |
| | `set_rapid_bool(var, val, ...)` | write `SimWorld.flags[var]` |
| | `get_rapid_data(var, ...)` | return `"TRUE"`/`"FALSE"` from flags |
| | `request_mastership`, `release_mastership`, others | no-op |
| `FakeEGM` | `start()` | start the supervisor thread (no socket bind) |
| | `stop()` | stop the supervisor thread |
| | `set_target_rad(joints)` | store target on `SimWorld` |
| | `get_feedback_rad()` / `has_feedback()` / `is_fresh(age)` | served from supervisor feedback bookkeeping |
| | `packets_rx` / `packets_tx` | read from `SimWorld` |

### Fake RAPID supervisor (mimics `PyEgm.mod`)

A daemon thread (owned by `FakeEGM`, started in `start()`, stopped in `stop()`)
that reproduces the controller-side `EGMRunJoint` handshake the GoFa scripts depend
on. `COND_TIME = 1.0` mirrors `PyEgm.mod`'s `\CondTime := 1`.

Loop at ~200 Hz, under `SimWorld.lock`:
- On `egm_go` rising edge (FALSE→TRUE): reset `last_change_time = now`,
  `last_applied_target = None` (so convergence isn't declared before streaming).
- While `egm_go` is TRUE:
  - mark feedback fresh: `feedback = q`, `feedback_time = now`; bump
    `packets_rx`/`packets_tx`.
  - if a target exists: apply it (`q = target`); if its *value* changed since
    `last_applied_target`, set `last_change_time = now`.
  - if `now - last_change_time >= COND_TIME`: set `egm_go = FALSE` (EGMRunJoint
    converged and returned).
- While `egm_go` is FALSE: do **not** refresh `feedback_time`, so `is_fresh()`
  goes stale shortly after — matching the real controller, which stops streaming
  after EGMRunJoint exits (Live mode relies on this to re-arm).
- `lead_go`: no-op (arm "compliant"; `q` unchanged).

This makes the GoFa execute path complete end to end:
`_start_egm_session` (set `egm_go=TRUE`, wait for `is_fresh`) → stream targets →
hold final target for `HOLD_AFTER_PLAY_S` (1.5 s > COND_TIME) → `_wait_egm_clear`
sees `egm_go` cleared.

### `install(robot_hint)`

1. Build `types.ModuleType` shims for `rtde_control`, `rtde_receive`,
   `hande_gripper`, `abb_rws`, `abb_egm`, each exposing the matching fake class
   (and `hande_gripper.DEFAULT_PORT`); insert into `sys.modules` **before** the
   target script runs.
2. Seed `SimWorld.q` to a per-robot home pose (see below).

Everything not in that shim list — `robot_common`, pyroki, jax, jaxlie, yourdfpy,
viser, `egm_pb2` — stays **real**.

### Home poses

A sane, non-singular starting config per robot, defined as tunable constants in
`robot_sim.py` (`UR_HOME`, `GOFA_HOME`). `install("ur15")` seeds `UR_HOME`,
`install("gofa")` seeds `GOFA_HOME`; `play` / `teleop` (robot known only at
runtime) seed a neutral default. Perfect-tracking means the home pose only sets the
starting joints before the first command, so exact values are not critical.

## `lib/dispatch.py`

```python
TARGETS = {"ur15": "teleop_ur15.py", "gofa": "teleop_gofa_egm.py",
           "play": "play_trajectory.py", "teleop": "teleop.py"}

def dispatch(target, rest, sim=False):
    if target not in TARGETS: <usage error>
    if sim:
        import robot_sim          # lazy: real path never imports sim machinery
        robot_sim.install(target)
    script = <scripts dir>/TARGETS[target]
    sys.argv = [TARGETS[target], *rest]   # play/teleop argparse sees the right argv
    runpy.run_path(script, run_name="__main__")
```

`dispatch` resolves the scripts dir from `__file__` (lib/ is alongside scripts/ via
the repo root), independent of CWD.

## `scripts/real.py` / `scripts/sim.py`

Each ~3 lines after the standard `sys.path` bootstrap: read `target = argv[1]`,
`rest = argv[2:]`, call `dispatch(target, rest, sim=False)` / `(…, sim=True)`. A
missing/invalid target prints the usage (`<ur15|gofa|play|teleop>`).

## UX

```bash
./robot_control/bin/python scripts/real.py ur15          # real UR15 teleop (viser)
./robot_control/bin/python scripts/sim.py  ur15          # same, simulated offline

./robot_control/bin/python scripts/real.py gofa          # real GoFa teleop (viser)
./robot_control/bin/python scripts/sim.py  gofa          # same, simulated offline

./robot_control/bin/python scripts/real.py play traj1 traj2 --speed 0.5
./robot_control/bin/python scripts/sim.py  play traj1 traj2 --speed 0.5

./robot_control/bin/python scripts/real.py teleop myname --robot ur
./robot_control/bin/python scripts/sim.py  teleop myname --robot ur
```

## What works in sim vs. the one limitation

**Works fully:** gizmo drag → Plan → Play (viz follows the planned trajectory
exactly), waypoint add/capture/save/load, Live gizmo-follow, gripper slider +
per-waypoint grip viz, the GoFa EGM execute handshake, headless `play_trajectory`
(real end-to-end segment/profile/settle logic), TCP-speed capping.

**Limitation — free-drive/teach cannot be hand-guided** (no physical arm to move).
In sim, `teachMode`/lead-through are no-ops, so "Capture" during free-drive just
re-grabs the current pose. Offline authoring is via the **gizmo + Add waypoint**,
**Live**, or **Plan** — the tools that don't need a hand. The headless `teleop.py`
recorder still runs (dashboard, keys, save all work) but captured points stay at
the home pose; it is a plumbing test, not realistic authoring. Documented, not
faked.

## Verification

**Automated**
- Unit-style smoke test that calls `robot_sim.install(...)` and drives the fakes
  directly: `servoJ` → `getActualQ` round-trips to the same `q`; the GoFa
  handshake (`set_rapid_bool("egm_go", True)` → `EGMSession.is_fresh()` becomes
  True → stream targets → hold → `egm_go` auto-clears after `COND_TIME`).
- End-to-end headless: `sim.py play <fixture> --no-confirm --speed 5` against a
  saved trajectory — exercises the real segment building, alpha profile, dwell, and
  settle loop with no robot. (Pick/commit a small fixture trajectory if none of the
  existing `trajectories/*.json` carry per-waypoint `q`.)

**Manual checklist**
- `sim.py ur15` and `sim.py gofa`: viser loads, IK warmup completes, gizmo → Plan →
  Play animates the URDF, Live follows the gizmo, gripper slider animates (UR).

## Docs

Update `CLAUDE.md` (and `README.md` if it lists run commands) to present
`real.py <target>` as the primary interface with `sim.py <target>` as its offline
twin, document the free-drive-in-sim limitation, and note the four implementation
scripts remain directly runnable.
