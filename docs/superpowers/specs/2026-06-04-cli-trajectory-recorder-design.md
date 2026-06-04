# CLI trajectory recorder — `teleop.py`

**Date:** 2026-06-04

## Goal

A headless (no viser) CLI for **recording** teach trajectories on either arm by
hand-guiding it, capturing waypoints with single keypresses, and saving in the
existing `trajectories/<name>.json` format so `play_trajectory.py` replays them
unchanged. This is the record-side counterpart to the replay-side
`play_trajectory.py`.

## Invocation

```bash
./robot_control/bin/python teleop.py [name] [--robot ur|gofa]
```

- Missing `name` → interactive prompt: `Trajectory name: `
- Missing `--robot` → interactive prompt: `Robot?  [1] UR  [2] GoFa: `
- The robot is chosen once; its connection is reused for the whole session
  (record many trajectories without reconnecting).

## Recording loop (per trajectory)

1. Get/confirm the trajectory name (first one may come from the CLI arg; later
   ones always prompt). **A blank name + Enter → safe exit** (release free-drive,
   close connections, quit).
2. Enter free-drive (hand-guiding):
   - **UR:** `rtde_c.teachMode()`.
   - **GoFa:** set `lead_go = TRUE` over RWS → `PyEgm.mod` calls
     `SetLeadThrough \On`. The physical lead-through button also works.
3. Put the terminal into raw single-key mode and run the key loop below.
4. On `Enter`: end free-drive, write `trajectories/<name>.json`, then return to
   the next-trajectory name prompt (same robot). If zero waypoints were
   captured, print "nothing to save" and skip the write.

## Key map (raw single-key, no Enter needed except where noted)

| Key | Action |
|---|---|
| `c` | Capture a waypoint at the current pose: live joints `q`, FK grasp `pos`/`wxyz`, and current gripper `grip` (UR). Prints a short confirmation (`captured waypoint N`). |
| `↑` | UR only: open gripper by one step (default 10%), clamped to 0%. Commanded to the real gripper live. No-op on GoFa. |
| `↓` | UR only: close gripper by one step (default 10%), clamped to 100%. Commanded live. No-op on GoFa. |
| `Enter` | End free-drive, save the trajectory, go to the next-name prompt. |
| `Esc` | **Soft stop:** end free-drive cleanly and exit the script (arm stays powered/idle). |
| `q` | **Hard protective stop:** UR → RTDE protective stop + `stopScript`; GoFa → clear `lead_go`/`egm_go` + motors-off via RWS. Then exit. |

Defaults the user can retune: `↑`=open / `↓`=close, gripper step = 10%.

## Output format

Identical to the existing trajectories so `play_trajectory.py` consumes it as-is:

```json
{
  "robot": "ur15",
  "created": "2026-06-04T12:00:00",
  "waypoints": [
    {"q": [6], "pos": [x,y,z], "wxyz": [w,x,y,z], "grip": 0.0}
  ]
}
```

- `robot` is `"ur15"` or `"gofa"` (matches what `play_trajectory.py` auto-detects).
- `q` = hand-guided joints read live at capture (replayed exactly; no IK needed).
- `pos`/`wxyz` = FK of the grasp point (UR: `tool0` × `TOOL0_T_GRASP`; GoFa: `tool0`).
- `grip` = current gripper fraction, UR only (0.0 open … 1.0 closed). GoFa
  waypoints omit `grip`, matching `_sample_gofa.json`.
- `created` = `datetime.datetime.now().isoformat(timespec="seconds")`.

## Implementation notes

- **Esc vs arrow keys:** arrow keys arrive as the escape sequence `ESC [ A` (up) /
  `ESC [ B` (down). On reading `ESC`, the key reader peeks with a ~50 ms `select`
  timeout: if `[` follows it decodes an arrow; if nothing follows it is a bare
  `Esc`. Standard termios + `select` approach.
- **Raw mode:** use `termios`/`tty.setcbreak` on the TTY, restored in a `finally`
  on every exit path. POSIX/macOS only (matches the dev/test environment).
- **FK for `pos`/`wxyz`:** reuse the existing jaxlie + pyroki FK approach from the
  teleop scripts. UR multiplies the `tool0` FK by `TOOL0_T_GRASP` (read from
  `hande.urdf`) for the grasp point; GoFa uses `tool0` directly. Incurs the same
  ~800 ms JAX warmup the other scripts pay.
- **UR gripper:** connect + reset + activate + wait for calibration before the
  arrows take effect — the same sequence `play_trajectory.py` uses. The Robotiq
  URCap socket (port 63352) coexists with `teachMode`. If the gripper is
  unreachable, log it and continue: arrows become no-ops and `grip` defaults to
  the last known value (0.0 = open at startup).
- **Connection reuse / config:** mirror the connection setup and the `UR_*` /
  `GOFA_*` constants already established in `play_trajectory.py` (IPs, ports, RWS
  user/password, RAPID module/flags). Keep them in sync with that file.
- **Safety:** `atexit` + `finally` always ends free-drive (UR `endTeachMode`,
  GoFa clear `lead_go`) so no exit path — including exceptions or `q` — leaves the
  arm in a loose/compliant state.

## GoFa caveat

GoFa free-drive depends on the EGM supervisor (`PyEgm.mod`) being installed and
parked, exactly like `teleop_gofa_egm.py`, and on `SetLeadThrough` working in the
current mode (see the lead-through caveats in `CLAUDE.md`). No new controller-side
changes are required — this reuses the existing `lead_go` path.

## Out of scope

- No viser / 3D viz (that is what the teleop scripts are for).
- No IK / planning (recording stores joints directly; replay is `play_trajectory.py`).
- No editing/trimming of captured waypoints beyond capture-as-you-go.
- No Windows support (POSIX raw-mode terminal only).
