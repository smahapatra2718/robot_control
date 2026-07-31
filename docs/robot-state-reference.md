# `RobotState` field reference

Exhaustive reference for the object returned by `GET /state` and streamed frame-by-frame
over `WS /telemetry` (**identical shape** — the WS sends `RobotState.to_dict()` too).
Both surfaces also accept `?flat=1` for a single-level, scalar-only variant of the same
data — see [Flat shape](#flat-shape--flat1) at the bottom.

Every value below was read out of the code, with the source line given so it stays checkable.
The struct is `lib/control/state.py`; it is filled once per poll by `RobotController._poll_once`
(`lib/control/base.py:157`) at `POLL_HZ` — **30 Hz on the UR15**, **10 Hz on the GoFa**
(`ur.py:30`, `gofa.py:22`).

```jsonc
{
  "ts": 1191980.218958208,
  "robot": "ur15",
  "q": [0.0, -1.0, 1.0, 0.0, 1.0, 0.0],
  "pose": { "pos": [1.1186, 0.3444, 0.6274], "wxyz": [-0.3390, -0.3390, -0.6205, -0.6205] },
  "gripper_frac": 0.0,
  "safety_state": "NORMAL",
  "controller_state": "1",
  "activity": "idle",
  "active_command": {              // never null; all-null whenever nothing is running
    "id": null, "kind": null, "status": null, "progress": null, "error": null
  },
  "conn_ok": true,
  "health": {}                     // UR15: always empty. GoFa: always {egm_rx, egm_tx}
}
```

---

## `ts` — float

`time.monotonic()` at the moment the snapshot was built (`base.py:158`).

- **Not** a wall-clock timestamp, and **not** comparable across processes or reboots. The
  epoch is arbitrary (on Linux, boot).
- Use it only for deltas: staleness (`now_monotonic - ts`) or frame spacing.
- If you need wall-clock, stamp it on arrival client-side.

## `robot` — string

Exactly one of:

| Value | Controller class |
|---|---|
| `"ur15"` | `URController` (`ur.py:29`) |
| `"gofa"` | `GoFaController` (`gofa.py:21`) |

Also the folder name under `trajectories/` and the URL prefix in multi-arm mode.

## `q` — array of 6 floats, radians

Always length 6 for both arms — never null, never a different length once connected.
Raw measured joint angles (UR: RTDE `getActualQ()`; GoFa: RWS `get_joints()`), **not**
the commanded target.

Order is positional and arm-specific:

| # | UR15 | GoFa |
|---|---|---|
| 0 | `shoulder_pan_joint` | `joint_1` |
| 1 | `shoulder_lift_joint` | `joint_2` |
| 2 | `elbow_joint` | `joint_3` |
| 3 | `wrist_1_joint` | `joint_4` |
| 4 | `wrist_2_joint` | `joint_5` |
| 5 | `wrist_3_joint` | `joint_6` |

Ranges, from the URDFs actually loaded at runtime:

| # | UR15 limits | GoFa limits |
|---|---|---|
| 0 | ±6.2832 rad (±360°) | ±3.1416 rad (±180°) |
| 1 | ±6.2832 rad (±360°) | ±3.1416 rad (±180°) |
| 2 | ±3.1416 rad (±180°) | −3.9270 … +1.4835 rad (−225° … +85°) |
| 3 | ±6.2832 rad (±360°) | ±3.1416 rad (±180°) |
| 4 | ±6.2832 rad (±360°) | ±3.1416 rad (±180°) |
| 5 | ±6.2832 rad (±360°) | ±3.1416 rad (±180°) |

Note the UR15's ±360° wrists: joint values are **not** wrapped to ±π, so a client that
normalizes angles will corrupt a valid configuration. GoFa `joint_3` is the asymmetric one.

## `pose` — object

```jsonc
"pose": { "pos": [x, y, z], "wxyz": [w, x, y, z] }
```

- `pos` — metres, in the **robot base frame**. 3 floats.
- `wxyz` — unit quaternion, **w first** (not xyzw as three.js/ROS use). 4 floats.
- Forward kinematics of `q`, computed locally via pyroki, so it always agrees with `q`.

**The frame differs per arm — this is the most common cross-arm bug:**

| Arm | `pose` refers to | Source |
|---|---|---|
| UR15 | the Hand-E **grasp point** — `tool0` × `TOOL0_T_GRASP`, ≈156 mm past the flange | `ur.py:91` |
| GoFa | **`tool0`** itself (the flange) | `gofa.py:71` |

So pose values are *not* portable between arms; joint values are. `POST /move/pose` takes a
target in this same per-arm frame.

Quaternion sign is not canonicalized: `wxyz` and `-wxyz` are the same orientation, and the
solver may return either. Compare orientations with `|q1·q2| ≈ 1`, never elementwise.

## `gripper_frac` — float in `[0.0, 1.0]`, or `null`

`0.0` = fully open, `1.0` = fully closed.

| Arm | Value |
|---|---|
| GoFa | **always `null`** — no gripper (`gofa.py:82`) |
| UR15 | the tracked fraction, or **`null` if the Hand-E socket is unreachable** (`ur.py:107`) |

`null` therefore means *"no gripper right now"*, **not** *"this is the GoFa"*. Feature-detect
on the value rather than on `robot`, and re-check it — a UR15 that starts with the gripper
offline reports `null` for the whole session, and `POST /gripper` 400s in that state.

It is the *commanded/tracked* value, not a measured encoder reading: it changes the instant a
gripper command is accepted, then the fingers take ≈0.8 s to actually travel (`_GRIPPER_MOVE_S`,
`ur.py:25`).

## `safety_state` — string

**UR15** — mapped from the RTDE safety mode (`_UR_SAFETY_MODES`, `ur.py:19`). The complete set:

| Mode | `safety_state` |
|---|---|
| 1 | `"NORMAL"` |
| 2 | `"REDUCED"` |
| 3 | `"PROTECTIVE_STOP"` |
| 4 | `"RECOVERY"` |
| 5 | `"SAFEGUARD_STOP"` |
| 6 | `"SYSTEM_EMERGENCY_STOP"` |
| 7 | `"ROBOT_EMERGENCY_STOP"` |
| 8 | `"VIOLATION"` |
| 9 | `"FAULT"` |
| 10 | `"VALIDATE_JOINT_ID"` |
| 11 | `"UNDEFINED"` |
| anything else | `"mode <n>"` — e.g. `"mode 12"` if UR adds one |
| read failed | `"UNKNOWN"` (and `conn_ok: false`) |

⚠️ **`POST /estop` produces `"PROTECTIVE_STOP"` (mode 3), not an emergency-stop value.** It calls
RTDE `triggerProtectiveStop()` (`ur.py:151`) — the strongest stop software can command. Modes 6
and 7 come from the **physical** e-stop circuit only: asserting that chain from software is
impossible by design, which is the point of a hardware e-stop. So if you press the console's
E-STOP and expect `safety_state` to read `ROBOT_EMERGENCY_STOP`, it never will; check for
`PROTECTIVE_STOP`, or just watch `activity` go to `"stopped"`. A protective stop is also
**not clearable over RTDE** — release it from the pendant.

**GoFa** — the raw RWS controller state from `/rw/panel/ctrl-state` (`abb_rws.py:90`), which is
**also copied into `controller_state`** because RWS exposes only that one signal
(`gofa.py:61`). Typical values are ABB's own vocabulary: `"init"`, `"motoron"`, `"motoroff"`,
`"guardstop"`, `"emergencystop"`, `"emergencystopreset"`, `"sysfail"` — lowercase, and
controller-defined rather than enumerated by us. On a read failure: `"UNKNOWN"`.

**Do not write a client that treats these two vocabularies as one enum.** There is no shared
"is the arm OK" value; branch on `robot` first, or just display the string.

## `controller_state` — string

| Arm | Value |
|---|---|
| UR15 | `str(<numeric safety mode>)` — e.g. `"1"`, `"3"`. A *number in a string*, not a name (`ur.py:87`) |
| GoFa | identical to `safety_state` (same RWS signal) |
| read failed | `"?"` (and `conn_ok: false`) |

On the UR15 this is strictly redundant with `safety_state` (the name vs its number); on the
GoFa it is literally the same string. It carries no extra information today on either arm.

## `activity` — string

Derived every poll by `_activity()` (`base.py:167`), in this precedence order:

| Value | When | Precedence |
|---|---|---|
| `"moving"` | a command with `kind: "moving"` is `running` (move/joints, move/pose, gripper) | 1 |
| `"playing"` | a command with `kind: "playing"` is `running` (`/play`) | 1 |
| `"freedrive"` | no running command **and** free-drive is engaged | 2 |
| `"stopped"` | no running command, not compliant, **and** the stop flag is still set | 3 |
| `"idle"` | none of the above | 4 (default) |

That is the complete set — five values, no others. Two consequences:

- `"moving"`/`"playing"` are exactly the two `kind` values, so `activity` mirrors
  `active_command.kind` while a command runs.
- `"stopped"` is **sticky** until the next command is submitted — `_submit` clears the stop
  flag (`base.py:193`), so an arm sits in `"stopped"` indefinitely after a `/stop`. It does not
  decay back to `"idle"` on its own. Treat it as "last motion was cut short", not as a fault.

**Documented race:** the stop flag is read outside the command lock, so `activity` and
`active_command` can disagree for one poll cycle at a stop/start boundary (`base.py:168`).
Don't assert consistency between them in client logic.

## `active_command` — object (never `null`)

**Always an object with all five sub-keys present**, from the first poll of the session. It
carries the **running** command and nothing else: the moment that command reaches a terminal
state, every value goes back to `null` (`empty_command()`, `state.py:8`, selected in
`_poll_once` at `base.py:160`). Idle after a command therefore looks identical to fresh boot.

```jsonc
// idle — at boot, and again the instant a command ends
"active_command": { "id": null, "kind": null, "status": null, "progress": null, "error": null }

// mid-motion — the only time it is populated
"active_command": { "id": 1, "kind": "moving", "status": "running", "progress": 0.4, "error": null }
```

Measured transition on a UR15 move (sim, 30 Hz poll):

```
t=0.00s  id=None  status=None      progress=None  activity=idle
t=0.02s  id=1     status=running   progress=0.0   activity=moving
t=0.62s  id=1     status=running   progress=1.0   activity=moving
t=0.88s  id=None  status=None      progress=None  activity=idle
```

Three traps:

- **Don't test the object's truthiness.** `if (state.active_command)` is always true — it's
  always an object. Test `active_command.id !== null`, or better, `status === "running"`.
- **`status` is only ever `"running"` or `null` here.** Note the trace above: it goes
  `running` → all-`null` with no intervening tick. **`done` / `failed` / `stopped` never appear
  in `/state` at all**, so the outcome of a command — and any `error` string — is *not*
  observable from state polling or the telemetry stream. `GET /command/{id}` retains it (backed
  by `_cmd_history`, `base.py:222`) and is the only source. If you submit a command whose result
  matters, hold its id and poll it. `activity` will tell you the arm went idle; only
  `/command/{id}` will tell you whether it succeeded.
- **It lags `GET /command/{id}` by up to one poll period.** The command endpoint reads the live
  record, while this is a copy taken by the state-poll thread (`base.py:156`). So a command can
  report `done` on `/command/{id}` while `/state` still shows it `running` — up to 33 ms on the
  UR15, 100 ms on the GoFa. Another reason to treat `/command/{id}` as authoritative.

The sub-key table below describes the values while a command is running.

| Field | Type | Values |
|---|---|---|
| `id` | int | Monotonic from **1**, `itertools.count(1)` (`base.py:111`). Per controller, so `/ur15` and `/gofa` number independently and both start at 1. |
| `kind` | string | `"moving"` (move/joints · move/pose · gripper) or `"playing"` (`/play`). **Only these two** (`base.py:257-272`). |
| `status` | string | `"running"` · `"done"` · `"failed"` · `"stopped"`. Only these four. |
| `progress` | float | `0.0` … `1.0`, in steps of `1/n` for `n` segments. See the warning below — it is coarser and less honest than it looks. |
| `error` | string or `null` | `str(exception)` when `status == "failed"`, else `null`. |

`status` transitions (`_run_cmd`, `base.py:200`) — set once, at completion:

- `running` → `done` — the motion function returned and the stop flag was clear.
- `running` → `stopped` — returned, but the stop flag was set (`/stop`, `/estop`, force-steal, or the deadman).
- `running` → `failed` — the motion function raised; `error` carries the message.

There is no `queued`/`pending` — a submitted command is `running` immediately (only one runs
at a time; a second submit raises `Busy` → HTTP 409).

### ⚠ `progress` is per-segment, and reports an interrupted segment as finished

`progress_cb((seg_idx + 1) / n)` is called at the end of each segment's iteration
**unconditionally** — including when that segment's inner loop broke early on a stop
(`ur.py:194`, `gofa.py:202`). Two consequences, both verified on a live server:

- **It never advances *within* a segment.** The value only changes at segment boundaries.
- **`/move/joints`, `/move/pose` and `/gripper` are always single-segment** (`base.py:253-272`),
  so their `progress` is only ever `0.0` or `1.0` — it carries no "how far along" information.
  A move **stopped halfway still reports `progress: 1.0`** with `status: "stopped"`. Observed:
  `{"id":1,"kind":"moving","status":"stopped","progress":1.0,"error":null}`.

So: only trust `progress` for a multi-waypoint `/play`, where `k/n` genuinely means *k of n
segments done*. For everything else use `status`, and never infer distance travelled from
`progress`. `status == "done"` additionally forces `progress` to exactly `1.0` (`base.py:214`).

**Edge case worth knowing:** if a stop arrives *after* the motion function returns but before
the status is written, the command reports `"stopped"` even though it actually completed. This
is a deliberate trade-off, flagged in the code (`base.py:203`).

The same object is returned by `GET /command/{id}`, which reads from a **bounded history of
the 64 most recent** commands (`_CMD_HISTORY_MAX`, `base.py:100`); older ids evict and then
return `404 {"detail":"unknown command id"}`.

### ⚠ Polling `/command/{id}` does not keep the deadman happy

`GET /command/{id}` is an auth-only read — it carries no `X-Lease`, so it does **not** refresh
the lease heartbeat. A client that acquires a lease, submits a motion, then sits in a
`/command/{id}` poll loop gets its motion killed by the watchdog after `watchdog_timeout_s`
(2 s). Observed while writing this doc: a 3-segment `/play` reached `progress: 0.3333` and then
flipped to `status: "stopped"` on its own, with nothing else touching the arm.

Hold a `WS /telemetry?token=…&lease=…` open for the duration of any motion. That is the
intended heartbeat, and it also carries `active_command`, so you don't need the poll loop.

## `conn_ok` — boolean

`true` if the last hardware read in the poll loop succeeded; `false` if it raised — in which
case `safety_state` is `"UNKNOWN"` and `controller_state` is `"?"` (`ur.py:88`, `gofa.py:68`).

Note what it does *not* cover: it reflects the **state read** only (RTDE `getSafetyMode` /
RWS `ctrl-state`). On the GoFa it says nothing about whether the EGM UDP path works — that is
what `health` is for. `q` and `pose` keep their last good values when `conn_ok` is `false`, so
a stale-but-plausible pose is exactly the failure mode to watch for; pair it with `ts`.

## `health` — object

Transport-specific extras. The key set is **fixed per arm** — it never changes at runtime, so
you can walk `health.egm_rx` on a GoFa without guarding — but it differs *between* arms, so
don't write one parser that assumes both.

| Arm | Contents |
|---|---|
| UR15 | `{}` — always empty, on both the success and failure paths (`ur.py:88`) |
| GoFa | `{"egm_rx": <int\|null>, "egm_tx": <int\|null>}` — always both keys; cumulative EGM UDP packet counts, `null` before an EGM session object exists (`_health()`, `gofa.py:62`) |
| either, on read failure | same keys as above, with `conn_ok: false` |

`egm_rx` / `egm_tx` are the single most useful GoFa diagnostic: during a motion **both should
be climbing**. `egm_rx` flat means the controller isn't sending to this host (check
`UCdevice`'s `RemoteAddress` against this machine's IP, and `RemotePortNumber` = 6510).
Counters are cumulative and never reset, so sample twice and diff.

---

## Flat shape — `?flat=1`

`GET /state?flat=1` and `WS /telemetry?…&flat=1` serialize the same snapshot through
`RobotState.to_flat_dict()` (`state.py:25`) instead of `to_dict()`, giving a **single-level
object whose every value is a scalar** — no arrays, no sub-objects. It exists for clients that
can't walk nested JSON (PLCs, LabVIEW, shell pipelines, spreadsheet importers).

It is purely additive: omitting `flat`, or `flat=0`, returns the nested shape documented above,
byte-for-byte unchanged. On the WS the accepted spellings are `1`, `true`, `yes`, `on`
(case-insensitive, `robot_api.py:288`); `GET /state` uses FastAPI's own bool coercion.

**Types are not stringified.** Floats stay floats, `conn_ok` stays a bool, `command_id` stays an
int. Only the nesting is removed — so every range, unit and enum documented above still applies
verbatim to the flattened key.

### Key mapping

| Nested | Flat | Notes |
|---|---|---|
| `ts`, `robot`, `gripper_frac`, `safety_state`, `controller_state`, `activity`, `conn_ok` | unchanged | already scalars |
| `q[i]` | `q_0` … `q_5` | index order is the joint order tabled above |
| `pose.pos` | `pose_pos_x`, `pose_pos_y`, `pose_pos_z` | metres |
| `pose.wxyz` | `pose_wxyz_w`, `pose_wxyz_x`, `pose_wxyz_y`, `pose_wxyz_z` | quaternion, `w` first |
| `active_command.<k>` | `command_id`, `command_kind`, `command_status`, `command_progress`, `command_error` | all `null` whenever nothing is running |
| `health.<k>` | `health_<k>` | e.g. `health_egm_rx` |

### Stability of the key set

The flat shape inherits the nested one's guarantee: every key is present on **every frame**, for
the life of the connection — an absent value is `null`, never a missing key. So:

- `gripper_frac` is present-and-`null` on the GoFa (and on a UR15 whose Hand-E socket is down).
- The five `command_*` keys are present-and-`null` whenever nothing is running, so a client can
  read `command_progress` unconditionally.
- `health_*` keys are fixed per arm: a GoFa always has `health_egm_rx`/`health_egm_tx` (`null`
  before an EGM session), a UR15 never has any.

The one cross-arm difference is that last bullet: the **core** keys are identical on both arms,
but `health_*` is not, so a single reader pointed at both arms should treat `health_*` as
optional columns. (`api_smoketest.py:test_state_flat` asserts exactly this — core keys identical
across arms; `test_state_stable_shape` asserts the full nested key set is unchanged between idle,
mid-command, and post-command.)

### Sample — GoFa, idle

```jsonc
{
  "ts": 1209557.463714875,
  "robot": "gofa",
  "q_0": 0.0, "q_1": 0.0, "q_2": 0.0, "q_3": 0.0, "q_4": 1.5708, "q_5": 0.0,
  "pose_pos_x": 0.5499996542930603,
  "pose_pos_y": 0.0,
  "pose_pos_z": 0.7179996967315674,
  "pose_wxyz_w": -1.7881393432617188e-06,
  "pose_wxyz_x": 0.0,
  "pose_wxyz_y": 1.0,
  "pose_wxyz_z": 0.0,
  "gripper_frac": null,          // gripper-less arm — key present, value null
  "safety_state": "motoron",
  "controller_state": "motoron",
  "activity": "idle",
  "command_id": null,
  "command_kind": null,
  "command_status": null,
  "command_progress": null,
  "command_error": null,
  "conn_ok": true,
  "health_egm_rx": 0,            // GoFa only, and only while an EGM session exists
  "health_egm_tx": 0
}
```

---

## Cross-cutting notes

- **Shape is stable, all the way down.** All 11 keys are always present — `RobotState` is a
  dataclass serialized with `asdict()`, so no key is ever omitted — and so are the sub-keys of
  the two nested objects: `active_command` always carries its five, `health` always carries its
  arm's set. Nothing starts empty and later grows keys, so every level can be walked unguarded.
  Only values vary. (`?flat=1` is stable on the same terms.)
- **Nullable values:** `gripper_frac`, every sub-key of `active_command`, and the GoFa's two
  `health` counters. All are null-*valued*, never absent.
- **`GET /state` before the first poll** raises, surfacing as a `500`, not an empty state
  (`base.py:180`). In practice `connect()` completes a poll before the server binds.
- **Snapshots are deep copies** (`base.py:182`), so what you get is internally consistent
  within one poll — with the one `activity` / `active_command` caveat noted above.
- **Polling faster than `POLL_HZ` gains nothing** — you'll get the same snapshot with the same
  `ts` twice. The GoFa's 10 Hz is the real telemetry resolution regardless of `telem_hz` on the
  WS.

## See also

- `API.md` — endpoints, request bodies, status codes, the lease.
- `lib/control/state.py` — the dataclass itself.
