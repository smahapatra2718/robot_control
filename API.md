# Remote Control API

HTTP + WebSocket remote control for the UR15 and GoFa arms. The network carries
**high-level goals and telemetry** — never the servo loop. A `RobotController`
(`lib/control/`) owns the hardware and runs the motion loop locally; this API
(`lib/robot_api.py`, `build_app`) only adapts that controller to HTTP/WS.

- **Read** robot state + liveness.
- **Write** motion (joints / pose / play / gripper) under a single-writer **lease**.
- **Stream** state over a telemetry WebSocket that doubles as a **deadman heartbeat**.
- **Stop / e-stop** from any authenticated client, lease or not — the always-open safety path.

Runs identically against real hardware (`real.py`) or the offline sim (`sim.py`),
since both drive the same controller.

---

## 1. Running the server

```bash
# real hardware
uv run scripts/real.py api ur15          # or: gofa
# offline sim (no robot, no network)
uv run scripts/sim.py  api ur15          # or: gofa

# options
uv run scripts/real.py api gofa --host 127.0.0.1 --port 8000
```

| Flag | Default | Notes |
|---|---|---|
| `robot` (positional) | *(both)* | `ur15` or `gofa`; **omit to serve both arms** (see below). |
| `--host` | `0.0.0.0` | All interfaces. Use `127.0.0.1` for loopback only. |
| `--port` | `8000` | TCP port. |

A single-arm server connects to that arm at startup and **aborts if it can't reach it**.
On a clean shutdown (Ctrl-C / SIGINT) it closes the controller (stopping any motion).

### Serving both arms

Run `api` with **no arm** to serve the UR15 and GoFa from one server. Startup is
**tolerant** — it starts even if only one arm is reachable; an unreachable arm is
listed as unavailable rather than aborting.

- `GET /robots` → `{"robots": [{"name": "ur15", "available": true}, …]}`, and `/health`
  gains `"multi": true`. (Single-arm servers have neither — that's how a client detects the mode.)
- **Every per-arm endpoint is namespaced under `/<robot>`** — `GET /ur15/state`,
  `POST /gofa/move/joints`, `WS /ur15/telemetry`, etc. Each arm keeps its **own
  independent lease, watchdog and telemetry**. Each also has its own Swagger at `/<robot>/docs`.
- The bundled web console auto-detects this mode and shows a per-arm switcher.

Single-arm servers (`api ur15`) keep everything at the root (`/state`, `/move/joints`, …) — unchanged.

### Web console & static assets

The server serves the static console at **`GET /`** plus its assets at **`/vendor/…`**
(vendored three.js + urdf-loader) and **`/models/<arm>/…`** (baked glTF arm bundles) —
all unauthenticated, like the dashboard shell itself; the bearer token still gates every
state/control endpoint. The console's 3D viewer reads live joint state from the existing
`WS /telemetry` — it adds **no new API endpoint**. Its optional target **gizmo** is likewise
client-side only: dragging it just fills the console's pose fields, and the motion goes out
over the ordinary lease-gated `POST /move/pose`.

### Auth token

The bearer token is read from the **`ROBOT_API_TOKEN`** environment variable:

```bash
ROBOT_API_TOKEN='a-long-random-secret' uv run scripts/real.py api ur15
```

If unset it defaults to `changeme` and prints a startup warning. **Set it for
anything past a direct cable.**

### Interactive docs (in-browser)

FastAPI serves an auto-generated API explorer — open it once the server is up:

| URL | What |
|---|---|
| `http://<host>:8000/docs` | **Swagger UI** — every endpoint with "try it out" forms |
| `http://<host>:8000/openapi.json` | raw **OpenAPI** schema (Postman / codegen) |

In **multi-arm mode** these are per-arm: the root `/docs` carries only `/health` + `/robots`
(the parent app's whole schema), so use `/ur15/docs` · `/gofa/docs` and `/ur15/openapi.json`.
The root docs page links to them.

Swagger UI is served from the vendored `web/vendor/swagger-ui/`, **not a CDN** — a machine
cabled straight to a robot has no DNS, and FastAPI's stock docs page renders blank there.
**ReDoc is not served** (`/redoc` → 404); it is CDN-only, so it was dropped rather than left
as a second blank page.

Auth is a per-request header field (no global "Authorize" button — it's a plain
header, not a declared security scheme): put `Bearer <token>` in `authorization`,
and your lease in `X-Lease` for writes. The `WS /telemetry` stream is **not** listed
there — WebSockets aren't part of OpenAPI.

---

## 2. Authentication

Every endpoint (including `/health` and the telemetry WS) requires the token.

- **HTTP:** `Authorization: Bearer <token>` header.
- **WebSocket:** `?token=<token>` query parameter (browsers can't set WS headers).

A bad or missing token returns **`401`** with a `WWW-Authenticate: Bearer`
challenge header. The token is compared in constant time.

```bash
curl -s localhost:8000/state -H "Authorization: Bearer $ROBOT_API_TOKEN"
```

---

## 3. Core concepts

### Single-writer lease

Reads and the safety path are open to any authenticated client. **All writes**
(`/move/*`, `/play`, `/gripper`, `/freedrive`, and saving/deleting trajectories)
require a **lease** — only one client holds it at a time, so two operators can't
fight over the arm. (Listing and loading trajectories are reads — auth only.)

1. `POST /control/acquire` → `{"lease_token": "…"}`. Returns **`409`** if already held.
2. Send the token as the **`X-Lease`** header on every write.
3. `POST /control/release` when done.

Acquire with `{"force": true}` **steals** a held lease (stopping the current
motion first) and invalidates the previous token. A write without a valid
`X-Lease` returns **`423 Locked`**. Releasing or losing the lease (force-steal,
deadman) also ends free-drive — a compliant arm never outlives its lease.

### Async commands

Writes are **non-blocking**: the server validates the request, submits it to the
controller, and immediately returns **`202`** with a `command_id`. One motion runs
at a time — submitting another while one is active returns **`409`** (`Busy`).

Track completion two ways:
- Poll `GET /command/{id}` until `status != "running"`.
- Watch the telemetry WS (`active_command` carries the same object).

### Telemetry heartbeat + deadman watchdog

An open telemetry WebSocket whose `?lease=` matches the current lease is a
**heartbeat**. A background watchdog stops the arm and releases the lease if the
lease holder goes silent (no heartbeat, no writes) for `watchdog_timeout_s`
(**default 2.0 s**) **while the arm is live — a motion running or free-drive
engaged**. This is a deadman: if your client crashes or the network drops mid-move
(or while the arm is compliant), the arm stops / releases on its own.

> Hold the telemetry WS open (with your `lease`) for the duration of any motion —
> at 20 Hz it refreshes the heartbeat automatically. Discrete writes also refresh
> it, but the gap between them can exceed the timeout.

### Safety path

`POST /stop` and `POST /estop` need **only authentication — no lease**. They are
always reachable, even by a client that doesn't hold the lease, so anyone can halt
the arm. `/stop` is a graceful decelerated stop; `/estop` is a hard stop.

---

## 4. Endpoint reference

| Method | Path | Lease? | Success | Purpose |
|---|---|---|---|---|
| `GET`  | `/health` | — | `200` | Liveness + robot name |
| `GET`  | `/state` | — | `200` | Full `RobotState` snapshot |
| `POST` | `/control/acquire` | — | `200` | Take the write lease (`{force}` to steal) |
| `POST` | `/control/release` | ✔ | `200` | Release the lease |
| `POST` | `/move/joints` | ✔ | `202` | Move to a joint configuration (MoveJ) |
| `POST` | `/move/pose` | ✔ | `202` | Move to a Cartesian pose (MoveL) |
| `POST` | `/play` | ✔ | `202` | Play a saved/inline trajectory |
| `POST` | `/gripper` | ✔ | `202` | Set gripper opening (UR only) |
| `POST` | `/freedrive` | ✔ | `200` | Engage/release hand-guiding (`{on}`) |
| `GET`  | `/trajectories` | — | `200` | List saved trajectory names for the arm |
| `GET`  | `/trajectories/{name}` | — | `200` | Load one trajectory's JSON |
| `POST` | `/trajectories` | ✔ | `200` | Save `{name, waypoints}` |
| `DELETE` | `/trajectories/{name}` | ✔ | `200` | Delete a saved trajectory |
| `GET`  | `/command/{id}` | — | `200` | Status of a submitted command |
| `POST` | `/stop` | — | `200` | Graceful stop |
| `POST` | `/estop` | — | `200` | Hard stop |
| `GET`  | `/camera/info` | — | `200` | Camera availability + geometry for the arm |
| `GET`  | `/camera/frame` | — | `200` | Latest JPEG, timestamps in `X-Frame-*` headers |
| `GET`  | `/camera/frame.json` | — | `200` | Same frame as base64 + timestamps |
| `GET`  | `/camera/stream` | — | `200` | MJPEG `multipart/x-mixed-replace` live feed |
| `WS`   | `/telemetry` | optional | — | Stream `RobotState`; heartbeat if lease-matched |

### GET /health
```bash
curl -s localhost:8000/health -H "Authorization: Bearer $TOK"
# {"ok": true, "robot": "ur15"}
```

### GET /state
Returns the latest [`RobotState`](#6-data-shapes) (see §6).
```bash
curl -s localhost:8000/state -H "Authorization: Bearer $TOK"
```
Add `?flat=1` for the [flat shape](#flat-shape--flat1) — the same data with every value a
scalar (`q_0`…`q_5`, `pose_pos_x`, `command_status`, …), for clients that can't walk nested JSON.
```bash
curl -s "localhost:8000/state?flat=1" -H "Authorization: Bearer $TOK"
```

### POST /control/acquire
Body (optional): `{"force": false}`.
```bash
curl -s -X POST localhost:8000/control/acquire -H "Authorization: Bearer $TOK"
# {"lease_token": "9f3c…"}            # or 409 if already held
# steal it:
curl -s -X POST localhost:8000/control/acquire -H "Authorization: Bearer $TOK" \
     -H 'Content-Type: application/json' -d '{"force": true}'
```

### POST /control/release
Requires `X-Lease`.
```bash
curl -s -X POST localhost:8000/control/release \
     -H "Authorization: Bearer $TOK" -H "X-Lease: $LEASE"
# {"released": true}                  # or 423 with a wrong/absent lease
```

### POST /move/joints
Body: `{"q": [6 floats, rad], "speed": 0<…≤1.0}` (`speed` default `1.0`).
Interpolates **in joint space** (MoveJ) — the exact configuration is reproduced,
no IK. `q` is shape- and finiteness-checked before it can reach the servo loop.
```bash
curl -s -X POST localhost:8000/move/joints \
     -H "Authorization: Bearer $TOK" -H "X-Lease: $LEASE" \
     -H 'Content-Type: application/json' \
     -d '{"q": [0.0, -1.4, 1.4, -1.4, -1.4, 0.2], "speed": 1.0}'
# 202 {"command_id": 1}
```

### POST /move/pose
Body: `{"pos": [x,y,z], "wxyz": [w,x,y,z], "speed": 0<…≤1.0}`.
Pose is the **grasp/EE frame** (same frame as `/state`'s `pose`). IK solves the
goal, then the **tool tip travels a straight line** (MoveL) to it.
```bash
curl -s -X POST localhost:8000/move/pose \
     -H "Authorization: Bearer $TOK" -H "X-Lease: $LEASE" \
     -H 'Content-Type: application/json' \
     -d '{"pos": [0.4, 0.1, 0.3], "wxyz": [0, 1, 0, 0], "speed": 0.5}'
```

### POST /play
Body: provide **one** of:
- `name` — a single saved trajectory (`trajectories/<robot>/<name>.json`, the arm being this endpoint's), or
- `names` — a **list** of saved names the **server chains**: it concatenates each trajectory's waypoints, in order, into one continuous motion (each seam becomes another segment), or
- `waypoints` — an inline list.

plus optional `speed`. Names are validated server-side (bad/missing → `400`, no path traversal).
```bash
curl -s -X POST localhost:8000/play \
     -H "Authorization: Bearer $TOK" -H "X-Lease: $LEASE" \
     -H 'Content-Type: application/json' \
     -d '{"name": "_sample_ur15", "speed": 1.0}'
# chain several into one continuous motion (server-side concatenation):
curl -s -X POST localhost:8000/play \
     -H "Authorization: Bearer $TOK" -H "X-Lease: $LEASE" \
     -H 'Content-Type: application/json' \
     -d '{"names": ["approach", "grasp", "retreat"], "speed": 0.5}'
```
Each waypoint is `{"q": [6]|null, "pos": [3], "wxyz": [4], "grip": float|null}`.
A waypoint with `q` replays those joints exactly; without it, IK solves from the
Cartesian pose. Segments between waypoints are straight-line (MoveL). Missing
both `name` and `waypoints` → `400`; an unknown name or malformed waypoint → `400`.

### POST /gripper
Body: `{"frac": 0.0..1.0}` (0 = open, 1 = closed). **UR only** — on the GoFa
(no gripper) this returns **`400`**.
```bash
curl -s -X POST localhost:8000/gripper \
     -H "Authorization: Bearer $TOK" -H "X-Lease: $LEASE" \
     -H 'Content-Type: application/json' -d '{"frac": 0.5}'
```

### POST /freedrive
Body: `{"on": true|false}`. Engages (`true`) or releases (`false`) hand-guiding
(UR `teachMode` / GoFa lead-through). Unlike the motion endpoints this is a
**synchronous mode toggle**, not an async command — it returns `200 {"freedrive":
bool}`, no `command_id`. While engaged the arm shows `activity: "freedrive"` and
**all motion endpoints return `409`** (mutually exclusive); engaging it `409`s if a
motion is already running. `/stop`, `/estop`, a force-steal, the deadman, and
`/control/release` all end it.
```bash
curl -s -X POST localhost:8000/freedrive \
     -H "Authorization: Bearer $TOK" -H "X-Lease: $LEASE" \
     -H 'Content-Type: application/json' -d '{"on": true}'      # {"freedrive": true}
```

### Trajectories  ·  GET/POST/DELETE /trajectories
Author and manage saved trajectories in `trajectories/<robot>/` (the arm is this
endpoint's). Names are validated (`[A-Za-z0-9_][A-Za-z0-9._-]{0,63}`, no path
separators) → bad name `422`. Capturing waypoints is done client-side off `/state`;
these endpoints only list/load/save/delete the files.

- `GET /trajectories` — auth only → `{"trajectories": ["_sample_ur15", …]}` (sorted).
- `GET /trajectories/{name}` — auth only → the stored `{robot, created, waypoints}`; unknown → `404`.
- `POST /trajectories` — **lease** → save (upsert) `{"name", "waypoints"}`; returns `{"saved": true, "name"}`. Each waypoint is validated (`pos[3]`, `wxyz[4]` finite; `q` `null`|`[6]`; `grip` `null`|number) → `422` on a bad shape or empty list.
- `DELETE /trajectories/{name}` — **lease** → `{"deleted": true, "name"}`; unknown → `404`.
```bash
curl -s localhost:8000/trajectories -H "Authorization: Bearer $TOK"
curl -s -X POST localhost:8000/trajectories \
     -H "Authorization: Bearer $TOK" -H "X-Lease: $LEASE" \
     -H 'Content-Type: application/json' \
     -d '{"name": "pickplace", "waypoints": [{"q":[0,-1,1,0,1,0.2],"pos":[0.4,0,0.3],"wxyz":[0,1,0,0],"grip":0.0}]}'
```

### GET /command/{id}
Returns the [command object](#command-object). Unknown or evicted id → `404`
(history is bounded at the 64 most-recent commands).
```bash
curl -s localhost:8000/command/1 -H "Authorization: Bearer $TOK"
# {"id": 1, "kind": "moving", "status": "done", "progress": 1.0, "error": null}
```

### POST /stop  ·  POST /estop
No lease required.
```bash
curl -s -X POST localhost:8000/stop  -H "Authorization: Bearer $TOK"   # {"stopped": true}
curl -s -X POST localhost:8000/estop -H "Authorization: Bearer $TOK"   # {"estopped": true}
```

⚠️ **`/estop` is a software stop, not the hardware e-stop.** On the UR15 it calls RTDE
`triggerProtectiveStop()`, so afterwards `safety_state` reads **`"PROTECTIVE_STOP"`** — *not*
`ROBOT_EMERGENCY_STOP`. Modes 6/7 come only from the physical e-stop circuit, which by design
cannot be asserted from software. Clear a protective stop from the pendant; RTDE can't. To
detect either kind programmatically, watch `activity == "stopped"` rather than matching one
`safety_state` string.

### Cameras  ·  GET /camera/info · /camera/frame · /camera/frame.json · /camera/stream

One USB camera per arm, so the endpoints are namespaced with everything else
(`/ur15/camera/frame`, `/gofa/camera/stream`). Reads — **authentication only, no lease**.

The token may be sent as a **`?token=` query param** as well as the `Authorization` header:
an `<img src=...>` can't set headers, and the telemetry WS already uses that spelling.

```bash
curl -s localhost:8000/camera/info -H "Authorization: Bearer $TOK"
# {"robot":"ur15","available":true,"index":0,"width":640,"height":480,"fps":15,
#  "seq":11,"ts":1215836.808,"ts_unix":1785538137.972}

curl -s -D- -o frame.jpg "localhost:8000/camera/frame?token=$TOK"
# X-Frame-Ts: 1215836.808263416        <- monotonic, SAME clock as RobotState.ts
# X-Frame-Ts-Unix: 1785538137.972138   <- wall clock
# X-Frame-Seq: 11
```

`/camera/frame.json` returns `{robot, ts, ts_unix, seq, jpeg_b64}` when you'd rather have one
JSON object than headers + body. `/camera/stream` is MJPEG for live viewing — drop it straight
into an `<img>`; browsers don't expose its per-part headers to JS, so use `frame`/`frame.json`
when you need to pair a frame with telemetry.

#### Pairing frames with telemetry

**This is the point of the two clocks.** `RobotState.ts` is `time.monotonic()` — an arbitrary
epoch — so a wall-clock-only frame timestamp would be *unpairable* with it. Every frame
therefore carries both, stamped together at the grab:

- **`ts`** — `time.monotonic()`, the **same clock** as `RobotState.ts`. The camera thread and
  the state poll live in one process, so `frame.ts - state.ts` is a real interval in seconds.
  This is what you join on.
- **`ts_unix`** — `time.time()`, wall clock, for logs and cross-machine correlation.

Because one frame carries both, it also **anchors** the monotonic clock: `ts_unix - ts` is the
offset that converts *any* `RobotState.ts` to absolute time. That's why `RobotState` needs no
new field.

```python
f  = httpx.get(f"{BASE}/camera/frame.json", headers=H).json()
st = httpx.get(f"{BASE}/state", headers=H).json()
skew_s   = abs(f["ts"] - st["ts"])            # same clock -> a real interval
state_utc = st["ts"] + (f["ts_unix"] - f["ts"])   # anchor monotonic to wall time
```

Expect a skew of up to one poll period between a frame and the nearest state (33 ms UR15,
100 ms GoFa) — they're sampled by independent threads.

#### Availability

Cameras are **best-effort**, like the Hand-E gripper: if the device index isn't configured,
cv2 is missing, or the camera won't deliver a frame, the server still starts and
`/camera/*` returns **`503`** with `/camera/info` reporting `{"available": false}`. Device
indices are machine-specific, so set them per host rather than editing code:

```bash
UR_CAMERA_INDEX=0 GOFA_CAMERA_INDEX=2 ROBOT_API_TOKEN=… uv run scripts/real.py api
```

`CAMERA_WIDTH`, `CAMERA_HEIGHT`, `CAMERA_FPS` and `CAMERA_JPEG_QUALITY` are the other knobs
(`robot_common.py`). `-1` disables an arm's camera.

---

## 5. Telemetry WebSocket

```
WS /telemetry?token=<token>&lease=<lease_token>&flat=1
```

Streams a JSON [`RobotState`](#6-data-shapes) at `telem_hz` (**default 20 Hz**).
`token` is required; `lease` is optional — supplying a lease that matches the
current holder turns the connection into the [heartbeat](#telemetry-heartbeat--deadman-watchdog).
`flat` is optional — `1`/`true`/`yes`/`on` streams the [flat shape](#flat-shape--flat1)
instead, exactly as `GET /state?flat=1` does. A bad token closes the socket with code `1008`.

```python
import json, websockets, asyncio
async def watch():
    url = f"ws://localhost:8000/telemetry?token={TOK}&lease={LEASE}"
    async with websockets.connect(url) as ws:
        while True:
            state = json.loads(await ws.recv())
            print(state["activity"], state["q"])
asyncio.run(watch())
```

---

## 6. Data shapes

> **Exhaustive field reference: [`docs/robot-state-reference.md`](docs/robot-state-reference.md)** —
> every possible value of `activity`, `safety_state`, `controller_state`, `health` and the
> command object, per arm, plus the `progress` and deadman gotchas. The summary below is the
> shape; that file is the contract.

### RobotState
Returned by `GET /state` and streamed over `/telemetry`.

```jsonc
{
  "ts": 1234.56,                       // monotonic timestamp of the snapshot
  "robot": "ur15",                     // "ur15" | "gofa"
  "q": [0.0, -1.4, 1.4, -1.4, -1.4, 0.2],   // 6 joint angles (rad)
  "pose": {                            // grasp/EE pose
    "pos":  [0.40, 0.10, 0.30],        //   metres
    "wxyz": [0.0, 1.0, 0.0, 0.0]       //   quaternion (w, x, y, z)
  },
  "gripper_frac": 0.0,                 // 0=open .. 1=closed; null if no gripper — always on
                                       //   the GoFa, and on the UR15 too if the Hand-E socket
                                       //   is unreachable, so null means "no gripper right now"
  "safety_state": "NORMAL",            // robot-reported safety state
  "controller_state": "ok",            // robot-reported controller/exec state
  "activity": "idle",                  // "idle" | "moving" | "playing" | "freedrive" | "stopped"
  "active_command": {                  // the RUNNING command (object below) — never null;
    "id": null,                        //   all-null whenever nothing is running, so idle
    "kind": null,                      //   looks the same as fresh boot. The terminal
    "status": null,                    //   status never appears here — GET /command/{id}
    "progress": null,                  //   is the only place it lives.
    "error": null
  },
  "conn_ok": true,                     // last hardware read succeeded
  "health": {}                         // transport-specific extras; fixed key set per arm —
                                       //   UR15 always {}, GoFa always
                                       //   {"egm_rx": <int|null>, "egm_tx": <int|null>}
}
```

#### The shape never changes

Every key above — and every sub-key of `active_command` and `health` — is present on **every
poll, from the first one**, for the life of the connection. No field starts empty and later
grows sub-keys, so a client can walk `state.active_command.status` or `state.health.egm_rx`
without guarding each level. When there's nothing to report the **value** is `null`; the key
stays. The only per-arm difference is which `health` keys exist (see the table above), and that
is fixed for a given arm at compile time, not discovered at runtime.

Two consequences worth pinning:

- **`active_command` is never `null`.** Test `active_command.id !== null` (or `status`), *not*
  the truthiness of the object itself — an always-present object is always truthy.
- **It clears the moment a command finishes, so no poll of `/state` ever sees the terminal
  status.** It goes `running` → all-null directly; `done` / `failed` / `stopped` never appear
  in it. Use `/state` for "is the arm busy right now", and **`GET /command/{id}` for how a
  command ended** — that is the only place the outcome and any `error` are retained. If you
  submit a command you care about the result of, poll its id; don't watch `/state` for it.

#### Flat shape — `?flat=1`

`GET /state?flat=1` and `WS /telemetry?…&flat=1` return the **same data with every value a
scalar** — no arrays, no sub-objects — for clients that can't walk nested JSON. Types are
unchanged (floats stay floats, `conn_ok` stays a bool); only the nesting is gone.

```jsonc
{
  "ts": 1209557.230674125,
  "robot": "ur15",
  "q_0": 0.0, "q_1": -1.5708, "q_2": 1.5708,   // …through q_5
  "q_3": -1.5708, "q_4": -1.5708, "q_5": 0.0,
  "pose_pos_x": 0.652499, "pose_pos_y": 0.182399, "pose_pos_z": 0.566200,
  "pose_wxyz_w": 0.0, "pose_wxyz_x": 0.707107,
  "pose_wxyz_y": -0.707107, "pose_wxyz_z": 0.0000026,
  "gripper_frac": 0.0,
  "safety_state": "NORMAL",
  "controller_state": "1",
  "activity": "idle",
  "command_id": null,                  // active_command flattened to command_*;
  "command_kind": null,                //   all null when nothing is running
  "command_status": null,
  "command_progress": null,
  "command_error": null,
  "conn_ok": true
                                       // + health_<key> per health entry — GoFa adds
                                       //   health_egm_rx / health_egm_tx
}
```

Mapping: `q` → `q_0`…`q_5`, `pose.pos` → `pose_pos_{x,y,z}`, `pose.wxyz` →
`pose_wxyz_{w,x,y,z}`, `active_command` → `command_{id,kind,status,progress,error}`,
`health` → `health_<key>`. Everything else keeps its name.

**The key set is stable**, exactly as in the nested shape. Every key above is present on every
frame for the life of the connection — an absent value is `null`, never a missing key. So
`gripper_frac` is present-and-`null` on the GoFa, and the five `command_*` keys are
present-and-`null` whenever nothing is running.

`health_*` mirrors `health`, so it is fixed **per arm** rather than shared across both: the
GoFa always emits `health_egm_rx`/`health_egm_tx` (`null` before an EGM session exists), and
the UR15 emits no `health_*` keys at all. Fixed columns per arm — just not the *same* columns.

Omitting `flat` (or `flat=0`) returns the nested shape above, unchanged — this is purely
additive, so existing clients are unaffected.

### Command object
Returned by `GET /command/{id}`; also embedded as `active_command` in `RobotState`.

```jsonc
{
  "id": 1,
  "kind": "moving",                    // "moving" (move/gripper) | "playing" (play)
  "status": "running",                 // "running" | "done" | "failed" | "stopped"
  "progress": 0.0,                     // fraction of segments completed, 0.0..1.0 (1.0 on done)
  "error": null                        // failure message string, or null
}
```

`status` meanings: `running` (in progress), `done` (completed), `stopped`
(pre-empted by `/stop`, `/estop`, a force-steal, or the watchdog), `failed`
(raised an error — see `error`).

---

## 7. Status codes & validation

| Code | When |
|---|---|
| `200` | Read / lease / safety call succeeded |
| `202` | Write accepted and submitted (`{command_id}` returned) |
| `400` | Bad command content: no `name`/`waypoints`, unknown trajectory, malformed waypoint, or `/gripper` on a gripper-less arm |
| `401` | Missing/invalid bearer token |
| `404` | `GET /command/{id}` for an unknown or evicted id |
| `409` | `acquire` while the lease is held, **or** a write while another motion is running (`Busy`) |
| `422` | Invalid vector or speed (see below) — rejected *before* reaching the servo loop |
| `423` | Write without a valid `X-Lease` |

**Vector / speed validation (`422`):**
- `q` must be a list of **6** finite numbers; `pos` a list of **3**; `wxyz` a list of **4**.
- `speed` must be a number in **`(0, 1.0]`**. This both caps it at `1.0` (so the API
  can never exceed the controller's `MAX_JOINT_SPEED`) and rejects `0`/negative
  values (which would wedge the motion profile). `speed` scales motion *below* the
  configured cap; it never raises it.

---

## 8. End-to-end example (Python / httpx)

```python
import time, httpx

BASE, TOK = "http://localhost:8000", "your-token"
auth = {"Authorization": f"Bearer {TOK}"}

with httpx.Client(base_url=BASE, timeout=10.0) as cl:
    assert cl.get("/state", headers=auth).json()["robot"] in ("ur15", "gofa")

    lease = cl.post("/control/acquire", headers=auth).json()["lease_token"]
    h = {**auth, "X-Lease": lease}

    cid = cl.post("/move/joints", headers=h,
                  json={"q": [0.0, -1.0, 1.0, 0.0, 1.0, 0.2], "speed": 1.0}
                  ).json()["command_id"]

    while cl.get(f"/command/{cid}", headers=auth).json()["status"] == "running":
        time.sleep(0.1)
    print("final:", cl.get(f"/command/{cid}", headers=auth).json())   # status: done

    cl.post("/control/release", headers=h)
```

> A real client should keep a `/telemetry?…&lease=…` WebSocket open for the
> duration so the deadman heartbeat is satisfied while the move runs.

---

## 9. Per-robot notes

| | UR15 | GoFa CRB 15000 |
|---|---|---|
| `gripper_frac` | float (Hand-E) | `null` — `/gripper` → `400` |
| Joints | 6 | 6 |
| TCP speed cap | none on this path | **0.25 m/s** (`MAX_TCP_SPEED`, collaborative) — `speed` only scales below it |

---

## 10. Testing offline

The whole API runs with no robot and no network against the sim fakes:

```bash
uv run scripts/api_smoketest.py    # in-process TestClient + real sim.py subprocess e2e
uv run scripts/sim.py api ur15     # serve the sim over real HTTP for manual poking
```

---

## 11. Security notes

- **Set `ROBOT_API_TOKEN`.** The `changeme` default is a dev convenience only.
- **Bind `--host 127.0.0.1`** unless you need off-box access; the API speaks plain
  HTTP (terminate TLS at a reverse proxy if it leaves the host).
- The lease prevents *concurrent* writers, not malicious ones — anyone with the
  token can `acquire`/`force` and can `stop`/`estop`. Treat the token as the
  trust boundary.
</content>
</invoke>
