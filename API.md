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
./robot_control/bin/python scripts/real.py api ur15          # or: gofa
# offline sim (no robot, no network)
./robot_control/bin/python scripts/sim.py  api ur15          # or: gofa

# options
./robot_control/bin/python scripts/real.py api gofa --host 127.0.0.1 --port 8000
```

| Flag | Default | Notes |
|---|---|---|
| `robot` (positional) | — | `ur15` or `gofa` |
| `--host` | `0.0.0.0` | All interfaces. Use `127.0.0.1` for loopback only. |
| `--port` | `8000` | TCP port. |

The server connects to the arm at startup and **aborts if it can't reach it**.
On a clean shutdown (Ctrl-C / SIGINT) it closes the controller (stopping any motion).

### Auth token

The bearer token is read from the **`ROBOT_API_TOKEN`** environment variable:

```bash
ROBOT_API_TOKEN='a-long-random-secret' ./robot_control/bin/python scripts/real.py api ur15
```

If unset it defaults to `changeme` and prints a startup warning. **Set it for
anything past a direct cable.**

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
(`/move/*`, `/play`, `/gripper`) require a **lease** — only one client holds it at
a time, so two operators can't fight over the arm.

1. `POST /control/acquire` → `{"lease_token": "…"}`. Returns **`409`** if already held.
2. Send the token as the **`X-Lease`** header on every write.
3. `POST /control/release` when done.

Acquire with `{"force": true}` **steals** a held lease (stopping the current
motion first) and invalidates the previous token. A write without a valid
`X-Lease` returns **`423 Locked`**.

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
(**default 2.0 s**) **while a motion is running**. This is a deadman: if your
client crashes or the network drops mid-move, the arm stops on its own.

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
| `GET`  | `/command/{id}` | — | `200` | Status of a submitted command |
| `POST` | `/stop` | — | `200` | Graceful stop |
| `POST` | `/estop` | — | `200` | Hard stop |
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
Body: provide **one** of `name` (a `trajectories/<name>.json`) or `waypoints`
(an inline list), plus optional `speed`.
```bash
curl -s -X POST localhost:8000/play \
     -H "Authorization: Bearer $TOK" -H "X-Lease: $LEASE" \
     -H 'Content-Type: application/json' \
     -d '{"name": "_sample_ur15", "speed": 1.0}'
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

---

## 5. Telemetry WebSocket

```
WS /telemetry?token=<token>&lease=<lease_token>
```

Streams a JSON [`RobotState`](#6-data-shapes) at `telem_hz` (**default 20 Hz**).
`token` is required; `lease` is optional — supplying a lease that matches the
current holder turns the connection into the [heartbeat](#telemetry-heartbeat--deadman-watchdog).
A bad token closes the socket with code `1008`.

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
  "gripper_frac": 0.0,                 // 0=open .. 1=closed; null if no gripper (GoFa)
  "safety_state": "NORMAL",            // robot-reported safety state
  "controller_state": "ok",            // robot-reported controller/exec state
  "activity": "idle",                  // "idle" | "moving" | "playing" | "stopped"
  "active_command": null,              // command object (below) or null
  "conn_ok": true,                     // last hardware read succeeded
  "health": {}                         // transport-specific extras
}
```

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
./robot_control/bin/python scripts/api_smoketest.py    # in-process TestClient + real sim.py subprocess e2e
./robot_control/bin/python scripts/sim.py api ur15     # serve the sim over real HTTP for manual poking
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
