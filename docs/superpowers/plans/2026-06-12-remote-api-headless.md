# Remote control API (headless) over RobotController — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A bearer-token-authed HTTP + WebSocket remote API over the existing `RobotController` — monitor state, issue high-level motion commands (async, returns a command id), stream telemetry, with a single-writer control lease and a heartbeat watchdog — all runnable and testable offline under `sim.py api <robot>`.

**Architecture:** `lib/robot_api.py` builds a FastAPI app over a connected `RobotController`. Commands are async: POST validates + `controller.<cmd>()` → `202 {command_id}`; status is polled at `GET /command/{id}` or streamed over `WS /telemetry`. A single write lease gates mutating endpoints; the lease holder must keep its telemetry WS (the heartbeat) alive or — during an active motion — a watchdog stops the arm and releases the lease. `scripts/api_server.py` is the `api` dispatcher target; it builds the controller and runs uvicorn. Because the controller sits on the sim-shadowed hardware clients, `sim.py api ur15` runs the whole API offline.

**Tech Stack:** Python 3.13; FastAPI + uvicorn (to install); `httpx`/`websockets` (already installed) for the tests. No pytest — tests are stdlib-`assert` scripts using FastAPI's in-process `TestClient` plus one subprocess e2e.

**Spec:** `docs/superpowers/specs/2026-06-09-remote-control-api-design.md` (this plan implements the headless API; the embedded viser viewer + the controller Live primitive are a separate follow-on plan).

---

## Background the implementer needs

- **The controller** (`lib/control/`, already built): `make_controller("ur15"|"gofa") -> RobotController` with `.connect()`/`.close()`, `get_state() -> RobotState` (`.to_dict()` is JSON-ready), async commands `move_to_joints(q, speed)` / `move_to_pose(pos, wxyz, speed)` / `play(name_or_waypoints, speed)` / `set_gripper(frac)` each returning an **int command id** (raises `control.Busy` if a motion is already running), `stop()` / `estop()` (preempt), `wait(cid, timeout)` (→ status string), `command_status(cid)` (→ dict or None). `RobotState.gripper_frac` is `None` for robots without a gripper (GoFa).
- **Offline test pattern:** `robot_sim.install(robot)` BEFORE `make_controller(robot)` makes the controller use the fake hardware (perfect tracking). Both `_sample_ur15.json` and `_sample_gofa.json` trajectories exist and carry per-waypoint `q`.
- **Imports:** `lib/` is on `sys.path`. `lib/robot_api.py` does `from control import ...` (the package) and `import robot_common as rc`. Scripts bootstrap repo-root + `lib/`.
- **FastAPI TestClient** (`from fastapi.testclient import TestClient`) runs the app in-process over the real controller (no uvicorn/socket needed) — used for unit-level endpoint tests. It supports `client.websocket_connect(...)`.

## File structure

| File | Responsibility |
|---|---|
| `lib/control/base.py` (modify) | bounded command-history retention so `command_status`/`wait` resolve finished commands |
| `lib/robot_api.py` (create) | `build_app(controller, token, telem_hz, watchdog_timeout_s) -> FastAPI` — all endpoints + WS + watchdog |
| `scripts/api_server.py` (create) | `api` entry: parse robot from argv, build controller, `build_app`, run uvicorn |
| `lib/dispatch.py` (modify) | add `"api": "api_server.py"` to `TARGETS` |
| `scripts/api_smoketest.py` (create) | stdlib-assert: in-process `TestClient` tests + one subprocess e2e against `sim.py api ur15` |
| `README.md`, `CLAUDE.md` (modify) | document the API + the new deps |

---

## Task 1: Controller command-history retention

So a client can `GET /command/{id}` (or `wait`) after a command finishes and a newer one starts. Today the controller keeps only the most-recent command (`command_status` returns None / `wait` returns "gone" for an aged-out id).

**Files:** Modify `lib/control/base.py`; modify `scripts/control_smoketest.py`.

- [ ] **Step 1: Write the failing test**

In `scripts/control_smoketest.py`, add above `main()`:

```python
def test_command_history():
    robot_sim.install("ur15")
    from control import make_controller
    c = make_controller("ur15")
    c.connect()
    try:
        cid1 = c.move_to_joints([0.0, -1.4, 1.4, -1.4, -1.4, 0.1], speed=5.0)
        assert c.wait(cid1, timeout=20.0) == "done"
        cid2 = c.move_to_joints(robot_sim.UR_HOME, speed=5.0)
        assert c.wait(cid2, timeout=20.0) == "done"
        # cid1 finished and was superseded by cid2 — its result must still be queryable
        st1 = c.command_status(cid1)
        assert st1 is not None and st1["status"] == "done", "command history not retained"
        assert c.wait(cid1, timeout=1.0) == "done", "wait() should resolve a retained finished command"
    finally:
        c.close()
    print("PASS test_command_history")
```

Wire it into `main()` after `test_controller_freedrive_grasp()`.

- [ ] **Step 2: Run it, verify it FAILS**

Run: `./robot_control/bin/python scripts/control_smoketest.py`
Expected: fails at `test_command_history` — `command_status(cid1)` returns None (cid1 superseded, no history).

- [ ] **Step 3: Implement**

In `lib/control/base.py`:

(a) add `import collections` at the top (with the other stdlib imports `copy`, `itertools`, `threading`, `time`).

(b) add a class constant on `RobotController` (next to `POLL_HZ`):

```python
    _CMD_HISTORY_MAX: int = 64
```

(c) in `__init__`, after `self._active: dict | None = None`, add:

```python
        self._cmd_history: "collections.OrderedDict[int, dict]" = collections.OrderedDict()
```

(d) in `_run_cmd`, the terminal-status block currently reads:

```python
        with self._cmd_lock:
            if self._active is not None and self._active["id"] == cid:
                self._active["status"] = status
                self._active["error"] = err
                if status == "done":
                    self._active["progress"] = 1.0
```

Replace it with (retain a copy in the bounded history):

```python
        with self._cmd_lock:
            if self._active is not None and self._active["id"] == cid:
                self._active["status"] = status
                self._active["error"] = err
                if status == "done":
                    self._active["progress"] = 1.0
                self._cmd_history[cid] = dict(self._active)
                while len(self._cmd_history) > self._CMD_HISTORY_MAX:
                    self._cmd_history.popitem(last=False)
```

(e) update `command_status` to fall back to history:

```python
    def command_status(self, cid: int) -> dict | None:
        with self._cmd_lock:
            if self._active is not None and self._active["id"] == cid:
                return dict(self._active)
            if cid in self._cmd_history:
                return dict(self._cmd_history[cid])
        return None
```

(f) update `wait` to resolve via history too. Replace the loop body's lock block:

```python
            with self._cmd_lock:
                a = self._active
                if a is not None and a["id"] == cid and a["status"] != "running":
                    return a["status"]
                if a is None or a["id"] > cid:
                    return "gone"
```

with:

```python
            with self._cmd_lock:
                a = self._active
                if a is not None and a["id"] == cid and a["status"] != "running":
                    return a["status"]
                if cid in self._cmd_history:
                    return self._cmd_history[cid]["status"]
                if a is None or a["id"] > cid:
                    return "gone"
```

- [ ] **Step 4: Run it, verify it PASSES**

Run: `./robot_control/bin/python scripts/control_smoketest.py`
Expected: all PASS lines incl. `PASS test_command_history`, then `ALL CONTROL SMOKE TESTS PASSED`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add lib/control/base.py scripts/control_smoketest.py
git commit -m "feat(control): retain a bounded command history for status/wait after completion"
```

---

## Task 2: Install FastAPI + uvicorn

**Files:** none in git (venv only); modify `README.md`.

- [ ] **Step 1: Install the deps**

Run: `./robot_control/bin/pip install fastapi "uvicorn[standard]"`
Expected: installs `fastapi`, `uvicorn`, `starlette`, `pydantic` (+ deps). `httpx`/`websockets` already present.

- [ ] **Step 2: Verify imports**

Run: `./robot_control/bin/python -c "import fastapi, uvicorn; from fastapi.testclient import TestClient; print('fastapi', fastapi.__version__)"`
Expected: prints `fastapi <version>`, no error.

- [ ] **Step 3: Document the dep in README setup**

In `README.md`, find the `## Setup (rebuild the venv)` pip install block (the `./robot_control/bin/pip install numpy viser ...` line). Append `fastapi` and `uvicorn[standard]` to that install list. Read the file to get the exact current line, then add the two packages to it (e.g. after `requests urllib3` add `fastapi "uvicorn[standard]"`).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "build: add fastapi + uvicorn for the remote API"
```

---

## Task 3: `robot_api.py` — `/state`, `/health`, bearer auth + TestClient scaffold

**Files:** Create `lib/robot_api.py`; create `scripts/api_smoketest.py`.

- [ ] **Step 1: Write the failing test**

Create `scripts/api_smoketest.py`:

```python
"""Offline smoke test for the remote API (lib/robot_api), in-process via FastAPI's
TestClient over a controller wired to the sim fakes. No robot, no network.

  ./robot_control/bin/python scripts/api_smoketest.py

Exits 0 on success, 1 on the first failed assertion.
"""
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import robot_sim  # noqa: E402

TOKEN = "test-token"


def _client(robot="ur15"):
    """Build a TestClient over a sim-backed controller. Returns (client, controller)."""
    robot_sim.install(robot)
    from control import make_controller
    from fastapi.testclient import TestClient
    from robot_api import build_app
    c = make_controller(robot)
    c.connect()
    app = build_app(c, token=TOKEN, telem_hz=50.0, watchdog_timeout_s=0.5)
    return TestClient(app), c


def _auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def test_state_and_auth():
    client, c = _client("ur15")
    try:
        # missing/bad token -> 401
        assert client.get("/state").status_code == 401
        assert client.get("/state", headers={"Authorization": "Bearer nope"}).status_code == 401
        # health + state with token
        h = client.get("/health", headers=_auth())
        assert h.status_code == 200 and h.json()["robot"] == "ur15"
        s = client.get("/state", headers=_auth())
        assert s.status_code == 200
        body = s.json()
        assert body["robot"] == "ur15" and len(body["q"]) == 6 and "pose" in body
    finally:
        c.close()
    print("PASS test_state_and_auth")


def main():
    test_state_and_auth()
    print("ALL API SMOKE TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
```

- [ ] **Step 2: Run it, verify it FAILS**

Run: `./robot_control/bin/python scripts/api_smoketest.py`
Expected: `ModuleNotFoundError: No module named 'robot_api'`.

- [ ] **Step 3: Implement `lib/robot_api.py` (state/health/auth)**

Create `lib/robot_api.py`:

```python
"""FastAPI remote-control API over a RobotController.

build_app(controller, token, ...) returns a FastAPI app exposing read/state +
async high-level commands + a telemetry WebSocket, gated by a bearer token and a
single write lease (see the design spec). The controller is the single hardware
owner; this module only adapts it to HTTP/WS.
"""
from __future__ import annotations

import asyncio
import secrets
import time

from fastapi import Body, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect

from control import Busy


def build_app(controller, token: str, telem_hz: float = 20.0,
              watchdog_timeout_s: float = 2.0) -> FastAPI:
    app = FastAPI(title="robot-control-api")
    # single write lease: {"token": str|None, "last_seen": monotonic float}
    lease = {"token": None, "last_seen": 0.0}

    def check_auth(authorization: str | None) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="bad or missing token")

    @app.get("/health")
    def health(authorization: str = Header(None)):
        check_auth(authorization)
        return {"ok": True, "robot": controller.robot_name}

    @app.get("/state")
    def state(authorization: str = Header(None)):
        check_auth(authorization)
        return controller.get_state().to_dict()

    return app
```

- [ ] **Step 4: Run it, verify it PASSES**

Run: `./robot_control/bin/python scripts/api_smoketest.py`
Expected: `PASS test_state_and_auth`, `ALL API SMOKE TESTS PASSED`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add lib/robot_api.py scripts/api_smoketest.py
git commit -m "feat(api): FastAPI app with /state /health + bearer auth"
```

---

## Task 4: `robot_api.py` — control lease

**Files:** Modify `lib/robot_api.py`; modify `scripts/api_smoketest.py`.

- [ ] **Step 1: Write the failing test**

In `scripts/api_smoketest.py`, add above `main()`:

```python
def test_lease():
    client, c = _client("ur15")
    try:
        r = client.post("/control/acquire", headers=_auth())
        assert r.status_code == 200, r.text
        lease_token = r.json()["lease_token"]
        assert lease_token
        # second acquire without force -> 409
        assert client.post("/control/acquire", headers=_auth()).status_code == 409
        # force acquire -> new token
        r2 = client.post("/control/acquire", headers=_auth(), json={"force": True})
        assert r2.status_code == 200 and r2.json()["lease_token"] != lease_token
        # release with the (new) lease
        rel = client.post("/control/release", headers={**_auth(), "X-Lease": r2.json()["lease_token"]})
        assert rel.status_code == 200
        # after release, acquire works again
        assert client.post("/control/acquire", headers=_auth()).status_code == 200
    finally:
        c.close()
    print("PASS test_lease")
```

Wire into `main()` after `test_state_and_auth()`.

- [ ] **Step 2: Run it, verify it FAILS**

Run: `./robot_control/bin/python scripts/api_smoketest.py`
Expected: fails at `test_lease` — `/control/acquire` is 404 (not implemented).

- [ ] **Step 3: Implement**

In `lib/robot_api.py`, add a lease-check helper and the two endpoints BEFORE `return app`:

```python
    def check_lease(x_lease: str | None) -> None:
        if lease["token"] is None or x_lease != lease["token"]:
            raise HTTPException(status_code=423, detail="no or invalid control lease")
        lease["last_seen"] = time.monotonic()

    @app.post("/control/acquire")
    def acquire(authorization: str = Header(None), force: bool = Body(False, embed=True)):
        check_auth(authorization)
        if lease["token"] is not None and not force:
            raise HTTPException(status_code=409, detail="control lease already held")
        if lease["token"] is not None and force:
            controller.stop()   # steal: stop whatever the old holder was doing
        lease["token"] = secrets.token_hex(8)
        lease["last_seen"] = time.monotonic()
        return {"lease_token": lease["token"]}

    @app.post("/control/release")
    def release(authorization: str = Header(None), x_lease: str = Header(None)):
        check_auth(authorization)
        check_lease(x_lease)
        lease["token"] = None
        return {"released": True}
```

(`check_lease` is also used by the command endpoints in Task 5. Place it where the inner functions can close over `lease`/`check_auth`.)

- [ ] **Step 4: Run it, verify it PASSES**

Run: `./robot_control/bin/python scripts/api_smoketest.py`
Expected: `PASS test_lease` added, `ALL API SMOKE TESTS PASSED`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add lib/robot_api.py scripts/api_smoketest.py
git commit -m "feat(api): single-writer control lease (acquire/release/force)"
```

---

## Task 5: `robot_api.py` — command endpoints + `/command/{id}` + stop/estop

**Files:** Modify `lib/robot_api.py`; modify `scripts/api_smoketest.py`.

- [ ] **Step 1: Write the failing test**

In `scripts/api_smoketest.py`, add above `main()`:

```python
def _poll_command(client, cid, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/command/{cid}", headers=_auth())
        assert r.status_code == 200, r.text
        st = r.json()
        if st["status"] != "running":
            return st["status"]
        time.sleep(0.05)
    return "timeout"


def test_commands():
    client, c = _client("ur15")
    try:
        lease = client.post("/control/acquire", headers=_auth()).json()["lease_token"]
        h = {**_auth(), "X-Lease": lease}
        # move without lease header -> 423
        assert client.post("/move/joints", headers=_auth(),
                           json={"q": robot_sim.UR_HOME, "speed": 5.0}).status_code == 423
        # move with lease -> 202 + command_id, completes, state reaches target
        target = [0.0, -1.4, 1.4, -1.4, -1.4, 0.2]
        r = client.post("/move/joints", headers=h, json={"q": target, "speed": 5.0})
        assert r.status_code == 202, r.text
        cid = r.json()["command_id"]
        assert _poll_command(client, cid) == "done"
        st = client.get("/state", headers=_auth()).json()
        assert max(abs(a - b) for a, b in zip(st["q"], target)) < 1e-6
        # play by name -> done
        rp = client.post("/play", headers=h, json={"name": "_sample_ur15", "speed": 5.0})
        assert rp.status_code == 202
        assert _poll_command(client, rp.json()["command_id"]) == "done"
        # gripper -> done
        rg = client.post("/gripper", headers=h, json={"frac": 0.5})
        assert rg.status_code == 202
        assert _poll_command(client, rg.json()["command_id"]) == "done"
        # stop (no lease needed) returns 200
        assert client.post("/stop", headers=_auth()).status_code == 200
        # unknown command id -> 404
        assert client.get("/command/999999", headers=_auth()).status_code == 404
    finally:
        c.close()
    print("PASS test_commands")


def test_gofa_no_gripper():
    client, c = _client("gofa")
    try:
        lease = client.post("/control/acquire", headers=_auth()).json()["lease_token"]
        h = {**_auth(), "X-Lease": lease}
        # GoFa has no gripper -> 400
        assert client.post("/gripper", headers=h, json={"frac": 0.5}).status_code == 400
        # but a move works
        r = client.post("/move/joints", headers=h,
                        json={"q": [0.0, 0.1, 0.0, 0.0, 1.5708, 0.0], "speed": 5.0})
        assert r.status_code == 202
        assert _poll_command(client, r.json()["command_id"]) == "done"
    finally:
        c.close()
    print("PASS test_gofa_no_gripper")
```

Wire both into `main()` after `test_lease()`.

- [ ] **Step 2: Run it, verify it FAILS**

Run: `./robot_control/bin/python scripts/api_smoketest.py`
Expected: fails at `test_commands` — `/move/joints` is 404.

- [ ] **Step 3: Implement**

In `lib/robot_api.py`, add before `return app`:

```python
    def _submit(fn):
        try:
            return {"command_id": fn()}
        except Busy as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.post("/move/joints", status_code=202)
    def move_joints(authorization: str = Header(None), x_lease: str = Header(None),
                    q: list = Body(...), speed: float = Body(1.0)):
        check_auth(authorization)
        check_lease(x_lease)
        return _submit(lambda: controller.move_to_joints(q, speed))

    @app.post("/move/pose", status_code=202)
    def move_pose(authorization: str = Header(None), x_lease: str = Header(None),
                  pos: list = Body(...), wxyz: list = Body(...), speed: float = Body(1.0)):
        check_auth(authorization)
        check_lease(x_lease)
        return _submit(lambda: controller.move_to_pose(pos, wxyz, speed))

    @app.post("/play", status_code=202)
    def play(authorization: str = Header(None), x_lease: str = Header(None),
             name: str = Body(None), waypoints: list = Body(None), speed: float = Body(1.0)):
        check_auth(authorization)
        check_lease(x_lease)
        target = name if name is not None else waypoints
        if target is None:
            raise HTTPException(status_code=400, detail="provide 'name' or 'waypoints'")
        return _submit(lambda: controller.play(target, speed))

    @app.post("/gripper", status_code=202)
    def gripper(authorization: str = Header(None), x_lease: str = Header(None),
                frac: float = Body(..., embed=True)):
        check_auth(authorization)
        check_lease(x_lease)
        if controller.get_state().gripper_frac is None:
            raise HTTPException(status_code=400, detail="this robot has no gripper")
        return _submit(lambda: controller.set_gripper(frac))

    @app.get("/command/{cid}")
    def command(cid: int, authorization: str = Header(None)):
        check_auth(authorization)
        st = controller.command_status(cid)
        if st is None:
            raise HTTPException(status_code=404, detail="unknown command id")
        return st

    @app.post("/stop")
    def stop(authorization: str = Header(None)):
        check_auth(authorization)
        controller.stop()
        return {"stopped": True}

    @app.post("/estop")
    def estop(authorization: str = Header(None)):
        check_auth(authorization)
        controller.estop()
        return {"estopped": True}
```

- [ ] **Step 4: Run it, verify it PASSES**

Run: `./robot_control/bin/python scripts/api_smoketest.py`
Expected: `PASS test_commands` and `PASS test_gofa_no_gripper`, `ALL API SMOKE TESTS PASSED`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add lib/robot_api.py scripts/api_smoketest.py
git commit -m "feat(api): async command endpoints + /command/{id} + stop/estop"
```

---

## Task 6: `robot_api.py` — telemetry WebSocket + heartbeat watchdog

**Files:** Modify `lib/robot_api.py`; modify `scripts/api_smoketest.py`.

- [ ] **Step 1: Write the failing test**

In `scripts/api_smoketest.py`, add above `main()`:

```python
def test_telemetry_ws():
    client, c = _client("ur15")
    try:
        with client.websocket_connect(f"/telemetry?token={TOKEN}") as ws:
            msg = ws.receive_json()
            assert msg["robot"] == "ur15" and len(msg["q"]) == 6
    finally:
        c.close()
    print("PASS test_telemetry_ws")


def test_telemetry_auth():
    client, c = _client("ur15")
    try:
        # bad token on the WS should be rejected (connect closes before a message)
        try:
            with client.websocket_connect("/telemetry?token=nope") as ws:
                ws.receive_json()
            raised = False
        except Exception:
            raised = True
        assert raised, "WS with bad token should be rejected"
    finally:
        c.close()
    print("PASS test_telemetry_auth")


def test_watchdog():
    # watchdog_timeout_s=0.5 from _client(); a slow move stays running long enough
    client, c = _client("ur15")
    try:
        lease = client.post("/control/acquire", headers=_auth()).json()["lease_token"]
        h = {**_auth(), "X-Lease": lease}
        # start a slow, long move (small speed => seconds of motion)
        far = [v + 0.8 for v in robot_sim.UR_HOME]
        r = client.post("/move/joints", headers=h, json={"q": far, "speed": 0.05})
        assert r.status_code == 202
        cid = r.json()["command_id"]
        # do NOT open the telemetry WS and send no further commands: the lease goes
        # stale and the watchdog must stop the active motion within ~timeout.
        deadline = time.monotonic() + 5.0
        status = "running"
        while time.monotonic() < deadline:
            status = client.get(f"/command/{cid}", headers=_auth()).json()["status"]
            if status != "running":
                break
            time.sleep(0.1)
        assert status == "stopped", f"watchdog should have stopped the motion, got {status}"
        # lease was auto-released
        assert client.post("/control/acquire", headers=_auth()).status_code == 200
    finally:
        c.close()
    print("PASS test_watchdog")
```

Wire all three into `main()` after `test_gofa_no_gripper()`.

- [ ] **Step 2: Run it, verify it FAILS**

Run: `./robot_control/bin/python scripts/api_smoketest.py`
Expected: fails at `test_telemetry_ws` — no `/telemetry` route (connect fails).

- [ ] **Step 3: Implement**

In `lib/robot_api.py`, add the WS endpoint and the watchdog before `return app`:

```python
    @app.websocket("/telemetry")
    async def telemetry(ws: WebSocket):
        if token and ws.query_params.get("token") != token:
            await ws.close(code=1008)
            return
        await ws.accept()
        ws_lease = ws.query_params.get("lease")
        try:
            while True:
                # an open WS from the lease holder is the heartbeat
                if ws_lease and ws_lease == lease["token"]:
                    lease["last_seen"] = time.monotonic()
                await ws.send_json(controller.get_state().to_dict())
                await asyncio.sleep(1.0 / telem_hz)
        except WebSocketDisconnect:
            pass

    @app.on_event("startup")
    async def _start_watchdog():
        async def _loop():
            while True:
                await asyncio.sleep(0.1)
                if lease["token"] is None:
                    continue
                st = controller.get_state()
                ac = st.active_command
                active = ac is not None and ac["status"] == "running"
                if active and (time.monotonic() - lease["last_seen"]) > watchdog_timeout_s:
                    controller.stop()          # deadman: stop the arm
                    lease["token"] = None       # release the lease
        asyncio.create_task(_loop())
```

- [ ] **Step 4: Run it, verify it PASSES**

Run: `./robot_control/bin/python scripts/api_smoketest.py`
Expected: `PASS test_telemetry_ws`, `PASS test_telemetry_auth`, `PASS test_watchdog`, `ALL API SMOKE TESTS PASSED`, exit 0. (The watchdog test takes ~1 s.)

- [ ] **Step 5: Commit**

```bash
git add lib/robot_api.py scripts/api_smoketest.py
git commit -m "feat(api): telemetry WebSocket + heartbeat watchdog (deadman)"
```

---

## Task 7: `scripts/api_server.py` + the `api` dispatcher target

**Files:** Create `scripts/api_server.py`; modify `lib/dispatch.py`.

- [ ] **Step 1: Add the `api` target to the dispatcher**

In `lib/dispatch.py`, add to the `TARGETS` dict (after `"teleop"`):

```python
    "api": "api_server.py",
```

- [ ] **Step 2: Create `scripts/api_server.py`**

```python
#!/usr/bin/env python
"""api_server.py <ur15|gofa> [--host H] [--port P] — serve the remote control API.

Builds a RobotController for the target robot and serves lib/robot_api over uvicorn.
The bearer token is read from ROBOT_API_TOKEN (default "changeme"). Launched via the
dispatcher: `real.py api ur15` (hardware) or `sim.py api ur15` (offline).
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import uvicorn  # noqa: E402

from control import make_controller  # noqa: E402
from robot_api import build_app  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Remote control API server (UR15 / GoFa).")
    ap.add_argument("robot", choices=["ur15", "gofa"], help="which arm to serve")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    token = os.environ.get("ROBOT_API_TOKEN", "changeme")
    if token == "changeme":
        print("WARNING: ROBOT_API_TOKEN not set — using 'changeme'. Set it before real use.")

    print(f"Connecting to {args.robot} ...")
    controller = make_controller(args.robot)
    controller.connect()
    app = build_app(controller, token=token)
    print(f"Remote API on http://{args.host}:{args.port}  (robot={args.robot})")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        controller.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the server boots offline (and reject bad usage)**

Run (background, then probe): start it, hit `/health`, then kill it:

```bash
ROBOT_API_TOKEN=test ./robot_control/bin/python scripts/sim.py api ur15 --port 8011 >/tmp/api_boot.log 2>&1 &
APID=$!
for i in $(seq 1 60); do grep -q "Uvicorn running" /tmp/api_boot.log 2>/dev/null && break; sleep 1; done
curl -s -H "Authorization: Bearer test" http://localhost:8011/health
echo
curl -s -o /dev/null -w "no-auth=%{http_code}\n" http://localhost:8011/state
kill $APID 2>/dev/null
cat /tmp/api_boot.log | tail -5
```

Expected: `/health` returns `{"ok":true,"robot":"ur15"}`; `no-auth=401`; the log shows `[sim] offline simulator active … (api)` and `Uvicorn running`. No traceback.

- [ ] **Step 4: Commit**

```bash
git add scripts/api_server.py lib/dispatch.py
git commit -m "feat(api): api_server.py + 'api' dispatcher target (real.py/sim.py api <robot>)"
```

---

## Task 8: End-to-end subprocess test (`sim.py api`)

**Files:** Modify `scripts/api_smoketest.py` (add a subprocess e2e using real HTTP + WS clients).

- [ ] **Step 1: Add the e2e test**

In `scripts/api_smoketest.py`, add this function above `main()` (it shells out to a real server and uses `httpx` + `websockets`):

```python
def test_e2e_subprocess():
    import json
    import signal
    import socket
    import subprocess
    import httpx

    port = 8021
    env = dict(os.environ, ROBOT_API_TOKEN=TOKEN)
    proc = subprocess.Popen(
        [os.path.join(_ROOT, "robot_control", "bin", "python"),
         os.path.join(_ROOT, "scripts", "sim.py"), "api", "ur15", "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=_ROOT,
    )
    try:
        # wait for the port to accept connections
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 60.0
        up = False
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    up = True
                    break
            except OSError:
                time.sleep(0.5)
        assert up, "API server did not come up"
        auth = {"Authorization": f"Bearer {TOKEN}"}
        # give the app a moment after the port opens
        time.sleep(1.0)
        with httpx.Client(base_url=base, timeout=10.0) as cl:
            assert cl.get("/state").status_code == 401
            assert cl.get("/state", headers=auth).json()["robot"] == "ur15"
            lease = cl.post("/control/acquire", headers=auth).json()["lease_token"]
            h = {**auth, "X-Lease": lease}
            target = [0.0, -1.4, 1.4, -1.4, -1.4, 0.2]
            cid = cl.post("/move/joints", headers=h,
                          json={"q": target, "speed": 5.0}).json()["command_id"]
            deadline = time.monotonic() + 20.0
            status = "running"
            while time.monotonic() < deadline:
                status = cl.get(f"/command/{cid}", headers=auth).json()["status"]
                if status != "running":
                    break
                time.sleep(0.1)
            assert status == "done", f"e2e move status {status}"
            st = cl.get("/state", headers=auth).json()
            assert max(abs(a - b) for a, b in zip(st["q"], target)) < 1e-6
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("PASS test_e2e_subprocess")
```

Wire it into `main()` LAST (after `test_watchdog()`).

- [ ] **Step 2: Run it, verify it PASSES**

Run: `./robot_control/bin/python scripts/api_smoketest.py`
Expected: all PASS lines including `PASS test_e2e_subprocess`, then `ALL API SMOKE TESTS PASSED`, exit 0. (The e2e launches a real server subprocess — allow time for jax/pyroki import + the move.)

- [ ] **Step 3: Commit**

```bash
git add scripts/api_smoketest.py
git commit -m "test(api): end-to-end against a real sim.py api server (httpx client)"
```

---

## Task 9: Documentation

**Files:** Modify `CLAUDE.md`; modify `README.md`.

- [ ] **Step 1: Add the API to the `scripts/` + `lib/` layout tree in `CLAUDE.md`**

Read `CLAUDE.md`. In the `scripts/` block of the project-layout tree, after `sim_smoketest.py`, add (fix connectors so the last entry keeps `└──`):

```
│   ├── api_server.py           #   remote control API server: real.py/sim.py api <ur15|gofa>
│   └── api_smoketest.py        #   stdlib-assert API test (TestClient + subprocess e2e)
```
(adjust the previously-last `sim_smoketest.py` connector to `├──`).

In the `lib/` block, after `control/`, add:

```
│   └── robot_api.py            #   FastAPI app over a RobotController (build_app) — the remote API
```
(adjust `control/`'s connector to `├──`).

- [ ] **Step 2: Add a "Remote API" section to `CLAUDE.md`**

Immediately after the `## RobotController core — \`lib/control/\`` section (before the `# UR15` heading), insert:

```
## Remote API — `scripts/api_server.py`

`real.py api <ur15|gofa>` (or `sim.py api <ur15|gofa>` offline) serves a FastAPI
HTTP+WebSocket remote API over the controller (`lib/robot_api.py`). High-level goals
only — the network carries commands + telemetry, never the servo loop. Endpoints (all
need `Authorization: Bearer $ROBOT_API_TOKEN`): `GET /state`,`/health`;
`POST /control/acquire`|`/release` (single write lease → `lease_token`, sent as the
`X-Lease` header on writes); `POST /move/joints`|`/move/pose`|`/play`|`/gripper`
(→ `202 {command_id}`, async — poll `GET /command/{id}` or watch `WS /telemetry`);
`POST /stop`|`/estop` (any authed client). The lease holder must keep its
`/telemetry` WS open as a heartbeat — if it lapses during an active motion the
watchdog stops the arm and releases the lease. Runs fully offline:
`./robot_control/bin/python scripts/api_smoketest.py` (TestClient + a subprocess e2e).
The embedded viser viewer + Live control are a follow-on.
```

- [ ] **Step 3: Add a run example to `README.md`**

In `README.md`, after the existing run block (the `real.py`/`sim.py` bash block), add a short line:

```
Remote control API (offline): `ROBOT_API_TOKEN=secret ./robot_control/bin/python scripts/sim.py api ur15` → HTTP+WS on `:8000` (see `CLAUDE.md` → Remote API).
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document the remote control API"
```

---

## Self-Review notes (for the implementer)

- **No pytest:** `scripts/api_smoketest.py` uses FastAPI's `TestClient` (in-process, over a sim-backed controller) for the unit tests and one real-subprocess e2e. Don't add pytest.
- **Async vs sync handlers:** the command/state endpoints are plain `def` (FastAPI runs them in a threadpool; they call the thread-safe controller directly). The `/telemetry` WS handler is `async` and calls `controller.get_state()` (a fast lock+copy) directly — acceptable.
- **`@app.on_event("startup")`** is used for the watchdog task; if the installed FastAPI version warns it's deprecated, that's fine (still functional). Do not refactor to lifespan unless it actually fails.
- **Watchdog semantics:** the lease's `last_seen` is refreshed by (a) any write command via `check_lease`, and (b) an open lease-matched `/telemetry` WS each tick. So a controlling client must hold the telemetry WS open during motion; `test_watchdog` deliberately holds neither and asserts the motion is stopped.
- **Out of scope here:** the embedded viser viewer, the controller Live primitive (`set_live_target`/`start_live`), TLS — all in the follow-on plan. The API exposes high-level goals only (no Live/streaming endpoint), per the spec.
```
