"""FastAPI remote-control API over a RobotController.

build_app(controller, token, ...) returns a FastAPI app exposing read/state +
async high-level commands + a telemetry WebSocket, gated by a bearer token and a
single write lease (see the design spec). The controller is the single hardware
owner; this module only adapts it to HTTP/WS.
"""
from __future__ import annotations

import asyncio
import math
import os
import secrets
import threading
import time

from fastapi import Body, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect

import robot_common as rc
from control import Busy


def build_app(controller, token: str, telem_hz: float = 20.0,
              watchdog_timeout_s: float = 2.0) -> FastAPI:
    app = FastAPI(title="robot-control-api")
    # single write lease: {"token": str|None, "last_seen": monotonic float}
    lease = {"token": None, "last_seen": 0.0}
    _lease_lock = threading.Lock()   # guards acquire/release + command validate-then-submit

    def check_auth(authorization: str | None) -> None:
        if not token:                       # token unset => auth disabled (the server entry requires one)
            return
        # constant-time compare (LAN tool, but the right habit for a secret) + RFC 6750 challenge header
        if not authorization or not secrets.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="bad or missing token",
                                headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health")
    def health(authorization: str = Header(None)):
        check_auth(authorization)
        return {"ok": True, "robot": controller.robot_name}

    @app.get("/state")
    def state(authorization: str = Header(None)):
        check_auth(authorization)
        return controller.get_state().to_dict()

    # The caller holds _lease_lock when the lease check must be atomic with a state
    # change (acquire/release here; validate-then-submit in the command endpoints).
    def check_lease(x_lease: str | None) -> None:
        if lease["token"] is None or x_lease != lease["token"]:
            raise HTTPException(status_code=423, detail="no or invalid control lease")
        lease["last_seen"] = time.monotonic()

    @app.post("/control/acquire")
    def acquire(authorization: str = Header(None), force: bool = Body(False, embed=True)):
        check_auth(authorization)
        # force is a JSON body field ({"force": true}); an empty body => force=False (embed=True)
        with _lease_lock:
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
        with _lease_lock:
            check_lease(x_lease)
            lease["token"] = None
        # free-drive only lives with the lease — relinquishing control drops a compliant arm
        if controller.get_state().activity == "freedrive":
            controller.stop_freedrive()
        return {"released": True}

    def _submit(fn):
        try:
            return {"command_id": fn()}
        except Busy as e:
            raise HTTPException(status_code=409, detail=str(e))
        except (FileNotFoundError, KeyError, ValueError) as e:
            # bad trajectory name / malformed waypoint / bad value caught before motion
            raise HTTPException(status_code=400, detail=f"bad command: {e}")

    def _check_vec(name: str, v, n: int) -> None:
        if not isinstance(v, list) or len(v) != n:
            raise HTTPException(status_code=422, detail=f"{name} must be a list of {n} numbers")
        if any(not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x) for x in v):
            raise HTTPException(status_code=422, detail=f"{name} has non-finite or non-numeric values")

    def _check_speed(speed) -> None:
        # cap at 1.0 so the API can't exceed MAX_JOINT_SPEED; reject <=0 (a negative/zero
        # speed makes the alpha-profile loop never terminate, wedging the motion slot).
        if not isinstance(speed, (int, float)) or not (0 < speed <= 1.0):
            raise HTTPException(status_code=422, detail="speed must be a number in (0, 1.0]")

    def _check_name(name) -> None:
        try:
            rc.validate_traj_name(name)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    def _check_waypoints(waypoints) -> None:
        if not isinstance(waypoints, list) or not waypoints:
            raise HTTPException(status_code=422, detail="waypoints must be a non-empty list")
        for i, wp in enumerate(waypoints):
            if not isinstance(wp, dict):
                raise HTTPException(status_code=422, detail=f"waypoint {i} must be an object")
            _check_vec(f"waypoint {i} pos", wp.get("pos"), 3)
            _check_vec(f"waypoint {i} wxyz", wp.get("wxyz"), 4)
            if wp.get("q") is not None:
                _check_vec(f"waypoint {i} q", wp["q"], controller.NUM_JOINTS)
            grip = wp.get("grip")
            if grip is not None and (not isinstance(grip, (int, float))
                                     or math.isnan(grip) or math.isinf(grip)):
                raise HTTPException(status_code=422,
                                    detail=f"waypoint {i} grip must be null or a finite number")

    @app.post("/move/joints", status_code=202)
    def move_joints(authorization: str = Header(None), x_lease: str = Header(None),
                    q: list = Body(...), speed: float = Body(1.0)):
        check_auth(authorization)
        _check_vec("q", q, controller.NUM_JOINTS)
        _check_speed(speed)
        with _lease_lock:                       # validate lease + submit atomically (vs force-steal)
            check_lease(x_lease)
            return _submit(lambda: controller.move_to_joints(q, speed))

    @app.post("/move/pose", status_code=202)
    def move_pose(authorization: str = Header(None), x_lease: str = Header(None),
                  pos: list = Body(...), wxyz: list = Body(...), speed: float = Body(1.0)):
        check_auth(authorization)
        _check_vec("pos", pos, 3)
        _check_vec("wxyz", wxyz, 4)
        _check_speed(speed)
        with _lease_lock:
            check_lease(x_lease)
            return _submit(lambda: controller.move_to_pose(pos, wxyz, speed))

    @app.post("/play", status_code=202)
    def play(authorization: str = Header(None), x_lease: str = Header(None),
             name: str = Body(None), names: list = Body(None),
             waypoints: list = Body(None), speed: float = Body(1.0)):
        check_auth(authorization)
        _check_speed(speed)
        # exactly one of: name (single), names (chain — the server concatenates each
        # trajectory's waypoints into one continuous motion), or waypoints (inline list).
        target = names if names is not None else (name if name is not None else waypoints)
        if target is None:
            raise HTTPException(status_code=400, detail="provide 'name', 'names', or 'waypoints'")
        with _lease_lock:
            check_lease(x_lease)
            # a bad/missing name (incl. in a chain) raises in load -> mapped to 400 by _submit
            return _submit(lambda: controller.play(target, speed))

    @app.post("/gripper", status_code=202)
    def gripper(authorization: str = Header(None), x_lease: str = Header(None),
                frac: float = Body(..., embed=True)):
        check_auth(authorization)
        # gripper capability is static (None for GoFa, never changes) — safe to check outside the lock
        if controller.get_state().gripper_frac is None:
            raise HTTPException(status_code=400, detail="this robot has no gripper")
        with _lease_lock:
            check_lease(x_lease)
            return _submit(lambda: controller.set_gripper(frac))

    @app.post("/freedrive")
    def freedrive(authorization: str = Header(None), x_lease: str = Header(None),
                  on: bool = Body(..., embed=True)):
        # Hand-guiding is a lease-gated mode toggle, not an async command (no command_id).
        # It's mutually exclusive with motion: starting it 409s if a command is running,
        # and while it's on the command endpoints 409 with "busy with free-drive".
        check_auth(authorization)
        with _lease_lock:
            check_lease(x_lease)
            try:
                controller.start_freedrive() if on else controller.stop_freedrive()
            except Busy as e:
                raise HTTPException(status_code=409, detail=str(e))
        return {"freedrive": bool(on)}

    # ---- trajectory authoring: list/load are auth-only reads; save/delete are
    #      lease-gated file writes (they never touch the arm, but write control
    #      should own the arm's trajectory set). Names are validated before any path
    #      is built (path-traversal guard). All scoped to trajectories/<robot>/.
    @app.get("/trajectories")
    def list_trajs(authorization: str = Header(None)):
        check_auth(authorization)
        return {"trajectories": rc.list_trajectories(controller.robot_name)}

    @app.get("/trajectories/{name}")
    def get_traj(name: str, authorization: str = Header(None)):
        check_auth(authorization)
        _check_name(name)
        try:
            return rc.load_trajectory(name, controller.robot_name)
        except FileNotFoundError:
            raise HTTPException(status_code=404,
                                detail=f"no trajectory {name!r} for {controller.robot_name}")

    @app.post("/trajectories")
    def save_traj(authorization: str = Header(None), x_lease: str = Header(None),
                  name: str = Body(...), waypoints: list = Body(...)):
        check_auth(authorization)
        _check_name(name)
        _check_waypoints(waypoints)
        with _lease_lock:
            check_lease(x_lease)
            rc.save_trajectory(name, controller.robot_name, waypoints)
        return {"saved": True, "name": name}

    @app.delete("/trajectories/{name}")
    def delete_traj(name: str, authorization: str = Header(None), x_lease: str = Header(None)):
        check_auth(authorization)
        _check_name(name)
        with _lease_lock:
            check_lease(x_lease)
            path = os.path.join(rc.TRAJ_DIR, controller.robot_name, f"{name}.json")
            if not os.path.exists(path):
                raise HTTPException(status_code=404,
                                    detail=f"no trajectory {name!r} for {controller.robot_name}")
            os.remove(path)
        return {"deleted": True, "name": name}

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

    def _watchdog_loop():
        # deadman: if the lease holder goes silent (no heartbeat WS, no commands) while the
        # arm is "live" — a motion running OR free-drive engaged — stop the arm and release
        # the lease. Exits when the controller is closed. (lease[...] accesses are single
        # dict ops — GIL-atomic.)
        while not controller.closed:
            time.sleep(0.1)
            tok = lease["token"]
            if tok is None:
                continue
            st = controller.get_state()
            ac = st.active_command
            live = (ac is not None and ac["status"] == "running") or st.activity == "freedrive"
            if not live:
                continue
            if (time.monotonic() - lease["last_seen"]) <= watchdog_timeout_s:
                continue
            with _lease_lock:
                # re-validate under the lock: only fire if it's still the same stale
                # lease (a force-acquire since our check would have changed token + last_seen)
                if (lease["token"] == tok
                        and (time.monotonic() - lease["last_seen"]) > watchdog_timeout_s):
                    controller.stop()       # deadman stop (also ends free-drive)
                    lease["token"] = None    # release the lease
    threading.Thread(target=_watchdog_loop, daemon=True, name="api-watchdog").start()

    return app


def build_multi_app(controllers: dict, token: str, telem_hz: float = 20.0,
                    watchdog_timeout_s: float = 2.0, unavailable=()) -> FastAPI:
    """Serve several arms from one server: mount build_app(ctrl) at /<name> for each
    connected controller, and advertise the roster (incl. arms that failed to connect)
    at /robots so a client can render a per-arm switcher. Each arm keeps its own
    independent lease, watchdog and telemetry — this is just a parent that namespaces
    them. Single-arm servers keep using build_app at the root; this is only the no-arm
    `api` mode."""
    app = FastAPI(title="robot-control-api (multi)")
    _order = {"ur15": 0, "gofa": 1}
    roster = ([{"name": n, "available": True} for n in controllers]
              + [{"name": n, "available": False} for n in unavailable])
    roster.sort(key=lambda r: _order.get(r["name"], 9))

    def _auth(authorization: str | None) -> None:
        if token and (not authorization
                      or not secrets.compare_digest(authorization, f"Bearer {token}")):
            raise HTTPException(status_code=401, detail="bad or missing token",
                                headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health")
    def health(authorization: str = Header(None)):
        _auth(authorization)
        return {"ok": True, "multi": True, "robots": roster}

    @app.get("/robots")
    def robots(authorization: str = Header(None)):
        _auth(authorization)
        return {"robots": roster}

    for name, ctrl in controllers.items():
        app.mount(f"/{name}", build_app(ctrl, token, telem_hz, watchdog_timeout_s))
    return app
