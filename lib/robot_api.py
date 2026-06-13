"""FastAPI remote-control API over a RobotController.

build_app(controller, token, ...) returns a FastAPI app exposing read/state +
async high-level commands + a telemetry WebSocket, gated by a bearer token and a
single write lease (see the design spec). The controller is the single hardware
owner; this module only adapts it to HTTP/WS.
"""
from __future__ import annotations

import asyncio
import math
import secrets
import threading
import time

from fastapi import Body, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect

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
        return {"released": True}

    def _submit(fn):
        try:
            return {"command_id": fn()}
        except Busy as e:
            raise HTTPException(status_code=409, detail=str(e))

    def _check_vec(name: str, v, n: int) -> None:
        if not isinstance(v, list) or len(v) != n:
            raise HTTPException(status_code=422, detail=f"{name} must be a list of {n} numbers")
        if any(not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x) for x in v):
            raise HTTPException(status_code=422, detail=f"{name} has non-finite or non-numeric values")

    @app.post("/move/joints", status_code=202)
    def move_joints(authorization: str = Header(None), x_lease: str = Header(None),
                    q: list = Body(...), speed: float = Body(1.0)):
        check_auth(authorization)
        _check_vec("q", q, controller.NUM_JOINTS)
        with _lease_lock:                       # validate lease + submit atomically (vs force-steal)
            check_lease(x_lease)
            return _submit(lambda: controller.move_to_joints(q, speed))

    @app.post("/move/pose", status_code=202)
    def move_pose(authorization: str = Header(None), x_lease: str = Header(None),
                  pos: list = Body(...), wxyz: list = Body(...), speed: float = Body(1.0)):
        check_auth(authorization)
        _check_vec("pos", pos, 3)
        _check_vec("wxyz", wxyz, 4)
        with _lease_lock:
            check_lease(x_lease)
            return _submit(lambda: controller.move_to_pose(pos, wxyz, speed))

    @app.post("/play", status_code=202)
    def play(authorization: str = Header(None), x_lease: str = Header(None),
             name: str = Body(None), waypoints: list = Body(None), speed: float = Body(1.0)):
        check_auth(authorization)
        target = name if name is not None else waypoints
        if target is None:
            raise HTTPException(status_code=400, detail="provide 'name' or 'waypoints'")
        with _lease_lock:
            check_lease(x_lease)
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

    return app
