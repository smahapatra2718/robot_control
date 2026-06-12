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

    return app
