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
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import robot_common as rc  # noqa: E402
from camera import open_camera  # noqa: E402
from control import make_controller  # noqa: E402
from robot_api import build_app, build_multi_app  # noqa: E402

ROBOTS = ("ur15", "gofa")
_DASHBOARD = os.path.join(_ROOT, "web", "dashboard.html")


def main() -> None:
    ap = argparse.ArgumentParser(description="Remote control API server (UR15 / GoFa).")
    ap.add_argument("robot", nargs="?", choices=list(ROBOTS), default=None,
                    help="arm to serve; omit to serve BOTH arms with a switcher (tolerant: "
                         "starts even if one arm is unreachable)")
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address (default 0.0.0.0 = all interfaces; use 127.0.0.1 for loopback only)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    token = os.environ.get("ROBOT_API_TOKEN", "changeme")
    if token == "changeme":
        print("WARNING: ROBOT_API_TOKEN not set — using 'changeme'. Set it before real use.")

    controllers: dict = {}
    cameras: dict = {}

    def _open_camera(name: str):
        """Best-effort: a missing camera degrades /camera/* to 503, it never blocks serving.
        Print the reason on failure — a silent 'none' sends you hunting the wrong thing."""
        cam = open_camera(name)
        if cam.error is None:
            print(f"  {name} camera: {cam.index!r} via {cam.backend} "
                  f"({cam.width}x{cam.height} @{cam.fps})")
        else:
            print(f"  {name} camera: unavailable — {cam.error}")
        # register it either way: a failed source has no frames so /camera/* still 503s,
        # but /camera/info can then report *why* instead of a bare "available: false".
        cameras[name] = cam
        return cam

    try:
        if args.robot:
            print(f"Connecting to {args.robot} ...")
            c = make_controller(args.robot)
            c.connect()
            controllers[args.robot] = c
            _open_camera(args.robot)
            app = build_app(c, token=token, camera=cameras.get(args.robot))
            print(f"Remote API on http://{args.host}:{args.port}  (robot={args.robot})")
        else:
            unavailable = []
            for name in ROBOTS:                      # tolerant: serve whatever connects
                print(f"Connecting to {name} ...")
                c = make_controller(name)
                try:
                    c.connect()
                    controllers[name] = c
                    print(f"  {name}: online")
                    _open_camera(name)
                except Exception as e:               # noqa: BLE001 - one arm down shouldn't sink the server
                    unavailable.append(name)
                    print(f"  {name}: UNAVAILABLE ({e})")
            if not controllers:
                raise SystemExit("no arms reachable — nothing to serve")
            app = build_multi_app(controllers, token=token, unavailable=unavailable,
                                  cameras=cameras)
            print(f"Remote API on http://{args.host}:{args.port}  "
                  f"(multi: online={list(controllers)} unavailable={unavailable})")

        # Serve the static web console (talks to the robot only via the API endpoints).
        # Vendored 3D libs + baked model bundles are static assets, served unauthenticated
        # like the dashboard shell itself — the bearer token still gates all state/control.
        _WEB = os.path.join(_ROOT, "web")
        for _route, _sub in (("/vendor", "vendor"), ("/models", "models")):
            _d = os.path.join(_WEB, _sub)
            if os.path.isdir(_d):
                app.mount(_route, StaticFiles(directory=_d), name=_sub)

        @app.get("/", include_in_schema=False)
        def dashboard():
            # no-cache: always revalidate so a pulled dashboard (e.g. a changed import
            # map) can't be masked by a stale browser copy during iteration.
            return FileResponse(_DASHBOARD, headers={"Cache-Control": "no-cache"})

        print(f"Web console:  http://{args.host}:{args.port}/")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        for cam in cameras.values():
            cam.close()
        for c in controllers.values():
            c.close()


if __name__ == "__main__":
    main()
