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
    try:
        if args.robot:
            print(f"Connecting to {args.robot} ...")
            c = make_controller(args.robot)
            c.connect()
            controllers[args.robot] = c
            app = build_app(c, token=token)
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
                except Exception as e:               # noqa: BLE001 - one arm down shouldn't sink the server
                    unavailable.append(name)
                    print(f"  {name}: UNAVAILABLE ({e})")
            if not controllers:
                raise SystemExit("no arms reachable — nothing to serve")
            app = build_multi_app(controllers, token=token, unavailable=unavailable)
            print(f"Remote API on http://{args.host}:{args.port}  "
                  f"(multi: online={list(controllers)} unavailable={unavailable})")

        # Serve the static web console (talks to the robot only via the API endpoints).
        @app.get("/", include_in_schema=False)
        def dashboard():
            return FileResponse(_DASHBOARD)

        print(f"Web console:  http://{args.host}:{args.port}/")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        for c in controllers.values():
            c.close()


if __name__ == "__main__":
    main()
