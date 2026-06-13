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
