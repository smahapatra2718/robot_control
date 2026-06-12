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
    # import after install() so the sim shim is already in sys.modules
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
