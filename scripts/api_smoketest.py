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
        # force-steal invalidated the original token: releasing with it -> 423
        assert client.post("/control/release",
                           headers={**_auth(), "X-Lease": lease_token}).status_code == 423
        # release with no X-Lease header -> 423
        assert client.post("/control/release", headers=_auth()).status_code == 423
        # release with the (new) lease
        rel = client.post("/control/release", headers={**_auth(), "X-Lease": r2.json()["lease_token"]})
        assert rel.status_code == 200
        # after release, acquire works again
        assert client.post("/control/acquire", headers=_auth()).status_code == 200
    finally:
        c.close()
    print("PASS test_lease")


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
        # malformed q (wrong length) -> 422 (must not reach servoJ)
        assert client.post("/move/joints", headers=h,
                           json={"q": [0.0, 1.0, 2.0], "speed": 1.0}).status_code == 422
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


def main():
    test_state_and_auth()
    test_lease()
    test_commands()
    test_gofa_no_gripper()
    print("ALL API SMOKE TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
