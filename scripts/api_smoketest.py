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


def _client(robot="ur15", watchdog_timeout_s=30.0):
    """Build a TestClient over a sim-backed controller. Returns (client, controller).
    Default watchdog_timeout_s is high so multi-second moves/plays in the command
    tests aren't stopped by the deadman; test_watchdog overrides it to 0.5."""
    robot_sim.install(robot)
    # import after install() so the sim shim is already in sys.modules
    from control import make_controller
    from fastapi.testclient import TestClient
    from robot_api import build_app
    c = make_controller(robot)
    c.connect()
    app = build_app(c, token=TOKEN, telem_hz=50.0, watchdog_timeout_s=watchdog_timeout_s)
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
                           json={"q": robot_sim.UR_HOME, "speed": 1.0}).status_code == 423
        # malformed q (wrong length) -> 422 (must not reach servoJ)
        assert client.post("/move/joints", headers=h,
                           json={"q": [0.0, 1.0, 2.0], "speed": 1.0}).status_code == 422
        # move with lease -> 202 + command_id, completes, state reaches target
        target = [0.0, -1.4, 1.4, -1.4, -1.4, 0.2]
        r = client.post("/move/joints", headers=h, json={"q": target, "speed": 1.0})
        assert r.status_code == 202, r.text
        cid = r.json()["command_id"]
        assert _poll_command(client, cid) == "done"
        st = client.get("/state", headers=_auth()).json()
        assert max(abs(a - b) for a, b in zip(st["q"], target)) < 1e-6
        # play by name -> done
        rp = client.post("/play", headers=h, json={"name": "_sample_ur15", "speed": 1.0})
        assert rp.status_code == 202
        assert _poll_command(client, rp.json()["command_id"]) == "done"
        # gripper -> done
        rg = client.post("/gripper", headers=h, json={"frac": 0.5})
        assert rg.status_code == 202
        assert _poll_command(client, rg.json()["command_id"]) == "done"
        # speed out of (0, 1.0] -> 422 (no zero/negative loop-wedge; no exceeding the joint-speed cap)
        for bad in (0, -1.0, 5.0):
            assert client.post("/move/joints", headers=h,
                               json={"q": target, "speed": bad}).status_code == 422, f"speed={bad}"
        # play with an unknown trajectory name -> 400, not a 500
        assert client.post("/play", headers=h, json={"name": "__nope__"}).status_code == 400
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
                        json={"q": [0.0, 0.1, 0.0, 0.0, 1.5708, 0.0], "speed": 1.0})
        assert r.status_code == 202
        assert _poll_command(client, r.json()["command_id"]) == "done"
    finally:
        c.close()
    print("PASS test_gofa_no_gripper")


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
        raised = False
        try:
            with client.websocket_connect("/telemetry?token=nope") as ws:
                ws.receive_json()
        except Exception:
            raised = True
        assert raised, "WS with bad token should be rejected"
    finally:
        c.close()
    print("PASS test_telemetry_auth")


def test_watchdog():
    # short watchdog so the deadman fires quickly; all other tests use the default 30 s
    client, c = _client("ur15", watchdog_timeout_s=0.5)
    try:
        lease = client.post("/control/acquire", headers=_auth()).json()["lease_token"]
        h = {**_auth(), "X-Lease": lease}
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


def test_e2e_subprocess():
    import signal
    import socket
    import subprocess
    import httpx

    port = 18021
    # Kill any stray server left from a previous failed run on this port.
    subprocess.run(
        ["bash", "-c", f"lsof -ti:{port} | xargs kill -9 2>/dev/null; true"],
        check=False, capture_output=True,
    )
    env = dict(os.environ, ROBOT_API_TOKEN=TOKEN)
    proc = subprocess.Popen(
        [os.path.join(_ROOT, "robot_control", "bin", "python"),
         os.path.join(_ROOT, "scripts", "sim.py"), "api", "ur15", "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=_ROOT,
    )
    try:
        # Wait until OUR subprocess's port is ready — poll the process to detect early exit,
        # and verify the listening socket belongs to our PID (not a stale server).
        deadline = time.monotonic() + 60.0
        up = False
        while time.monotonic() < deadline:
            ret = proc.poll()
            if ret is not None:
                out = proc.stdout.read().decode(errors="replace")
                raise RuntimeError(f"Server exited early (rc={ret}):\n{out}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    up = True
                    break
            except OSError:
                time.sleep(0.5)
        if not up:
            out = proc.stdout.read().decode(errors="replace")
            raise AssertionError(f"API server did not come up within 60s. Output:\n{out}")
        base = f"http://127.0.0.1:{port}"
        auth = {"Authorization": f"Bearer {TOKEN}"}
        with httpx.Client(base_url=base, timeout=10.0) as cl:
            # Retry until the app returns a real HTTP response. Uvicorn may accept TCP
            # connections before the ASGI app is fully initialized (while JAX/pyroki
            # finish loading), causing connection resets for a few seconds.
            http_deadline = time.monotonic() + 30.0
            while True:
                ret = proc.poll()
                if ret is not None:
                    out = proc.stdout.read().decode(errors="replace")
                    raise RuntimeError(f"Server exited during HTTP wait (rc={ret}):\n{out}")
                try:
                    r = cl.get("/state")
                    assert r.status_code == 401
                    break
                except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError):
                    if time.monotonic() >= http_deadline:
                        out = proc.stdout.read().decode(errors="replace")
                        raise AssertionError(
                            f"Server did not serve HTTP within 30s. Output:\n{out}"
                        )
                    time.sleep(0.5)
            assert cl.get("/state", headers=auth).json()["robot"] == "ur15"
            lease = cl.post("/control/acquire", headers=auth).json()["lease_token"]
            h = {**auth, "X-Lease": lease}
            # small move from the sim home so it completes well within the server's
            # 2s watchdog (this e2e doesn't hold a telemetry-WS heartbeat; the watchdog
            # deadman is tested separately in test_watchdog).
            target = [0.0, -1.0, 1.0, 0.0, 1.0, 0.2]
            cid = cl.post("/move/joints", headers=h,
                          json={"q": target, "speed": 1.0}).json()["command_id"]
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
            proc.wait()          # reap so it doesn't linger as a zombie
        proc.stdout.close()      # release the pipe FD
    print("PASS test_e2e_subprocess")


def main():
    test_state_and_auth()
    test_lease()
    test_commands()
    test_gofa_no_gripper()
    test_telemetry_ws()
    test_telemetry_auth()
    test_watchdog()
    test_e2e_subprocess()
    print("ALL API SMOKE TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
