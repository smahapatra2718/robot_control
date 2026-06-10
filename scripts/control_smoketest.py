"""Offline smoke test for the RobotController core (lib/control), driven against
the sim fakes (lib/robot_sim). No robot, no network.

  ./robot_control/bin/python scripts/control_smoketest.py

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
from control.state import RobotState  # noqa: E402


def test_state_dataclass():
    s = RobotState(
        ts=1.0, robot="ur15", q=[0.0] * 6,
        pose={"pos": [0.1, 0.2, 0.3], "wxyz": [1.0, 0.0, 0.0, 0.0]},
        gripper_frac=0.0, safety_state="NORMAL", controller_state="ok",
        activity="idle", active_command=None, conn_ok=True,
    )
    d = s.to_dict()
    assert d["robot"] == "ur15"
    assert d["q"] == [0.0] * 6
    assert d["pose"]["pos"] == [0.1, 0.2, 0.3]
    assert d["pose"]["wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert d["ts"] == 1.0
    assert d["gripper_frac"] == 0.0
    assert d["conn_ok"] is True
    assert d["active_command"] is None
    assert d["health"] == {}
    print("PASS test_state_dataclass")


def test_ur_connect_state():
    robot_sim.install("ur15")
    from control import make_controller
    c = make_controller("ur15")
    c.connect()
    try:
        st = c.get_state()
        assert st.robot == "ur15"
        assert st.q == robot_sim.UR_HOME, "state q should be the seeded UR home"
        assert st.safety_state == "NORMAL"
        assert st.gripper_frac == 0.0
        assert len(st.pose["pos"]) == 3 and len(st.pose["wxyz"]) == 4
        assert st.activity == "idle"
    finally:
        c.close()
    print("PASS test_ur_connect_state")


def test_ur_move():
    robot_sim.install("ur15")
    from control import make_controller
    c = make_controller("ur15")
    c.connect()
    try:
        target = [0.0, -1.4, 1.4, -1.4, -1.4, 0.2]
        cid = c.move_to_joints(target, speed=5.0)
        assert c.wait(cid, timeout=20.0) == "done", "move did not complete"
        st = c.get_state()
        assert max(abs(a - b) for a, b in zip(st.q, target)) < 1e-6, "did not reach target"
        cid2 = c.move_to_joints(robot_sim.UR_HOME, speed=5.0)
        assert c.wait(cid2, timeout=20.0) == "done"
    finally:
        c.close()
    print("PASS test_ur_move")


def test_ur_play_gripper():
    robot_sim.install("ur15")
    from control import make_controller
    c = make_controller("ur15")
    c.connect()
    try:
        cid = c.play("_sample_ur15", speed=5.0)
        assert c.wait(cid, timeout=40.0) == "done", "play did not complete"
        import robot_common as rc
        wps = rc.load_trajectory("_sample_ur15")["waypoints"]
        q_final = wps[-1]["q"]
        st = c.get_state()
        assert max(abs(a - b) for a, b in zip(st.q, q_final)) < 1e-6, "play did not reach final waypoint"
        # the play's gripper-on-change should leave the tracked grip at the final waypoint's grip
        expected_grip = rc.norm_grip(wps[-1].get("grip"))
        if expected_grip is not None:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and abs(c.get_state().gripper_frac - expected_grip) > 1e-9:
                time.sleep(0.02)
            assert abs(c.get_state().gripper_frac - expected_grip) < 1e-9, "play did not leave gripper at final grip"
        gid = c.set_gripper(0.5)
        assert c.wait(gid, timeout=10.0) == "done"
        # state is eventually-consistent at POLL_HZ — let the poll thread catch up
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and abs(c.get_state().gripper_frac - 0.5) > 1e-9:
            time.sleep(0.02)
        assert abs(c.get_state().gripper_frac - 0.5) < 1e-9, "set_gripper did not update state"
    finally:
        c.close()
    print("PASS test_ur_play_gripper")


def test_gofa_connect_state():
    robot_sim.install("gofa")
    from control import make_controller
    c = make_controller("gofa")
    c.connect()
    try:
        st = c.get_state()
        assert st.robot == "gofa"
        assert st.q == robot_sim.GOFA_HOME, "state q should be the seeded GoFa home"
        assert st.gripper_frac is None, "GoFa has no gripper"
        assert len(st.pose["pos"]) == 3 and len(st.pose["wxyz"]) == 4
    finally:
        c.close()
    print("PASS test_gofa_connect_state")


def test_gofa_move_play():
    robot_sim.install("gofa")
    from control import make_controller
    c = make_controller("gofa")
    c.connect()
    try:
        target = [0.0, 0.1, 0.0, 0.0, 1.5708, 0.0]
        cid = c.move_to_joints(target, speed=5.0)
        assert c.wait(cid, timeout=30.0) == "done", "gofa move did not complete"
        st = c.get_state()
        assert max(abs(a - b) for a, b in zip(st.q, target)) < 1e-6, "gofa did not reach target"
        pid = c.play("_sample_gofa", speed=5.0)
        assert c.wait(pid, timeout=40.0) == "done", "gofa play did not complete"
        import robot_common as rc
        q_final = rc.load_trajectory("_sample_gofa")["waypoints"][-1]["q"]
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and \
                max(abs(a - b) for a, b in zip(c.get_state().q, q_final)) > 1e-6:
            time.sleep(0.02)
        assert max(abs(a - b) for a, b in zip(c.get_state().q, q_final)) < 1e-6, \
            "gofa play did not reach final waypoint"
    finally:
        c.close()
    print("PASS test_gofa_move_play")


def main():
    test_state_dataclass()
    test_ur_connect_state()
    test_ur_move()
    test_ur_play_gripper()
    test_gofa_connect_state()
    test_gofa_move_play()
    print("ALL CONTROL SMOKE TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
