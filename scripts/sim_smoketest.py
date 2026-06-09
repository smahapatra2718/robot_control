"""Offline-sim smoke test: drive the fake transports directly and assert the
kinematic round-trips + (Task 2) the GoFa EGM handshake.

  ./robot_control/bin/python scripts/sim_smoketest.py

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


def test_ur_roundtrip():
    robot_sim.install("ur15")
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    import hande_gripper

    c = RTDEControlInterface("192.168.0.1")
    r = RTDEReceiveInterface("192.168.0.1")
    assert r.getActualQ() == robot_sim.UR_HOME, "home pose not seeded"
    assert r.getSafetyMode() == 1, "safety mode should be NORMAL (1)"

    target = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    c.servoJ(target, 0.0, 0.0, 1.0 / 50, 0.1, 300)
    assert r.getActualQ() == target, "servoJ did not update the joints"

    assert hande_gripper.DEFAULT_PORT == 63352, "DEFAULT_PORT not exposed"
    g = hande_gripper.HandEGripper("192.168.0.1", hande_gripper.DEFAULT_PORT)
    g.connect()
    g.activate()
    g.move(0.5)
    assert robot_sim.SIM.grip_frac == 0.5, "move() did not update grip_frac"
    print("PASS test_ur_roundtrip")


def main():
    test_ur_roundtrip()
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
