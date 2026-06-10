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


def main():
    test_state_dataclass()
    print("ALL CONTROL SMOKE TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
