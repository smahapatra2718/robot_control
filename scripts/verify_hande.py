"""
One-shot Hand-E comms probe. Run this BEFORE trusting gripper control in
teleop_ur15.py.

Prereq: the Robotiq *Grippers* URCap must be installed (its background daemon
serves the gripper on <robot_ip>:63352, independent of the running program, so
it coexists with ur_rtde). Nothing to swap. On PolyScope X the Services
firewall may block 63352 by default -- if connect fails, allow that port
(Settings -> Security -> Services), the same place you enabled RTDE/Dashboard.

What it does: connect, activate, open, close, open, printing status each step.
If this passes, the same HandEGripper drives teleop_ur15.py unchanged.

  ./robot_control/bin/python scripts/verify_hande.py
"""

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "lib")):  # repo root + lib/ (our modules)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hande_gripper  # noqa: E402
import robot_common as rc  # noqa: E402

HOST = rc.UR_ROBOT_IP    # same UR controller as teleop_ur15.py
PORT = hande_gripper.DEFAULT_PORT


def main() -> int:
    print(f"Connecting to Robotiq URCap socket at {HOST}:{PORT}")
    g = hande_gripper.HandEGripper(HOST, PORT)
    try:
        g.connect()
    except Exception as e:
        print(f"  connect FAILED: {e}")
        print("  -> Is the Robotiq Grippers URCap installed?")
        print("  -> Is port 63352 open in the PolyScope X Services firewall?")
        print("     (Settings -> Security -> Services -- same place as RTDE/Dashboard.)")
        return 1
    print("  connected.")

    try:
        print("Activating...")
        g.activate()
        print(f"  activated. status={g.status()}")

        for label, action in (("OPEN", g.open), ("CLOSE", g.close_gripper), ("OPEN", g.open)):
            print(f"{label}...")
            action()
            time.sleep(1.5)
            print(f"  status={g.status()}")
    except Exception as e:
        print(f"  FAILED during motion: {e}")
        return 1
    finally:
        g.close()

    print("\nOK -- Hand-E responds. teleop_ur15.py gripper control is good to go.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
