"""
One-shot Hand-E comms probe. Run this BEFORE trusting gripper control in
teleop_ur15.py.

Prereq (control path A): the UR's tool RS-485 must be exposed as a TCP socket.
Swap the Robotiq *Grippers* URCap for the Robotiq *RS485* URCap (or enable the
Tool Communication Interface) so a background daemon forwards the wrist
connector to HOST:PORT below. This is independent of the running program, so it
coexists with ur_rtde.

What it does: connect, activate, open, close, open, printing status each step.
If this passes, the same HandEGripper drives teleop_ur15.py unchanged.

  ./robot_control/bin/python verify_hande.py
"""

import sys
import time

import hande_gripper

HOST = "192.168.125.2"   # must match ROBOT_IP in teleop_ur15.py
PORT = hande_gripper.DEFAULT_PORT


def main() -> int:
    print(f"Connecting to Hand-E socket at {HOST}:{PORT}")
    g = hande_gripper.HandEGripper(HOST, PORT)
    try:
        g.connect()
    except Exception as e:
        print(f"  connect FAILED: {e}")
        print("  -> Is the RS485/Tool-Comm URCap installed and forwarding this port?")
        print("     On PolyScope X the port/forwarding may differ; this is the unknown")
        print("     we're verifying. Check the URCap's exposed socket and update PORT.")
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
