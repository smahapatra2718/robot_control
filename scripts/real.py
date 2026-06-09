#!/usr/bin/env python
"""real.py <ur15|gofa|play|teleop> [args] — run a teleop entry point on real hardware.

Thin verb over lib/dispatch.py; the offline twin is sim.py.
  ./robot_control/bin/python scripts/real.py ur15
  ./robot_control/bin/python scripts/real.py play traj1 --speed 0.5
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dispatch import dispatch  # noqa: E402

dispatch("real.py", sys.argv[1:], sim=False)
