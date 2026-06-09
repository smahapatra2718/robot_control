#!/usr/bin/env python
"""sim.py <ur15|gofa|play|teleop> [args] — run a teleop entry point OFFLINE (no robot).

Injects fake robot transports (lib/robot_sim.py) into sys.modules, then runs the
real, unmodified script. The real twin is real.py.
  ./robot_control/bin/python scripts/sim.py ur15
  ./robot_control/bin/python scripts/sim.py play _sample_ur15 --no-confirm
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dispatch import dispatch  # noqa: E402

dispatch("sim.py", sys.argv[1:], sim=True)
