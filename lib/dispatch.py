"""Shared dispatcher for the real and simulated entry points.

real.py / sim.py both call dispatch(): identical target->script map and argv
plumbing, differing only in whether the offline sim shim is installed first.
Single source of truth so real and sim never drift in what they can launch.
"""
from __future__ import annotations

import os
import runpy
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")

TARGETS = {
    "ur15": "teleop_ur15.py",
    "gofa": "teleop_gofa_egm.py",
    "play": "play_trajectory.py",
    "teleop": "teleop.py",
    "api": "api_server.py",
}


def dispatch(prog: str, argv: list[str], sim: bool = False) -> None:
    """argv = sys.argv[1:]. Resolve the target, optionally install the sim shim,
    then runpy the real script as __main__ with the remaining args."""
    if not argv or argv[0] not in TARGETS:
        print(f"usage: {prog} <{'|'.join(TARGETS)}> [args...]", file=sys.stderr)
        raise SystemExit(2)
    target, rest = argv[0], argv[1:]
    if sim:
        import robot_sim  # lazy: the real path never imports sim machinery
        robot_sim.install(target)
        print(f"[sim] offline simulator active — no robot, no network ({target})")
    script = os.path.join(_SCRIPTS, TARGETS[target])
    sys.argv = [TARGETS[target], *rest]   # so play/teleop argparse sees the right argv
    runpy.run_path(script, run_name="__main__")
