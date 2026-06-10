"""Headless replay of a saved teach trajectory on the UR15 or GoFa.

Reads trajectories/<name>.json, auto-detects the robot from its "robot" field,
and replays the stored joint waypoints on the real arm (after a confirm prompt)
-- no viser. IK-solver-free: every waypoint must already carry "q" (from Capture
or Plan-and-save in viser). The GoFa path imports pyroki for forward kinematics
only, to enforce the MAX_TCP_SPEED collaborative cap.

Pass several names to chain them: they play back-to-back as one continuous
motion (each trajectory's last waypoint -> the next's first becomes a normal
move segment), the Hand-E is calibrated ONCE at the start (not between
trajectories), and the gripper state carries across the seam. All chained
trajectories must target the same robot.

  ./robot_control/bin/python scripts/play_trajectory.py <name> [more names...] [--speed S] [--dry-run] [--no-confirm]
"""

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "lib")):  # repo root + lib/ (our modules)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import robot_common as rc
from robot_common import (
    MIN_SEG_DURATION_S, DWELL_S, GRIP_PREDELAY_S,
    norm_grip,
    UR_MAX_JOINT_SPEED,
    GOFA_MAX_JOINT_SPEED,
)

from control import make_controller


def confirm(prompt: str) -> bool:
    """Ask a y/N question, but first discard any type-ahead so a 'y' pressed
    DURING the (slow) gripper calibration can't auto-answer the prompt — you must
    press y after the prompt actually appears."""
    try:
        import sys
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass   # non-tty / non-POSIX: nothing buffered to flush
    return input(prompt).strip().lower() == "y"


def load_trajectory(name: str) -> dict:
    data = rc.load_trajectory(name)
    wps = data.get("waypoints", [])
    if not wps:
        raise SystemExit(f"{name}.json has no waypoints.")
    for i, wp in enumerate(wps):
        if wp.get("q") is None:
            raise SystemExit(
                f"waypoint {i} in {name}.json has no joints -- open '{name}' in viser, "
                f"Plan, and re-save (the CLI replays stored joints, it does not run IK)."
            )
    return data


def build_segments(q_start: np.ndarray, waypoints: list[dict]):
    """First segment goes from the robot's current joints to waypoint 1, then
    waypoint-to-waypoint. Each segment is (q_start, q_goal, grip)."""
    segments = []
    q = np.asarray(q_start, dtype=np.float64)
    for wp in waypoints:
        q_next = np.asarray(wp["q"], dtype=np.float64)
        segments.append((q.copy(), q_next, norm_grip(wp.get("grip"))))
        q = q_next
    return segments


def estimate_duration(segments, max_joint_speed: float, speed: float) -> float:
    total = 0.0
    for q_start, q_goal, grip in segments:
        delta = q_goal - q_start
        seg = max(MIN_SEG_DURATION_S, float(np.max(np.abs(delta))) / max_joint_speed)
        total += seg / max(0.1, speed) + DWELL_S
        if grip is not None:
            total += GRIP_PREDELAY_S
    return total


def print_plan(robot: str, segments, speed: float) -> None:
    mjs = UR_MAX_JOINT_SPEED if robot == "ur15" else GOFA_MAX_JOINT_SPEED
    print(f"Robot: {robot}")
    print(f"Segments: {len(segments)} (first = current pose -> waypoint 1)")
    for i, (a, b, grip) in enumerate(segments):
        dmax = float(np.max(np.abs(b - a)))
        tag = f"  grip->{int(round(grip * 100))}% closed" if grip is not None else ""
        print(f"  seg {i + 1}: max|dq|={np.degrees(dmax):6.1f} deg{tag}")
    print(f"Estimated duration: {estimate_duration(segments, mjs, speed):.1f} s "
          f"(speed={speed})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Headless trajectory replay (UR15 / GoFa).")
    ap.add_argument("name", nargs="+",
                    help="trajectory name(s) (trajectories/<name>.json); multiple play in order")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed scale (default 1.0)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit, no motion")
    ap.add_argument("--no-confirm", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    datas = [load_trajectory(n) for n in args.name]
    robots = {d.get("robot") for d in datas}
    if len(robots) > 1:
        raise SystemExit(f"cannot chain different robots in one run: {sorted(robots)}")
    robot = datas[0].get("robot")
    if robot not in ("ur15", "gofa"):
        raise SystemExit(f"unknown robot {robot!r} in {args.name[0]}.json")

    # Concatenate every trajectory's waypoints into one continuous list. The
    # players already calibrate the gripper once and replay a single waypoint
    # list, so chaining = one calibration, one settle, and each seam (prev last
    # waypoint -> next first) is just another move segment.
    all_wps = [wp for d in datas for wp in d["waypoints"]]
    combined = {"robot": robot, "waypoints": all_wps}
    if len(args.name) > 1:
        chain = ", ".join(f"{n} ({len(d['waypoints'])} wp)" for n, d in zip(args.name, datas))
        print(f"Chaining {len(args.name)} trajectories: {chain}")
        print("  -> one continuous motion; Hand-E calibrated once at the start.")

    if args.dry_run:
        # current pose unknown without a robot connection: show segments between
        # the stored waypoints (waypoint1->2->...), which is the bulk of the plan.
        segs = build_segments(np.asarray(all_wps[0]["q"], dtype=np.float64), all_wps[1:])
        print("[dry-run] (move-to-start segment omitted; needs a live robot pose)")
        print_plan(robot, segs, args.speed)
        return

    play_on_controller(robot, combined, args.speed, args.no_confirm)


def play_on_controller(robot: str, data, speed, no_confirm):
    c = make_controller(robot)
    print(f"Connecting to {robot} ...")
    c.connect()
    try:
        # display-only plan; the controller builds its own segments from a live q read
        segments = build_segments(c.get_state().q, data["waypoints"])
        print_plan(robot, segments, speed)
        if not no_confirm and not confirm(f"Execute on the real {robot}? [y/N] "):
            print("Aborted."); return
        cid = c.play(data["waypoints"], speed=speed)
        status = c.wait(cid, timeout=600.0)
        st = c.command_status(cid)
        if status != "done":
            print(f"Play ended: {status}" + (f" ({st['error']})" if st and st.get("error") else ""))
        else:
            print("Done.")
    finally:
        c.close()


if __name__ == "__main__":
    main()
