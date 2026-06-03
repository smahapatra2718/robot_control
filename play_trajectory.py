"""Headless replay of a saved teach trajectory on the UR15 or GoFa.

Reads trajectories/<name>.json, auto-detects the robot from its "robot" field,
and replays the stored joint waypoints on the real arm (after a confirm prompt)
-- no viser. IK-solver-free: every waypoint must already carry "q" (from Capture
or Plan-and-save in viser). The GoFa path imports pyroki for forward kinematics
only, to enforce the MAX_TCP_SPEED collaborative cap.

  ./robot_control/bin/python play_trajectory.py <name> [--speed S] [--dry-run] [--no-confirm]
"""

import argparse
import json
import os
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ_DIR = os.path.join(_HERE, "trajectories")

# ---- profile (mirrors the teleop scripts) ----
RAMP_FRAC = 0.25
MIN_SEG_DURATION_S = 0.5
DWELL_S = 0.2
GRIP_PREDELAY_S = 0.5

# ---- UR15 ----
UR_ROBOT_IP = "192.168.125.2"
UR_MAX_JOINT_SPEED = 1.0
UR_STREAM_HZ = 50
UR_SERVO_LOOKAHEAD = 0.1
UR_SERVO_GAIN = 300
UR_SERVO_STOP_DECEL = 2.0
UR_SETTLE_GAIN = 600
UR_SETTLE_EPS_RAD = 0.00002
UR_SETTLE_STALL_TICKS = 10
UR_SETTLE_MAX_S = 3.0
UR_GRIPPER_FINGER_OPEN = 0.025
UR_GRIPPER_MASS = 1.0
UR_GRIPPER_COG = (0.0, 0.0, 0.06)

# ---- GoFa ----
GOFA_ROBOT_IP = "192.168.125.1"
GOFA_RWS_USER = "Default User"
GOFA_RWS_PASSWORD = "robotics"
GOFA_RAPID_MODULE = "PyEgm"
GOFA_RAPID_GO_FLAG = "egm_go"
GOFA_EGM_LOCAL_PORT = 6510
GOFA_MAX_JOINT_SPEED = 1.0
GOFA_MAX_TCP_SPEED = 0.25
GOFA_STREAM_HZ = 100
GOFA_HOLD_AFTER_PLAY_S = 1.5
GOFA_URDF_PATH = os.path.join(_HERE, "crb15000_5_95.urdf")
GOFA_MESH_DIR_PREFIX = os.path.join(_HERE, "abb_desc")


def alpha_to_s(alpha: float, r: float = RAMP_FRAC) -> float:
    """Trapezoidal velocity profile: alpha in [0,1] -> traversed fraction in [0,1]."""
    v_peak = 1.0 / (1.0 - r)
    if alpha < r:
        return 0.5 * v_peak * alpha * alpha / r
    if alpha < 1.0 - r:
        return 0.5 * v_peak * r + v_peak * (alpha - r)
    return 1.0 - 0.5 * v_peak * (1.0 - alpha) ** 2 / r


def load_trajectory(name: str) -> dict:
    path = os.path.join(TRAJ_DIR, f"{name}.json")
    with open(path) as f:
        data = json.load(f)
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
        segments.append((q.copy(), q_next, wp.get("grip")))
        q = q_next
    return segments


def estimate_duration(segments, max_joint_speed: float, speed: float) -> float:
    total = 0.0
    for q_start, q_goal, grip in segments:
        delta = q_goal - q_start
        seg = max(MIN_SEG_DURATION_S, float(np.max(np.abs(delta))) / max_joint_speed)
        total += seg / max(0.1, speed) + DWELL_S
        if grip in ("open", "close"):
            total += GRIP_PREDELAY_S
    return total


def print_plan(robot: str, segments, speed: float) -> None:
    mjs = UR_MAX_JOINT_SPEED if robot == "ur15" else GOFA_MAX_JOINT_SPEED
    print(f"Robot: {robot}")
    print(f"Segments: {len(segments)} (first = current pose -> waypoint 1)")
    for i, (a, b, grip) in enumerate(segments):
        dmax = float(np.max(np.abs(b - a)))
        tag = f"  grip->{grip}" if grip in ("open", "close") else ""
        print(f"  seg {i + 1}: max|dq|={np.degrees(dmax):6.1f} deg{tag}")
    print(f"Estimated duration: {estimate_duration(segments, mjs, speed):.1f} s "
          f"(speed={speed})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Headless trajectory replay (UR15 / GoFa).")
    ap.add_argument("name", help="trajectory name (trajectories/<name>.json)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed scale (default 1.0)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit, no motion")
    ap.add_argument("--no-confirm", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    data = load_trajectory(args.name)
    robot = data.get("robot")
    if robot not in ("ur15", "gofa"):
        raise SystemExit(f"unknown robot {robot!r} in {args.name}.json")

    if args.dry_run:
        # current pose unknown without a robot connection: show segments between
        # the stored waypoints (waypoint1->2->...), which is the bulk of the plan.
        wps = data["waypoints"]
        segs = build_segments(np.asarray(wps[0]["q"], dtype=np.float64), wps[1:])
        print("[dry-run] (move-to-start segment omitted; needs a live robot pose)")
        print_plan(robot, segs, args.speed)
        return

    if robot == "ur15":
        play_ur15(data, args.speed, args.no_confirm)
    else:
        play_gofa(data, args.speed, args.no_confirm)


def play_ur15(data, speed, no_confirm):
    import hande_gripper
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    rtde_r = RTDEReceiveInterface(UR_ROBOT_IP)
    rtde_c = RTDEControlInterface(UR_ROBOT_IP)
    try:
        rtde_c.setPayload(UR_GRIPPER_MASS, list(UR_GRIPPER_COG))
    except Exception as e:
        print(f"setPayload failed ({e}); end-of-play droop may be larger.")

    # gripper best-effort + known-open startup state (mirrors teleop_ur15.py)
    try:
        gripper = hande_gripper.HandEGripper(UR_ROBOT_IP, hande_gripper.DEFAULT_PORT)
        gripper.connect()
        gripper.activate()
        gripper.open()
        print("Hand-E connected + opened.")
    except Exception as e:
        gripper = None
        print(f"Hand-E unavailable ({e}); motion-only (grip actions skipped).")
    cur_grip = "open"

    q_now = np.array(rtde_r.getActualQ(), dtype=np.float64)
    segments = build_segments(q_now, data["waypoints"])
    print_plan("ur15", segments, speed)
    if not no_confirm and input("Execute on the real UR15? [y/N] ").strip().lower() != "y":
        print("Aborted."); return

    dt = 1.0 / UR_STREAM_HZ
    try:
        for seg_idx, (q_start, q_goal, grip) in enumerate(segments):
            delta = q_goal - q_start
            seg_duration = max(MIN_SEG_DURATION_S, float(np.max(np.abs(delta))) / UR_MAX_JOINT_SPEED)
            print(f"Segment {seg_idx + 1}/{len(segments)}")
            alpha = 0.0
            while alpha < 1.0:
                q = q_start + delta * alpha_to_s(alpha)
                rtde_c.servoJ(q.tolist(), 0.0, 0.0, dt, UR_SERVO_LOOKAHEAD, UR_SERVO_GAIN)
                time.sleep(dt)
                alpha = min(1.0, alpha + dt * speed / seg_duration)

            if grip in ("open", "close") and grip != cur_grip:
                for _ in range(int(GRIP_PREDELAY_S * UR_STREAM_HZ)):  # settle before actuating
                    rtde_c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, UR_SERVO_LOOKAHEAD, UR_SERVO_GAIN)
                    time.sleep(dt)
                if gripper is not None:
                    print(f"Gripper: {grip}")
                    getattr(gripper, "open" if grip == "open" else "close_gripper")()
                    time.sleep(0.8)  # let the fingers move
                cur_grip = grip

            if seg_idx < len(segments) - 1:  # inter-waypoint dwell
                for _ in range(int(max(0.0, DWELL_S / max(0.1, speed)) * UR_STREAM_HZ)):
                    rtde_c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, UR_SERVO_LOOKAHEAD, UR_SERVO_GAIN)
                    time.sleep(dt)

        # final settle: hold the last target until the measured joints arrive
        q_final = segments[-1][1]
        deadline = time.monotonic() + UR_SETTLE_MAX_S
        best, stalls = float("inf"), 0
        while True:
            rtde_c.servoJ(q_final.tolist(), 0.0, 0.0, dt, UR_SERVO_LOOKAHEAD, UR_SETTLE_GAIN)
            time.sleep(dt)
            err = float(np.max(np.abs(np.array(rtde_r.getActualQ()) - q_final)))
            if err < best - UR_SETTLE_EPS_RAD:
                best, stalls = err, 0
            else:
                stalls += 1
            if stalls >= UR_SETTLE_STALL_TICKS or time.monotonic() > deadline:
                print(f"[settle] final joint error {np.degrees(err):.3f} deg")
                break
    finally:
        rtde_c.servoStop(UR_SERVO_STOP_DECEL)
        rtde_c.stopScript()
    print("Done.")


def play_gofa(data, speed, no_confirm):
    raise SystemExit("play_gofa not implemented yet")  # Task 6


if __name__ == "__main__":
    main()
