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
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "lib")):  # repo root + lib/ (our modules)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import robot_common as rc
from robot_common import (
    RAMP_FRAC, MIN_SEG_DURATION_S, DWELL_S, GRIP_PREDELAY_S, GRIP_EPS,
    alpha_to_s, norm_grip,
    UR_ROBOT_IP, UR_MAX_JOINT_SPEED, UR_STREAM_HZ, UR_SERVO_LOOKAHEAD, UR_SERVO_GAIN,
    UR_SERVO_STOP_DECEL, UR_SETTLE_GAIN, UR_SETTLE_EPS_RAD, UR_SETTLE_STALL_TICKS,
    UR_SETTLE_MAX_S, UR_GRIPPER_MASS, UR_GRIPPER_COG,
    GOFA_ROBOT_IP, GOFA_RWS_USER, GOFA_RWS_PASSWORD, GOFA_RAPID_MODULE,
    GOFA_RAPID_GO_FLAG, GOFA_EGM_LOCAL_PORT, GOFA_MAX_JOINT_SPEED, GOFA_MAX_TCP_SPEED,
    GOFA_STREAM_HZ, GOFA_HOLD_AFTER_PLAY_S, GOFA_URDF_PATH, GOFA_MESH_DIR_PREFIX,
)


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

    if robot == "ur15":
        play_ur15(combined, args.speed, args.no_confirm)
    else:
        play_gofa(combined, args.speed, args.no_confirm)


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

    # gripper best-effort: only fall back to motion-only if it's unreachable. If it
    # IS reachable, WAIT for activation calibration (it auto-references its full
    # open/close range) and the initial open to finish before any motion runs --
    # the trajectory's grip actions assume a calibrated gripper.
    try:
        gripper = hande_gripper.HandEGripper(UR_ROBOT_IP, hande_gripper.DEFAULT_PORT)
        gripper.connect()
    except Exception as e:
        gripper = None
        print(f"Hand-E unreachable ({e}); motion-only (grip actions skipped).")

    if gripper is not None:
        print("Resetting + activating Hand-E; waiting for full calibration...")
        try:
            # Force a clean ACT 0->1 cycle so we actually observe the auto-
            # calibration (STA 0->3), not a stale STA==3 from a prior session that
            # would let activate() return while the fingers are still referencing.
            gripper.reset(timeout=5.0)               # ACT=0 -> STA==0
            gripper.activate(timeout=20.0)           # ACT=1 -> STA==3 (calibration done)
            gripper.open()
            gripper.wait_until_idle(timeout=10.0)    # wait out the open move
            print("Hand-E calibrated + open.")
        except Exception as e:
            raise SystemExit(
                f"Hand-E activation/calibration failed: {e}. Power-cycle the gripper "
                f"or run verify_hande.py, then retry (or unplug it to run motion-only)."
            )
    cur_grip = 0.0   # fraction closed (0=open), matches the known-open startup state

    q_now = np.array(rtde_r.getActualQ(), dtype=np.float64)
    segments = build_segments(q_now, data["waypoints"])
    print_plan("ur15", segments, speed)
    if not no_confirm and not confirm("Execute on the real UR15? [y/N] "):
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

            if grip is not None and abs(grip - cur_grip) > GRIP_EPS:
                for _ in range(int(GRIP_PREDELAY_S * UR_STREAM_HZ)):  # settle before actuating
                    rtde_c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, UR_SERVO_LOOKAHEAD, UR_SERVO_GAIN)
                    time.sleep(dt)
                if gripper is not None:
                    print(f"Gripper: {int(round(grip * 100))}% closed")
                    gripper.move(grip)
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
    import jax.numpy as jnp
    import jaxlie
    import pyroki as pk
    import yourdfpy

    import abb_egm
    import abb_rws

    urdf = yourdfpy.URDF.load(GOFA_URDF_PATH, filename_handler=rc.make_mesh_resolver(GOFA_MESH_DIR_PREFIX))
    robot = pk.Robot.from_urdf(urdf)
    tcp_idx = robot.links.names.index("tool0")

    def tcp_xyz(q):
        Ts = robot.forward_kinematics(cfg=jnp.array(q))
        return np.asarray(jaxlie.SE3(Ts[tcp_idx]).translation())

    def cap_seg_duration(q_start, delta, seg_duration, dt):
        """Stretch seg_duration so peak TCP speed stays <= GOFA_MAX_TCP_SPEED."""
        alpha, prev_p, peak = 0.0, tcp_xyz(q_start), 0.0
        while alpha < 1.0:
            alpha = min(1.0, alpha + dt / seg_duration)
            p = tcp_xyz(q_start + delta * alpha_to_s(alpha))
            peak = max(peak, float(np.linalg.norm(p - prev_p)) / dt)
            prev_p = p
        if peak > GOFA_MAX_TCP_SPEED:
            seg_duration *= peak / GOFA_MAX_TCP_SPEED
        return seg_duration

    rws = abb_rws.RWSClient(host=GOFA_ROBOT_IP, user=GOFA_RWS_USER, password=GOFA_RWS_PASSWORD)
    rws.request_mastership()
    rws.set_rapid_bool(GOFA_RAPID_GO_FLAG, False, module=GOFA_RAPID_MODULE)
    egm = abb_egm.EGMSession(local_port=GOFA_EGM_LOCAL_PORT)
    egm.start()

    q_now = np.array(rws.get_joints(), dtype=np.float64)
    segments = build_segments(q_now, data["waypoints"])
    print_plan("gofa", segments, speed)
    if not no_confirm and input("Execute on the real GoFa? [y/N] ").strip().lower() != "y":
        print("Aborted."); return

    dt = 1.0 / GOFA_STREAM_HZ

    def start_egm():
        egm.set_target_rad(q_now.tolist())
        rws.set_rapid_bool(GOFA_RAPID_GO_FLAG, True, module=GOFA_RAPID_MODULE)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if egm.is_fresh(max_age_s=0.1):
                return True
            time.sleep(0.05)
        return False

    if not start_egm():
        print("EGM did not start (no packets in 3s). Is PyEgm parked at WaitUntil egm_go?")
        return
    try:
        for seg_idx, (q_start, q_goal, _grip) in enumerate(segments):
            delta = q_goal - q_start
            seg_duration = max(MIN_SEG_DURATION_S, float(np.max(np.abs(delta))) / GOFA_MAX_JOINT_SPEED)
            seg_duration = cap_seg_duration(q_start, delta, seg_duration, dt)
            print(f"Segment {seg_idx + 1}/{len(segments)}")
            alpha = 0.0
            while alpha < 1.0:
                q = q_start + delta * alpha_to_s(alpha)
                egm.set_target_rad(q.tolist())
                time.sleep(dt)
                alpha = min(1.0, alpha + dt * speed / seg_duration)
            if seg_idx < len(segments) - 1:
                for _ in range(int(max(0.0, DWELL_S / max(0.1, speed)) * GOFA_STREAM_HZ)):
                    egm.set_target_rad(q_goal.tolist())
                    time.sleep(dt)

        hold = segments[-1][1]
        for _ in range(int(GOFA_HOLD_AFTER_PLAY_S * GOFA_STREAM_HZ)):  # let \CondTime fire
            egm.set_target_rad(hold.tolist())
            time.sleep(dt)
    finally:
        try:
            rws.set_rapid_bool(GOFA_RAPID_GO_FLAG, False, module=GOFA_RAPID_MODULE)
        except Exception:
            pass
    print("Done.")


if __name__ == "__main__":
    main()
