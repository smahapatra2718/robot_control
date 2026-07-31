"""Offline smoke test for the RobotController core (lib/control), driven against
the sim fakes (lib/robot_sim). No robot, no network.

  uv run scripts/control_smoketest.py

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
from control.state import COMMAND_KEYS, RobotState, empty_command  # noqa: E402


def test_state_dataclass():
    s = RobotState(
        ts=1.0, robot="ur15", q=[0.0] * 6,
        pose={"pos": [0.1, 0.2, 0.3], "wxyz": [1.0, 0.0, 0.0, 0.0]},
        gripper_frac=0.0, safety_state="NORMAL", controller_state="ok",
        activity="idle", active_command=empty_command(), conn_ok=True,
    )
    d = s.to_dict()
    assert d["robot"] == "ur15"
    assert d["q"] == [0.0] * 6
    assert d["pose"]["pos"] == [0.1, 0.2, 0.3]
    assert d["pose"]["wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert d["ts"] == 1.0
    assert d["gripper_frac"] == 0.0
    assert d["conn_ok"] is True
    # sub-keys always present, all None when nothing is running (never a bare null)
    assert d["active_command"] == {k: None for k in COMMAND_KEYS}
    assert d["health"] == {}
    print("PASS test_state_dataclass")


def test_ur_connect_state():
    robot_sim.install("ur15")
    from control import make_controller
    c = make_controller("ur15")
    c.connect()
    try:
        st = c.get_state()
        assert st.robot == "ur15"
        assert st.q == robot_sim.UR_HOME, "state q should be the seeded UR home"
        assert st.safety_state == "NORMAL"
        assert st.gripper_frac == 0.0
        assert len(st.pose["pos"]) == 3 and len(st.pose["wxyz"]) == 4
        assert st.activity == "idle"
    finally:
        c.close()
    print("PASS test_ur_connect_state")


def test_ur_move():
    robot_sim.install("ur15")
    from control import make_controller
    c = make_controller("ur15")
    c.connect()
    try:
        target = [0.0, -1.4, 1.4, -1.4, -1.4, 0.2]
        cid = c.move_to_joints(target, speed=5.0)
        assert c.wait(cid, timeout=20.0) == "done", "move did not complete"
        st = c.get_state()
        assert max(abs(a - b) for a, b in zip(st.q, target)) < 1e-6, "did not reach target"
        cid2 = c.move_to_joints(robot_sim.UR_HOME, speed=5.0)
        assert c.wait(cid2, timeout=20.0) == "done"
    finally:
        c.close()
    print("PASS test_ur_move")


def test_ur_estop_safety_state():
    """estop() must be visible in safety_state, not just activity. The sim used to hardcode
    getSafetyMode()->NORMAL, so an e-stop changed nothing and this was untestable offline."""
    robot_sim.install("ur15")
    from control import make_controller
    c = make_controller("ur15")
    c.connect()
    try:
        assert c.get_state().safety_state == "NORMAL", c.get_state().safety_state
        c.estop()
        for _ in range(100):                      # poll thread runs at 30 Hz
            st = c.get_state()
            if st.safety_state != "NORMAL":
                break
            time.sleep(0.02)
        # /estop is triggerProtectiveStop(), so this is PROTECTIVE_STOP — the emergency-stop
        # modes (6/7) come only from the physical circuit and can't be asserted in software.
        assert st.safety_state == "PROTECTIVE_STOP", st.safety_state
        assert st.controller_state == "3", st.controller_state
        assert st.activity == "stopped", st.activity
        # and it does not clear itself — a real protective stop is released at the pendant
        time.sleep(0.2)
        assert c.get_state().safety_state == "PROTECTIVE_STOP"
    finally:
        c.close()
    print("PASS test_ur_estop_safety_state")


def test_ur_play_gripper():
    robot_sim.install("ur15")
    from control import make_controller
    c = make_controller("ur15")
    c.connect()
    try:
        cid = c.play("_sample_ur15", speed=5.0)
        assert c.wait(cid, timeout=40.0) == "done", "play did not complete"
        import robot_common as rc
        wps = rc.load_trajectory("_sample_ur15", "ur15")["waypoints"]
        q_final = wps[-1]["q"]
        st = c.get_state()
        assert max(abs(a - b) for a, b in zip(st.q, q_final)) < 1e-6, "play did not reach final waypoint"
        # the play's gripper-on-change should leave the tracked grip at the final waypoint's grip
        expected_grip = rc.norm_grip(wps[-1].get("grip"))
        if expected_grip is not None:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and abs(c.get_state().gripper_frac - expected_grip) > 1e-9:
                time.sleep(0.02)
            assert abs(c.get_state().gripper_frac - expected_grip) < 1e-9, "play did not leave gripper at final grip"
        gid = c.set_gripper(0.5)
        assert c.wait(gid, timeout=10.0) == "done"
        # state is eventually-consistent at POLL_HZ — let the poll thread catch up
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and abs(c.get_state().gripper_frac - 0.5) > 1e-9:
            time.sleep(0.02)
        assert abs(c.get_state().gripper_frac - 0.5) < 1e-9, "set_gripper did not update state"
    finally:
        c.close()
    print("PASS test_ur_play_gripper")


def test_gofa_connect_state():
    robot_sim.install("gofa")
    from control import make_controller
    c = make_controller("gofa")
    c.connect()
    try:
        st = c.get_state()
        assert st.robot == "gofa"
        assert st.q == robot_sim.GOFA_HOME, "state q should be the seeded GoFa home"
        assert st.gripper_frac is None, "GoFa has no gripper"
        assert len(st.pose["pos"]) == 3 and len(st.pose["wxyz"]) == 4
    finally:
        c.close()
    print("PASS test_gofa_connect_state")


def test_gofa_move_play():
    robot_sim.install("gofa")
    from control import make_controller
    c = make_controller("gofa")
    c.connect()
    try:
        target = [0.0, 0.1, 0.0, 0.0, 1.5708, 0.0]
        cid = c.move_to_joints(target, speed=5.0)
        assert c.wait(cid, timeout=30.0) == "done", "gofa move did not complete"
        st = c.get_state()
        assert max(abs(a - b) for a, b in zip(st.q, target)) < 1e-6, "gofa did not reach target"
        pid = c.play("_sample_gofa", speed=5.0)
        assert c.wait(pid, timeout=40.0) == "done", "gofa play did not complete"
        import robot_common as rc
        q_final = rc.load_trajectory("_sample_gofa", "gofa")["waypoints"][-1]["q"]
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and \
                max(abs(a - b) for a, b in zip(c.get_state().q, q_final)) > 1e-6:
            time.sleep(0.02)
        assert max(abs(a - b) for a, b in zip(c.get_state().q, q_final)) < 1e-6, \
            "gofa play did not reach final waypoint"
    finally:
        c.close()
    print("PASS test_gofa_move_play")


def test_controller_freedrive_grasp():
    for robot in ("ur15", "gofa"):
        robot_sim.install(robot)
        from control import make_controller, Busy
        c = make_controller(robot)
        c.connect()
        try:
            pos, wxyz = c.grasp_pose(c.get_state().q)
            assert len(pos) == 3 and len(wxyz) == 4, f"{robot} grasp_pose shape"
            c.start_freedrive()   # must not raise (no-op-ish in sim)
            # free-drive is mutually exclusive with the command executor
            try:
                c.move_to_joints(c.get_state().q)
                raised = False
            except Busy:
                raised = True
            assert raised, f"{robot}: motion during free-drive must raise Busy"
            c.stop_freedrive()
            # after stopping free-drive, motion is accepted again
            cid = c.move_to_joints(c.get_state().q)
            assert c.wait(cid, timeout=20.0) == "done", f"{robot}: move after free-drive"
            g = c.adjust_grip(0.1)
            if robot == "ur15":
                assert g is not None and abs(g - 0.1) < 1e-9, "ur adjust_grip should step the grip"
            else:
                assert g is None, "gofa has no gripper"
        finally:
            c.close()
    print("PASS test_controller_freedrive_grasp")


def _max_line_deviation(points):
    """Max perpendicular distance from each point to the chord through endpoints (m)."""
    import numpy as np
    p = np.asarray(points, dtype=float)
    a, b = p[0], p[-1]
    ab = b - a
    L = float(np.linalg.norm(ab))
    if L < 1e-12:
        return float(np.max(np.linalg.norm(p - a, axis=1)))
    u = ab / L
    rel = p - a
    perp = rel - np.outer(rel @ u, u)
    return float(np.max(np.linalg.norm(perp, axis=1)))


def test_cartesian_path_straight():
    """The EE path for a segment must be a straight Cartesian line (MoveL), not the
    curved sweep joint-space interpolation produces (MoveJ). Asserts the new
    Cartesian interpolation is straight AND that the scenario genuinely arcs under
    the old joint-lerp (so the test can't pass trivially)."""
    import numpy as np
    robot_sim.install("ur15")
    from control import make_controller
    c = make_controller("ur15")
    c.connect()
    try:
        # rotate the base joint ~1.2 rad: the tool sweeps a wide horizontal arc under
        # joint-space interpolation, which is exactly what a straight move must avoid.
        q_start = np.asarray(robot_sim.UR_HOME, dtype=float)
        q_goal = q_start + np.asarray([1.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        ss = [i / 40.0 for i in range(41)]

        # old joint-space lerp — the curved path the user is complaining about
        joint_pts = [c._fk_pose(q_start + (q_goal - q_start) * s)[0] for s in ss]
        arc = _max_line_deviation(joint_pts)
        assert arc > 1e-2, f"scenario does not arc under joint-lerp ({arc*1000:.1f} mm); pick a better test pose"

        # new Cartesian interpolation — must be straight to within IK error
        at = c._cartesian_q(q_start, q_goal)
        cart_pts = [c._fk_pose(at(s))[0] for s in ss]
        dev = _max_line_deviation(cart_pts)
        assert dev < 2e-3, f"Cartesian path deviates {dev*1000:.2f} mm from straight line"
        # endpoints anchored exactly to the requested joint configs
        assert np.max(np.abs(at(0.0) - q_start)) < 1e-9, "s=0 must be q_start exactly"
        assert np.max(np.abs(at(1.0) - q_goal)) < 1e-9, "s=1 must be q_goal exactly"
        print(f"PASS test_cartesian_path_straight (arc {arc*1000:.0f} mm -> straight {dev*1000:.2f} mm)")
    finally:
        c.close()


def test_live_servo_straight():
    """The Live follower (bounded Cartesian step toward the gizmo + seeded IK + joint
    clamp, re-evaluated from the live pose each tick) must drive the tool along a
    straight line to a reachable target, not the joint-space arc it used to."""
    import numpy as np
    from control.base import step_pose_toward
    robot_sim.install("gofa")
    from control import make_controller
    c = make_controller("gofa")
    c.connect()
    try:
        q0 = np.asarray(robot_sim.GOFA_HOME, dtype=float)
        p0, w0 = (np.asarray(v, dtype=float) for v in c._fk_pose(q0))
        # reachable target = the tool pose of a nearby joint config, same orientation
        p_tgt = np.asarray(c._fk_pose(q0 + np.array([0.6, 0.2, -0.2, 0.0, 0.0, 0.0]))[0], dtype=float)
        dt = 1.0 / 30.0
        q_cmd, pts = q0.copy(), [p0]
        for _ in range(3000):
            p_cur, w_cur = (np.asarray(v, dtype=float) for v in c._fk_pose(q_cmd))
            p_ref, w_ref = step_pose_toward(p_cur, w_cur, p_tgt, w0, 0.25 * dt, 1.5 * dt)
            q_t = np.asarray(c._ik(p_ref, w_ref, q_cmd), dtype=float)
            q_cmd = q_cmd + np.clip(q_t - q_cmd, -1.0 * dt, 1.0 * dt)
            pts.append(np.asarray(c._fk_pose(q_cmd)[0], dtype=float))
            if np.linalg.norm(p_tgt - pts[-1]) < 1e-3:
                break
        reached = float(np.linalg.norm(p_tgt - pts[-1]))
        dev = _max_line_deviation(pts)
        assert reached < 5e-3, f"live servo stalled {reached*1000:.1f} mm short of target"
        assert dev < 3e-3, f"live servo path deviates {dev*1000:.2f} mm from straight line"
        print(f"PASS test_live_servo_straight (reached {reached*1000:.1f} mm, straight {dev*1000:.2f} mm)")
    finally:
        c.close()


def test_command_history():
    robot_sim.install("ur15")
    from control import make_controller
    c = make_controller("ur15")
    c.connect()
    try:
        cid1 = c.move_to_joints([0.0, -1.4, 1.4, -1.4, -1.4, 0.1], speed=5.0)
        assert c.wait(cid1, timeout=20.0) == "done"
        cid2 = c.move_to_joints(robot_sim.UR_HOME, speed=5.0)
        assert c.wait(cid2, timeout=20.0) == "done"
        # cid1 finished and was superseded by cid2 — its result must still be queryable
        st1 = c.command_status(cid1)
        assert st1 is not None and st1["status"] == "done", "command history not retained"
        assert c.wait(cid1, timeout=1.0) == "done", "wait() should resolve a retained finished command"
    finally:
        c.close()
    print("PASS test_command_history")


def main():
    test_state_dataclass()
    test_ur_connect_state()
    test_ur_move()
    test_ur_estop_safety_state()
    test_ur_play_gripper()
    test_gofa_connect_state()
    test_gofa_move_play()
    test_controller_freedrive_grasp()
    test_cartesian_path_straight()
    test_live_servo_straight()
    test_command_history()
    print("ALL CONTROL SMOKE TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
