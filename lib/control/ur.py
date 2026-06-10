"""URController — UR15 over RTDE servoJ + Hand-E gripper, behind RobotController."""
from __future__ import annotations

import time

import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description

import hande_gripper
import pyroki_snippets as pks
import robot_common as rc

from .base import RobotController

_UR_SAFETY_MODES = {
    1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY",
    5: "SAFEGUARD_STOP", 6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP",
    8: "VIOLATION", 9: "FAULT", 10: "VALIDATE_JOINT_ID", 11: "UNDEFINED",
}


class URController(RobotController):
    robot_name = "ur15"
    POLL_HZ = 30.0

    def __init__(self) -> None:
        super().__init__()
        self._urdf = load_robot_description(rc.UR_ROBOT_DESCRIPTION)
        self._robot = pk.Robot.from_urdf(self._urdf)
        self._tcp = self._robot.links.names.index(rc.TARGET_LINK)
        g = yourdfpy.URDF.load(rc.UR_GRIPPER_URDF_PATH,
                               filename_handler=rc.make_mesh_resolver(rc.UR_MESH_DIR_PREFIX))
        g.update_cfg(np.array([rc.UR_GRIPPER_FINGER_OPEN]))
        self._tool0_T_grasp = jaxlie.SE3.from_matrix(
            jnp.asarray(g.get_transform(rc.UR_GRASP_LINK, rc.TARGET_LINK)))
        self._c = None
        self._r = None
        self._gripper = None
        self._grip_frac = 0.0

    # ---- lifecycle ----
    def _connect(self) -> None:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
        self._r = RTDEReceiveInterface(rc.UR_ROBOT_IP)
        self._c = RTDEControlInterface(rc.UR_ROBOT_IP)
        try:
            self._c.setPayload(rc.UR_GRIPPER_MASS, list(rc.UR_GRIPPER_COG))
        except Exception as e:
            print(f"setPayload failed ({e}).")
        try:
            self._gripper = hande_gripper.HandEGripper(rc.UR_ROBOT_IP, hande_gripper.DEFAULT_PORT)
            self._gripper.connect()
            self._gripper.activate()
            self._gripper.open()
            self._grip_frac = 0.0
        except Exception as e:
            self._gripper = None
            print(f"Hand-E unavailable ({e}); viz/move only.")

    def _close(self) -> None:
        for fn in (lambda: self._c.servoStop(rc.UR_SERVO_STOP_DECEL),
                   self._c.stopScript, self._c.disconnect, self._r.disconnect,
                   (self._gripper.close if self._gripper is not None else (lambda: None))):
            try:
                fn()
            except Exception:
                pass

    # ---- reads ----
    def _read_q(self):
        return np.asarray(self._r.getActualQ(), dtype=float)

    def _read_safety(self):
        try:
            mode = self._r.getSafetyMode()
            return _UR_SAFETY_MODES.get(mode, f"mode {mode}"), str(mode), True, {}
        except Exception:
            return "UNKNOWN", "?", False, {}

    def _fk_pose(self, q):
        Ts = self._robot.forward_kinematics(cfg=jnp.array(q))
        T = jaxlie.SE3(Ts[self._tcp]).multiply(self._tool0_T_grasp)
        return np.asarray(T.translation()), np.asarray(T.rotation().wxyz)

    def _ik(self, pos, wxyz, q_seed):
        # gizmo/waypoint targets are at the grasp point; map back to a tool0 target.
        T_grasp = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(jnp.asarray(wxyz)), jnp.asarray(pos))
        T_tool0 = T_grasp.multiply(self._tool0_T_grasp.inverse())
        return np.asarray(pks.solve_ik_seeded(
            robot=self._robot, target_link_name=rc.TARGET_LINK,
            target_position=np.asarray(T_tool0.translation()),
            target_wxyz=np.asarray(T_tool0.rotation().wxyz),
            q_seed=q_seed, rest_weight=2.0))

    def _gripper_frac(self):
        return self._grip_frac

    # ---- stops ----
    def _graceful_stop(self) -> None:
        try:
            self._c.servoStop(rc.UR_SERVO_STOP_DECEL)
        except Exception:
            pass

    def _hard_stop(self) -> None:
        try:
            self._c.triggerProtectiveStop()
        except Exception:
            pass

    # ---- motion (move/settle; gripper added in Task 4) ----
    def _run_play(self, segments, speed, progress_cb) -> None:
        dt = 1.0 / rc.UR_STREAM_HZ
        n = len(segments)
        for seg_idx, (q_start, q_goal, _grip) in enumerate(segments):
            if self._cmd_stop.is_set():
                break
            delta = q_goal - q_start
            seg_dur = max(rc.MIN_SEG_DURATION_S, float(np.max(np.abs(delta))) / rc.UR_MAX_JOINT_SPEED)
            alpha = 0.0
            while alpha < 1.0:
                if self._cmd_stop.is_set():
                    break
                q = q_start + delta * rc.alpha_to_s(alpha)
                self._c.servoJ(q.tolist(), 0.0, 0.0, dt, rc.UR_SERVO_LOOKAHEAD, rc.UR_SERVO_GAIN)
                time.sleep(dt)
                alpha = min(1.0, alpha + dt * speed / seg_dur)
            if not self._cmd_stop.is_set() and seg_idx < n - 1:
                hold = max(0.0, rc.DWELL_S / max(0.1, speed))
                for _ in range(int(hold * rc.UR_STREAM_HZ)):
                    if self._cmd_stop.is_set():
                        break
                    self._c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, rc.UR_SERVO_LOOKAHEAD, rc.UR_SERVO_GAIN)
                    time.sleep(dt)
            progress_cb((seg_idx + 1) / n)
        # final settle: hold the last target until measured joints arrive (lifted
        # from teleop_ur15.py:608-629 / play_trajectory.py:222-236).
        if not self._cmd_stop.is_set():
            q_final = segments[-1][1]
            deadline = time.monotonic() + rc.UR_SETTLE_MAX_S
            best, stalls = float("inf"), 0
            while not self._cmd_stop.is_set():
                self._c.servoJ(q_final.tolist(), 0.0, 0.0, dt, rc.UR_SERVO_LOOKAHEAD, rc.UR_SETTLE_GAIN)
                time.sleep(dt)
                err = float(np.max(np.abs(np.asarray(self._r.getActualQ()) - q_final)))
                if err < best - rc.UR_SETTLE_EPS_RAD:
                    best, stalls = err, 0
                else:
                    stalls += 1
                if stalls >= rc.UR_SETTLE_STALL_TICKS or time.monotonic() > deadline:
                    break
        try:
            self._c.servoStop(rc.UR_SERVO_STOP_DECEL)
        except Exception:
            pass
