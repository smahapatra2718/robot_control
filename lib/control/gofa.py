"""GoFaController — ABB GoFa over EGM (UDP) + RWS, behind RobotController."""
from __future__ import annotations

import time

import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy

import abb_egm
import abb_rws
import pyroki_snippets as pks
import robot_common as rc

from .base import RobotController, Unsupported


class GoFaController(RobotController):
    robot_name = "gofa"
    POLL_HZ = 10.0     # RWS polling rate (matches teleop_gofa_egm)

    def __init__(self) -> None:
        super().__init__()
        self._urdf = yourdfpy.URDF.load(
            rc.GOFA_URDF_PATH, filename_handler=rc.make_mesh_resolver(rc.GOFA_MESH_DIR_PREFIX))
        self._robot = pk.Robot.from_urdf(self._urdf)
        self._tcp = self._robot.links.names.index(rc.TARGET_LINK)
        self._rws = None
        self._egm = None

    # ---- lifecycle ----
    def _connect(self) -> None:
        self._rws = abb_rws.RWSClient(host=rc.GOFA_ROBOT_IP, user=rc.GOFA_RWS_USER,
                                      password=rc.GOFA_RWS_PASSWORD)
        try:
            self._rws.request_mastership()
        except Exception as e:
            print(f"WARNING: could not acquire mastership: {e}")
        for flag in (rc.GOFA_RAPID_GO_FLAG, rc.GOFA_RAPID_LEAD_FLAG):
            try:
                self._rws.set_rapid_bool(flag, False, module=rc.GOFA_RAPID_MODULE)
            except Exception:
                pass
        self._egm = abb_egm.EGMSession(local_port=rc.GOFA_EGM_LOCAL_PORT)
        self._egm.start()

    def _close(self) -> None:
        for fn in (lambda: self._rws.set_rapid_bool(rc.GOFA_RAPID_GO_FLAG, False, module=rc.GOFA_RAPID_MODULE),
                   self._egm.stop, self._rws.release_mastership):
            try:
                fn()
            except Exception:
                pass

    # ---- reads ----
    def _read_q(self):
        return np.asarray(self._rws.get_joints(), dtype=float)

    def _read_safety(self):
        # GoFa's RWS exposes a single controller-state signal; use it for both the
        # safety_state and controller_state fields.
        try:
            st = self._rws.get_controller_state()
            health = {"egm_rx": self._egm.packets_rx, "egm_tx": self._egm.packets_tx} if self._egm else {}
            return st, st, True, health
        except Exception:
            return "UNKNOWN", "?", False, {}

    def _fk_pose(self, q):
        Ts = self._robot.forward_kinematics(cfg=jnp.array(q))
        T = jaxlie.SE3(Ts[self._tcp])
        return np.asarray(T.translation()), np.asarray(T.rotation().wxyz)

    def _ik(self, pos, wxyz, q_seed):
        return np.asarray(pks.solve_ik_seeded(
            robot=self._robot, target_link_name=rc.TARGET_LINK,
            target_position=np.asarray(pos), target_wxyz=np.asarray(wxyz),
            q_seed=q_seed, rest_weight=2.0))

    def _gripper_frac(self):
        return None

    def _gripper_blocking(self, frac, progress_cb):
        raise Unsupported("GoFa has no gripper")

    # ---- freedrive ----
    def _start_freedrive(self) -> None:
        self._rws.set_rapid_bool(rc.GOFA_RAPID_GO_FLAG, False, module=rc.GOFA_RAPID_MODULE)
        self._rws.set_rapid_bool(rc.GOFA_RAPID_LEAD_FLAG, True, module=rc.GOFA_RAPID_MODULE)

    def _stop_freedrive(self) -> None:
        try:
            self._rws.set_rapid_bool(rc.GOFA_RAPID_LEAD_FLAG, False, module=rc.GOFA_RAPID_MODULE)
        except Exception:
            pass

    # ---- stops ----
    def _graceful_stop(self) -> None:
        try:
            self._rws.set_rapid_bool(rc.GOFA_RAPID_GO_FLAG, False, module=rc.GOFA_RAPID_MODULE)
        except Exception:
            pass

    def _hard_stop(self) -> None:
        for fn in (lambda: self._rws.set_rapid_bool(rc.GOFA_RAPID_GO_FLAG, False, module=rc.GOFA_RAPID_MODULE),
                   self._rws.stop_program):
            try:
                fn()
            except Exception:
                pass

    # ---- EGM session + motion ----
    def _start_egm(self) -> bool:
        q_now = self._read_q_copy()
        self._egm.set_target_rad(q_now.tolist())
        try:
            self._rws.set_rapid_bool(rc.GOFA_RAPID_GO_FLAG, True, module=rc.GOFA_RAPID_MODULE)
        except Exception:
            return False
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if self._egm.is_fresh(0.1):
                return True
            time.sleep(0.05)
        # timed out — clear egm_go so RAPID doesn't sit in EGMRunJoint chasing a dead stream
        try:
            self._rws.set_rapid_bool(rc.GOFA_RAPID_GO_FLAG, False, module=rc.GOFA_RAPID_MODULE)
        except Exception:
            pass
        return False

    def _cap_seg_duration(self, q_start, delta, seg_duration, dt):
        alpha, prev_p, peak = 0.0, self._fk_pose(q_start)[0], 0.0
        while alpha < 1.0:
            alpha = min(1.0, alpha + dt / seg_duration)
            p = self._fk_pose(q_start + delta * rc.alpha_to_s(alpha))[0]
            peak = max(peak, float(np.linalg.norm(p - prev_p)) / dt)
            prev_p = p
        if peak > rc.GOFA_MAX_TCP_SPEED:
            seg_duration *= peak / rc.GOFA_MAX_TCP_SPEED
        return seg_duration

    def _run_play(self, segments, speed, progress_cb) -> None:
        dt = 1.0 / rc.GOFA_STREAM_HZ
        if not self._start_egm():
            raise RuntimeError("EGM did not start (no packets in 3s)")
        n = len(segments)
        try:
            for seg_idx, (q_start, q_goal, _grip) in enumerate(segments):
                if self._cmd_stop.is_set():
                    break
                delta = q_goal - q_start
                seg_dur = max(rc.MIN_SEG_DURATION_S, float(np.max(np.abs(delta))) / rc.GOFA_MAX_JOINT_SPEED)
                seg_dur = self._cap_seg_duration(q_start, delta, seg_dur, dt)
                alpha = 0.0
                while alpha < 1.0:
                    if self._cmd_stop.is_set():
                        break
                    q = q_start + delta * rc.alpha_to_s(alpha)
                    self._egm.set_target_rad(q.tolist())
                    time.sleep(dt)
                    alpha = min(1.0, alpha + dt * speed / seg_dur)
                if not self._cmd_stop.is_set() and seg_idx < n - 1:
                    for _ in range(int(max(0.0, rc.DWELL_S / max(0.1, speed)) * rc.GOFA_STREAM_HZ)):
                        if self._cmd_stop.is_set():
                            break
                        self._egm.set_target_rad(q_goal.tolist())
                        time.sleep(dt)
                progress_cb((seg_idx + 1) / n)
            if not self._cmd_stop.is_set():
                hold = segments[-1][1]
                for _ in range(int(rc.GOFA_HOLD_AFTER_PLAY_S * rc.GOFA_STREAM_HZ)):
                    self._egm.set_target_rad(hold.tolist())
                    time.sleep(dt)
        finally:
            try:
                self._rws.set_rapid_bool(rc.GOFA_RAPID_GO_FLAG, False, module=rc.GOFA_RAPID_MODULE)
            except Exception:
                pass
