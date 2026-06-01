"""
viser + pyroki teleop for ABB GoFa CRB 15000 using EGM (Externally Guided
Motion) for streaming joint control.

Difference from teleop_gofa.py:
  - teleop_gofa.py    : MoveAbsJ commit-and-wait per segment. Speed slider is
                         viz-only. Real motion speed set by v_tcp in RAPID.
  - teleop_gofa_egm.py: EGM joint streaming at STREAM_HZ. The same q computed
                         from the trapezoidal alpha goes to BOTH viser AND the
                         EGM target stream every tick — viz and robot move in
                         lockstep, just like teleop_ur15.py with servoJ.

Prerequisite: install_gofa_egm.py has been run successfully (PyEgm.mod loaded,
EGM_COMM.cfg + EGM_MOC.cfg loaded, controller rebooted, PP-to-Main + Play
done so the RAPID supervisor is parked at WaitUntil egm_go).

Run:
  ./robot_control/bin/python teleop_gofa_egm.py
"""

import os
import sys
import threading
import time

import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import viser
import yourdfpy
from viser.extras import ViserUrdf

import abb_egm
import abb_rws

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyroki_snippets as pks  # noqa: E402

# ---------- config ----------
ROBOT_IP = "192.168.125.1"
RWS_USER = "Default User"
RWS_PASSWORD = "robotics"
URDF_PATH = "crb15000_12_127.urdf"
URDF_MESH_DIR_PREFIX = "abb_desc"
TARGET_LINK = "tool0"

RAPID_TASK = "T_ROB1"
RAPID_GO_FLAG_VAR = "egm_go"        # bool in PyEgm.mod
RAPID_MODULE = "PyEgm"

EGM_LOCAL_PORT = 6510                # must match RemotePortNumber in EGM_COMM.cfg

POLL_HZ = 10                         # RWS state polling (idle viz only)
PLAY_HZ = 60                         # viz refresh when idle
STREAM_HZ = 100                      # EGM target stream rate (controller side runs ~250Hz)
MAX_JOINT_SPEED = 1.0                # rad/s peak per joint at slider=1.0
MIN_SEG_DURATION_S = 0.5
DWELL_S = 0.2
RAMP_FRAC = 0.25
HOLD_AFTER_PLAY_S = 1.5              # hold final target this long so RAPID's CondTime triggers


def _alpha_to_s(alpha: float, r: float = RAMP_FRAC) -> float:
    """Trapezoidal velocity profile: alpha in [0,1] -> traversed fraction in [0,1]."""
    v_peak = 1.0 / (1.0 - r)
    if alpha < r:
        return 0.5 * v_peak * alpha * alpha / r
    if alpha < 1.0 - r:
        return 0.5 * v_peak * r + v_peak * (alpha - r)
    return 1.0 - 0.5 * v_peak * (1.0 - alpha) ** 2 / r


# ---------- robot model ----------
def _resolve_mesh(fname: str) -> str:
    if fname.startswith("package://"):
        pkg, rest = fname[len("package://") :].split("/", 1)
        return os.path.join(URDF_MESH_DIR_PREFIX, pkg, rest)
    return fname


urdf = yourdfpy.URDF.load(URDF_PATH, filename_handler=_resolve_mesh)
robot = pk.Robot.from_urdf(urdf)
TARGET_LINK_IDX = robot.links.names.index(TARGET_LINK)
NUM_JOINTS = robot.joints.num_actuated_joints

# ---------- shared state ----------
state_lock = threading.Lock()
current_q = np.zeros(NUM_JOINTS)
last_poll_ok = False
waypoints: list[tuple[np.ndarray, np.ndarray]] = []
waypoint_frames: list = []
plan_segments: list[tuple[np.ndarray, np.ndarray]] | None = None
playing = threading.Event()
stop_flag = threading.Event()

# ---------- RWS (for state + RAPID flag control) ----------
rws = abb_rws.RWSClient(host=ROBOT_IP, user=RWS_USER, password=RWS_PASSWORD)

try:
    rws.request_mastership()
    print("RAPID mastership acquired.")
except Exception as e:
    print(f"WARNING: could not acquire RAPID mastership: {e}")
    print("  Execute-on-robot will fail. Resolve by releasing mastership from")
    print("  any other RWS client or rebooting the controller.")

# Safety: ensure egm_go is FALSE at startup so a stray TRUE doesn't fire EGM.
try:
    rws.set_rapid_bool("egm_go", False, module=RAPID_MODULE)
    print("Safety init: egm_go = FALSE.")
except Exception as e:
    print(f"Safety init failed: {e}")

# ---------- EGM (always listening; only active when RAPID enters EGMRunJoint) ----------
egm = abb_egm.EGMSession(local_port=EGM_LOCAL_PORT)
egm.start()
print(f"EGM listening on UDP {EGM_LOCAL_PORT}.")


def poll_loop() -> None:
    """RWS state polling at POLL_HZ. Used for idle visualization only — during
    an Execute Play, the play loop drives the URDF directly from the streamed
    target, so this loop's writes are skipped via `playing.is_set()` in viz_loop."""
    global current_q, last_poll_ok
    period = 1.0 / POLL_HZ
    while True:
        try:
            q = np.array(rws.get_joints(), dtype=np.float64)
            with state_lock:
                current_q = q
                last_poll_ok = True
        except Exception:
            with state_lock:
                last_poll_ok = False
        time.sleep(period)


threading.Thread(target=poll_loop, daemon=True).start()
time.sleep(0.4)


def ee_pose(q: np.ndarray) -> jaxlie.SE3:
    Ts = robot.forward_kinematics(cfg=jnp.array(q))
    return jaxlie.SE3(Ts[TARGET_LINK_IDX])


# ---------- viser ----------
server = viser.ViserServer()
server.scene.add_grid("/ground", width=2, height=2)
viser_urdf = ViserUrdf(server, urdf, root_node_name="/base")

with state_lock:
    q0 = current_q.copy()
T_ee0 = ee_pose(q0)
gizmo = server.scene.add_transform_controls(
    "/ee_target",
    scale=0.25,
    position=np.asarray(T_ee0.translation()),
    wxyz=np.asarray(T_ee0.rotation().wxyz),
)

# ---------- GUI ----------
gui_status = server.gui.add_text("Status", initial_value="Idle", disabled=True)
gui_rws_status = server.gui.add_text("RWS", initial_value="?", disabled=True)
gui_egm_status = server.gui.add_text("EGM", initial_value="idle", disabled=True)
gui_wp_count = server.gui.add_text("Waypoints", initial_value="0", disabled=True)
gui_add_wp = server.gui.add_button("Add waypoint (from gizmo)")
gui_pop_wp = server.gui.add_button("Remove last waypoint")
gui_clear_wp = server.gui.add_button("Clear waypoints")
gui_plan = server.gui.add_button("Plan")
gui_play = server.gui.add_button("Play", disabled=True)
gui_stop = server.gui.add_button("Stop", disabled=True)
gui_reset = server.gui.add_button("Reset gizmo to current EE")
gui_speed = server.gui.add_slider("Speed (unified)", min=0.1, max=2.0, step=0.05, initial_value=1.0)
gui_execute = server.gui.add_checkbox("Execute on robot (EGM stream)", initial_value=False)


def _refresh_wp_count() -> None:
    gui_wp_count.value = str(len(waypoints))


@gui_add_wp.on_click
def _(_):
    pos = np.asarray(gizmo.position)
    wxyz = np.asarray(gizmo.wxyz)
    waypoints.append((pos, wxyz))
    handle = server.scene.add_frame(
        f"/waypoints/{len(waypoints) - 1}",
        position=pos, wxyz=wxyz, axes_length=0.12, axes_radius=0.005,
    )
    waypoint_frames.append(handle)
    _refresh_wp_count()
    gui_status.value = f"Added waypoint {len(waypoints)}"


@gui_pop_wp.on_click
def _(_):
    if not waypoints:
        return
    waypoints.pop()
    waypoint_frames.pop().remove()
    _refresh_wp_count()


@gui_clear_wp.on_click
def _(_):
    for h in waypoint_frames:
        h.remove()
    waypoint_frames.clear()
    waypoints.clear()
    _refresh_wp_count()


@gui_reset.on_click
def _(_):
    with state_lock:
        q = current_q.copy()
    T = ee_pose(q)
    gizmo.position = np.asarray(T.translation())
    gizmo.wxyz = np.asarray(T.rotation().wxyz)
    gui_status.value = "Gizmo reset to current EE"


@gui_plan.on_click
def _(_):
    global plan_segments
    targets = waypoints if waypoints else [(np.asarray(gizmo.position), np.asarray(gizmo.wxyz))]
    with state_lock:
        q = current_q.copy()
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for i, (pos, wxyz) in enumerate(targets):
        try:
            q_next = pks.solve_ik_seeded(
                robot=robot, target_link_name=TARGET_LINK,
                target_position=pos, target_wxyz=wxyz,
                q_seed=q, rest_weight=2.0,
            )
        except Exception as e:
            gui_status.value = f"IK failed at waypoint {i + 1}: {e}"
            return
        segments.append((q.copy(), np.asarray(q_next)))
        q = np.asarray(q_next)
    plan_segments = segments
    total = sum(np.linalg.norm(b - a) for a, b in segments)
    gui_play.disabled = False
    gui_status.value = f"Planned {len(segments)} segment(s), total |Δq|={total:.3f}"


def _start_egm_session() -> bool:
    """Trigger RAPID to enter EGMRunJoint and wait for the first feedback packet.

    Returns True if the controller is streaming to us, False on timeout.
    """
    with state_lock:
        q_now = current_q.copy()
    # Pre-load target so the controller has a "hold here" pose when it starts.
    egm.set_target_rad(q_now.tolist())
    try:
        rws.set_rapid_bool(RAPID_GO_FLAG_VAR, True, module=RAPID_MODULE)
    except Exception as e:
        gui_status.value = f"Could not set egm_go=TRUE: {e}"
        return False
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if egm.is_fresh(max_age_s=0.1):
            return True
        time.sleep(0.05)
    gui_status.value = "EGM did not start — no packets from controller in 3s"
    try:
        rws.set_rapid_bool(RAPID_GO_FLAG_VAR, False, module=RAPID_MODULE)
    except Exception:
        pass
    return False


def _wait_egm_clear(timeout_s: float = 8.0) -> bool:
    """After last segment, wait for RAPID to clear egm_go (= EGMRunJoint exited)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if rws.get_rapid_data(RAPID_GO_FLAG_VAR, module=RAPID_MODULE).upper() == "FALSE":
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def _post_execute_cleanup() -> None:
    global plan_segments
    for h in waypoint_frames:
        h.remove()
    waypoint_frames.clear()
    waypoints.clear()
    _refresh_wp_count()
    with state_lock:
        q = current_q.copy()
    T = ee_pose(q)
    gizmo.position = np.asarray(T.translation())
    gizmo.wxyz = np.asarray(T.rotation().wxyz)
    plan_segments = None
    gui_play.disabled = True


def _play() -> None:
    assert plan_segments is not None
    execute = gui_execute.value
    if execute:
        gui_execute.value = False
    dt = 1.0 / STREAM_HZ
    completed = False

    playing.set()
    stop_flag.clear()
    gui_play.disabled = True
    gui_stop.disabled = False
    gui_status.value = "Starting..."

    try:
        if execute:
            gui_status.value = "Starting EGM session..."
            if not _start_egm_session():
                return
            gui_egm_status.value = "streaming"

        for seg_idx, (q_start, q_goal) in enumerate(plan_segments):
            if stop_flag.is_set():
                break
            delta = q_goal - q_start
            seg_duration = max(MIN_SEG_DURATION_S, float(np.max(np.abs(delta))) / MAX_JOINT_SPEED)
            gui_status.value = f"Segment {seg_idx + 1}/{len(plan_segments)}"

            alpha = 0.0
            while alpha < 1.0:
                if stop_flag.is_set():
                    break
                speed = float(gui_speed.value)
                eased = _alpha_to_s(alpha)
                q = q_start + delta * eased
                viser_urdf.update_cfg(q)
                if execute:
                    egm.set_target_rad(q.tolist())
                time.sleep(dt)
                alpha = min(1.0, alpha + dt * speed / seg_duration)

            # Inter-segment dwell: keep streaming the segment endpoint so the
            # controller holds position rather than drifting.
            if not stop_flag.is_set() and seg_idx < len(plan_segments) - 1:
                dwell_ticks = int(max(0.0, DWELL_S / max(0.1, float(gui_speed.value))) * STREAM_HZ)
                viser_urdf.update_cfg(q_goal)
                for _ in range(dwell_ticks):
                    if stop_flag.is_set():
                        break
                    if execute:
                        egm.set_target_rad(q_goal.tolist())
                    time.sleep(dt)

        completed = not stop_flag.is_set()

        if execute:
            # Hold the final target long enough for RAPID's \CondTime (1s) to
            # trigger and exit EGMRunJoint. If stopped mid-flight, hold current
            # robot feedback instead so it stops where it is rather than slamming
            # to the unfinished waypoint.
            if completed:
                hold_target = plan_segments[-1][1].copy()
            else:
                fb = egm.get_feedback_rad()
                hold_target = np.array(fb) if fb is not None else current_q.copy()
            gui_status.value = "Settling (EGM convergence)..."
            for _ in range(int(HOLD_AFTER_PLAY_S * STREAM_HZ)):
                egm.set_target_rad(hold_target.tolist())
                time.sleep(dt)

            if _wait_egm_clear():
                gui_egm_status.value = "idle"
            else:
                gui_egm_status.value = "stuck? (egm_go still TRUE)"

    finally:
        playing.clear()
        gui_stop.disabled = True
        gui_status.value = "Stopped" if stop_flag.is_set() else "Done"

    if execute and completed:
        time.sleep(0.5)
        _post_execute_cleanup()
    else:
        gui_play.disabled = plan_segments is None


@gui_play.on_click
def _(_):
    if plan_segments is None:
        return
    threading.Thread(target=_play, daemon=True).start()


@gui_stop.on_click
def _(_):
    stop_flag.set()


def viz_loop() -> None:
    period = 1.0 / PLAY_HZ
    while True:
        if not playing.is_set():
            with state_lock:
                q = current_q.copy()
                ok = last_poll_ok
            viser_urdf.update_cfg(q)
            gui_rws_status.value = "OK" if ok else "DISCONNECTED"
            gui_egm_status.value = (
                f"rx={egm.packets_rx} tx={egm.packets_tx}"
                if egm.has_feedback() else "idle (no packets)"
            )
        time.sleep(period)


threading.Thread(target=viz_loop, daemon=True).start()

print("viser running. Open the URL printed above in a browser.")
print(f"RWS target: https://{ROBOT_IP}  user={RWS_USER!r}")
print(f"EGM local port: {EGM_LOCAL_PORT}")
while True:
    time.sleep(1.0)
