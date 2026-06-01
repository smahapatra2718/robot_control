"""
viser + pyroki teleop scaffold for a UR15.

Flow:
  - Background thread polls real UR15 joints via RTDE -> shared `current_q`.
  - Viser shows the live state. A 6-DoF gizmo sits at the EE.
  - User drags the gizmo freely. Robot does NOT move.
  - "Plan"  -> solve IK from current_q to gizmo pose, build a joint-space trajectory.
  - "Play"  -> animate the URDF through the trajectory in viser.
              If "Execute on robot" is checked, also stream to the UR15.

Run from project root so `pyroki_snippets/` is on the path:
  ./robot_control/bin/python teleop_ur15.py
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
from robot_descriptions.loaders.yourdfpy import load_robot_description
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from viser.extras import ViserUrdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyroki_snippets as pks  # noqa: E402

# ---------- config ----------
ROBOT_IP = "192.168.125.2"
ROBOT_DESCRIPTION = "ur15_description"
TARGET_LINK = "tool0"
POLL_HZ = 30
PLAY_HZ = 60                    # viz refresh when idle
STREAM_HZ = 50                  # shared rate for viz playback and servoJ
MAX_JOINT_SPEED = 1.0           # rad/s peak per joint at slider=1.0
MIN_SEG_DURATION_S = 0.5        # floor on segment time so tiny moves are still smooth
DWELL_S = 0.2                   # pause at each intermediate waypoint
RAMP_FRAC = 0.25                # fraction of segment spent ramping up (same for ramp-down)
SERVO_LOOKAHEAD = 0.1           # servoJ lookahead_time (s)
SERVO_GAIN = 300                # servoJ gain
SERVO_STOP_DECEL = 2.0          # rad/s^2 at end-of-trajectory servoStop (default 10 is harsh)


def _alpha_to_s(alpha: float, r: float = RAMP_FRAC) -> float:
    """Trapezoidal velocity profile: parabolic accel, linear cruise, parabolic decel.
    Maps alpha in [0,1] to traversed fraction s in [0,1] with s'(0)=s'(1)=0.
    r is the ramp fraction (0 < r <= 0.5).
    """
    v_peak = 1.0 / (1.0 - r)  # cruise speed making total area = 1
    if alpha < r:
        return 0.5 * v_peak * alpha * alpha / r
    if alpha < 1.0 - r:
        return 0.5 * v_peak * r + v_peak * (alpha - r)
    return 1.0 - 0.5 * v_peak * (1.0 - alpha) ** 2 / r

# ---------- robot model ----------
urdf = load_robot_description(ROBOT_DESCRIPTION)
robot = pk.Robot.from_urdf(urdf)
TARGET_LINK_IDX = robot.links.names.index(TARGET_LINK)
NUM_JOINTS = robot.joints.num_actuated_joints

# ---------- shared state ----------
state_lock = threading.Lock()
current_q = np.zeros(NUM_JOINTS)
waypoints: list[tuple[np.ndarray, np.ndarray]] = []  # list of (position xyz, wxyz)
waypoint_frames: list = []                            # corresponding viser frame handles
plan_segments: list[tuple[np.ndarray, np.ndarray]] | None = None  # list of (q_start, q_end)
playing = threading.Event()
stop_flag = threading.Event()

# ---------- UR15 RTDE ----------
rtde_r = RTDEReceiveInterface(ROBOT_IP)
rtde_c = RTDEControlInterface(ROBOT_IP)   # only used if Execute is toggled on


def poll_loop() -> None:
    global current_q
    period = 1.0 / POLL_HZ
    while True:
        q = np.asarray(rtde_r.getActualQ(), dtype=np.float64)
        with state_lock:
            current_q = q
        time.sleep(period)


threading.Thread(target=poll_loop, daemon=True).start()
time.sleep(0.2)  # let one reading land before reading current_q below


def ee_pose(q: np.ndarray) -> jaxlie.SE3:
    Ts = robot.forward_kinematics(cfg=jnp.array(q))
    return jaxlie.SE3(Ts[TARGET_LINK_IDX])


# ---------- viser ----------
server = viser.ViserServer()
server.scene.add_grid("/ground", width=2, height=2)

# Display-only yaw: render the robot + gizmo + waypoints rotated about world Z.
# Everything nested under /world inherits this rotation, but viser node poses are
# parent-relative, so each child's local pose (which IK reads/writes) stays in the
# robot base frame — the real robot motion is unchanged.
VIZ_YAW_DEG = 30.0
_half_yaw = np.deg2rad(VIZ_YAW_DEG) / 2.0
server.scene.add_frame(
    "/world",
    show_axes=False,
    wxyz=(np.cos(_half_yaw), 0.0, 0.0, np.sin(_half_yaw)),
    position=(0.0, 0.0, 0.0),
)
viser_urdf = ViserUrdf(server, urdf, root_node_name="/world/base")

with state_lock:
    q0 = current_q.copy()
T_ee0 = ee_pose(q0)
gizmo = server.scene.add_transform_controls(
    "/world/ee_target",
    scale=0.25,
    position=np.asarray(T_ee0.translation()),
    wxyz=np.asarray(T_ee0.rotation().wxyz),
)

# ---------- GUI ----------
gui_status = server.gui.add_text("Status", initial_value="Idle", disabled=True)
gui_wp_count = server.gui.add_text("Waypoints", initial_value="0", disabled=True)
gui_add_wp = server.gui.add_button("Add waypoint (from gizmo)")
gui_pop_wp = server.gui.add_button("Remove last waypoint")
gui_clear_wp = server.gui.add_button("Clear waypoints")
gui_plan = server.gui.add_button("Plan")
gui_play = server.gui.add_button("Play", disabled=True)
gui_stop = server.gui.add_button("Stop", disabled=True)
gui_reset = server.gui.add_button("Reset gizmo to current EE")
gui_speed = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.05, initial_value=1.0)
gui_execute = server.gui.add_checkbox("Execute on robot", initial_value=False)


def _refresh_wp_count() -> None:
    gui_wp_count.value = str(len(waypoints))


@gui_add_wp.on_click
def _(_):
    pos = np.asarray(gizmo.position)
    wxyz = np.asarray(gizmo.wxyz)
    waypoints.append((pos, wxyz))
    handle = server.scene.add_frame(
        f"/world/waypoints/{len(waypoints)-1}",
        position=pos,
        wxyz=wxyz,
        axes_length=0.12,
        axes_radius=0.005,
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
    gui_status.value = "Removed last waypoint"


@gui_clear_wp.on_click
def _(_):
    for h in waypoint_frames:
        h.remove()
    waypoint_frames.clear()
    waypoints.clear()
    _refresh_wp_count()
    gui_status.value = "Cleared waypoints"


@gui_plan.on_click
def _(_):
    global plan_segments
    # If no waypoints have been added, treat the current gizmo pose as a single target.
    targets = waypoints if waypoints else [(np.asarray(gizmo.position), np.asarray(gizmo.wxyz))]
    with state_lock:
        q = current_q.copy()
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for i, (pos, wxyz) in enumerate(targets):
        try:
            q_next = pks.solve_ik_seeded(
                robot=robot,
                target_link_name=TARGET_LINK,
                target_position=pos,
                target_wxyz=wxyz,
                q_seed=q,
                rest_weight=2.0,
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


@gui_reset.on_click
def _(_):
    with state_lock:
        q = current_q.copy()
    T = ee_pose(q)
    gizmo.position = np.asarray(T.translation())
    gizmo.wxyz = np.asarray(T.rotation().wxyz)
    gui_status.value = "Gizmo reset to current EE"


def _post_execute_cleanup() -> None:
    """After a successful execute, drop the consumed waypoints/plan and re-anchor the
    gizmo at the new EE pose so the next plan starts from where the robot now is."""
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
    execute = gui_execute.value   # latched at play-start
    if execute:
        gui_execute.value = False  # require explicit opt-in for every executed move
    dt = 1.0 / STREAM_HZ
    completed = False

    playing.set()
    stop_flag.clear()
    gui_play.disabled = True
    gui_stop.disabled = False

    try:
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
                    rtde_c.servoJ(q.tolist(), 0.0, 0.0, dt, SERVO_LOOKAHEAD, SERVO_GAIN)
                time.sleep(dt)
                alpha = min(1.0, alpha + dt * speed / seg_duration)

            # dwell at the waypoint (still streaming to hold pose firmly)
            if not stop_flag.is_set() and seg_idx < len(plan_segments) - 1:
                hold = max(0.0, DWELL_S / max(0.1, float(gui_speed.value)))
                t = 0.0
                while t < hold and not stop_flag.is_set():
                    if execute:
                        rtde_c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, SERVO_LOOKAHEAD, SERVO_GAIN)
                    time.sleep(dt)
                    t += dt
        completed = not stop_flag.is_set()
    finally:
        if execute:
            rtde_c.servoStop(SERVO_STOP_DECEL)
        playing.clear()
        gui_stop.disabled = True
        gui_status.value = "Stopped" if stop_flag.is_set() else "Done"

    if execute and completed:
        # Wait briefly for current_q to catch up to the commanded final pose
        time.sleep(0.15)
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
            viser_urdf.update_cfg(q)
        time.sleep(period)


threading.Thread(target=viz_loop, daemon=True).start()

print("viser running. Open the URL printed above in a browser.")
while True:
    time.sleep(1.0)
