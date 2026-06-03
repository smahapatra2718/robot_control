"""
viser + pyroki teleop scaffold for a UR15.

Flow:
  - Background thread polls real UR15 joints via RTDE -> shared `current_q`.
  - Viser shows the live state. A 6-DoF gizmo sits at the gripper grasp point.
  - User drags the gizmo freely. Robot does NOT move.
  - "Plan"  -> solve IK from current_q to gizmo pose, build a joint-space trajectory.
  - "Play"  -> animate the URDF through the trajectory in viser.
              If "Execute on robot" is checked, also stream to the UR15.

A Robotiq Hand-E gripper is mounted on the wrist: its mesh rides tool0 in viser,
the gizmo/IK target sits at the grasp point (a fixed tool0 offset, so IK is
unchanged), and Open/Close buttons drive the real gripper over the Robotiq
URCap socket (see hande_gripper.py) plus a matching viz animation.

Run from project root so `pyroki_snippets/` is on the path:
  ./robot_control/bin/python teleop_ur15.py
"""

import datetime
import itertools
import json
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
from robot_descriptions.loaders.yourdfpy import load_robot_description
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from viser.extras import ViserUrdf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import hande_gripper  # noqa: E402
import pyroki_snippets as pks  # noqa: E402

# ---------- config ----------
ROBOT_IP = "192.168.125.2"
ROBOT_DESCRIPTION = "ur15_description"
TARGET_LINK = "tool0"
POLL_HZ = 30
PLAY_HZ = 60                    # viz refresh when idle
STREAM_HZ = 50                  # shared rate for viz playback and servoJ
LIVE_HZ = 30                    # live gizmo-follow: IK + servoJ update rate
GIZMO_SNAP_TWEEN_S = 0.8        # slew the gizmo orientation when snapping in Live (in-place reorient)
MAX_JOINT_SPEED = 1.0           # rad/s peak per joint at slider=1.0
MIN_SEG_DURATION_S = 0.5        # floor on segment time so tiny moves are still smooth
DWELL_S = 0.2                   # pause at each intermediate waypoint
RAMP_FRAC = 0.25                # fraction of segment spent ramping up (same for ramp-down)
SERVO_LOOKAHEAD = 0.1           # servoJ lookahead_time (s)
SERVO_GAIN = 300                # servoJ gain during motion
SERVO_STOP_DECEL = 2.0          # rad/s^2 at end-of-trajectory servoStop (default 10 is harsh)
SETTLE_GAIN = 600               # stiffer servoJ gain for the static end-of-play hold (tighter convergence)
SETTLE_TOL_RAD = 0.0            # 0 = no "good enough" early-out; converge to the servoJ floor (plateau) or the cap
SETTLE_EPS_RAD = 0.00002        # min per-check improvement to count as "still converging"
SETTLE_STALL_TICKS = 10         # consecutive non-improving checks => at the servoJ floor, stop
SETTLE_MAX_S = 3.0              # hard cap on the final convergence hold

# ---- Hand-E gripper ----
GRIPPER_URDF_PATH = os.path.join(_HERE, "hande.urdf")
GRIPPER_HOST = ROBOT_IP         # Robotiq Grippers URCap socket server (on the UR controller)
GRIPPER_PORT = hande_gripper.DEFAULT_PORT
GRIPPER_FINGER_OPEN = 0.025     # per-side finger travel (m) = URDF upper limit (open)
GRIPPER_TWEEN_S = 0.8           # viz finger animation duration to match the real move
GRIPPER_MASS = 1.0              # Hand-E payload (kg) told to the UR so it compensates gravity at the loaded wrist
GRIPPER_COG = (0.0, 0.0, 0.06)  # payload center of gravity in the tool-flange frame (m); raise if you add a workpiece
GRIP_PREDELAY_S = 0.5           # hold/settle the arm this long before actuating the gripper
TRAJ_DIR = os.path.join(_HERE, "trajectories")  # saved teach trajectories (<name>.json)


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


# ---------- gripper model ----------
def _resolve_gripper_mesh(fname: str) -> str:
    if fname.startswith("package://"):
        pkg, rest = fname[len("package://") :].split("/", 1)
        return os.path.join(_HERE, pkg, rest)
    return fname


gripper_urdf = yourdfpy.URDF.load(GRIPPER_URDF_PATH, filename_handler=_resolve_gripper_mesh)
# Fixed tool0 -> grasp-point (hande_end) offset, read straight from the URDF.
# The gripper is rigid, so this is a constant; IK keeps targeting tool0.
gripper_urdf.update_cfg(np.array([GRIPPER_FINGER_OPEN]))
TOOL0_T_GRASP = jaxlie.SE3.from_matrix(
    jnp.asarray(gripper_urdf.get_transform("robotiq_hande_end", "tool0"))
)

# ---------- shared state ----------
state_lock = threading.Lock()
current_q = np.zeros(NUM_JOINTS)
# Each waypoint: {"q": [6 joints]|None, "pos": [x,y,z], "wxyz": [w,x,y,z], "grip": "open"|"close"|None}
# q is filled by free-drive capture (real joints) or backfilled by IK at Plan.
waypoints: list[dict] = []
waypoint_frames: list = []                            # corresponding viser frame handles
plan_segments: list[tuple[np.ndarray, np.ndarray, str | None]] | None = None  # (q_start, q_goal, grip)
gripper_finger = GRIPPER_FINGER_OPEN   # displayed per-side finger opening (m); rendered value
gripper_state = "open"                 # single source of truth: "open" | "close". Known because
                                       # we command open at startup and only we change it after.
playing = threading.Event()
live = threading.Event()               # live gizmo-follow (continuous IK + servoJ) active
freedrive = threading.Event()          # hand-guiding (teachMode) active
stop_flag = threading.Event()
shutdown = threading.Event()           # set on Ctrl-C so daemon loops stop touching RTDE

# ---------- UR15 RTDE ----------
rtde_r = RTDEReceiveInterface(ROBOT_IP)
rtde_c = RTDEControlInterface(ROBOT_IP)   # only used if Execute is toggled on

# Tell the controller about the gripper so it compensates gravity at the loaded
# wrist — otherwise servoJ holds the loaded joints slightly below target (droop).
try:
    rtde_c.setPayload(GRIPPER_MASS, list(GRIPPER_COG))
    print(f"Payload set: {GRIPPER_MASS} kg @ {GRIPPER_COG} m (gripper).")
except Exception as e:
    print(f"setPayload failed ({e}); end-of-play droop may be larger.")

# ---------- Hand-E gripper (best-effort: viz still works if it's unreachable) ----------
try:
    gripper: hande_gripper.HandEGripper | None = hande_gripper.HandEGripper(
        GRIPPER_HOST, GRIPPER_PORT
    )
    gripper.connect()
    gripper.activate()
    gripper.open()   # known initial state: open => gripper_state ("open") matches reality
    print("Hand-E gripper connected + activated + opened.")
except Exception as e:
    gripper = None
    print(f"Hand-E gripper unavailable ({e}); running viz-only gripper.")


# UR safety modes (rtde_r.getSafetyMode()). Anything but NORMAL means the
# controller has stopped the arm — e-stop, protective stop (collision / force
# limit / joint out of range), or safeguard.
_UR_SAFETY_MODES = {
    1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY",
    5: "SAFEGUARD_STOP", 6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP",
    8: "VIOLATION", 9: "FAULT", 10: "VALIDATE_JOINT_ID", 11: "UNDEFINED",
}

# Set once the GUI is built; the poll thread starts before then, so it stays
# None until assigned and the poller skips it until then.
gui_safety = None


def poll_loop() -> None:
    global current_q
    period = 1.0 / POLL_HZ
    last_safety = None
    tick = 0
    while not shutdown.is_set():
        q = np.asarray(rtde_r.getActualQ(), dtype=np.float64)
        with state_lock:
            current_q = q
        # ~3 Hz safety-state check; print only on a transition.
        tick += 1
        if tick % max(1, POLL_HZ // 3) == 0:
            try:
                mode = rtde_r.getSafetyMode()
            except Exception:
                mode = last_safety
            if mode != last_safety:
                name = _UR_SAFETY_MODES.get(mode, f"mode {mode}")
                if mode == 1:
                    if last_safety is not None:
                        print("[safety] cleared -> NORMAL")
                    box = "NORMAL"
                else:
                    print(f"[safety] *** {name} *** "
                          "— robot stopped; clear it on the pendant")
                    box = f"{name} — clear on pendant"
                if gui_safety is not None:
                    gui_safety.value = box
                last_safety = mode
        time.sleep(period)


threading.Thread(target=poll_loop, daemon=True).start()
time.sleep(0.2)  # let one reading land before reading current_q below


def ee_pose(q: np.ndarray) -> jaxlie.SE3:
    Ts = robot.forward_kinematics(cfg=jnp.array(q))
    return jaxlie.SE3(Ts[TARGET_LINK_IDX])


def grasp_pose(q: np.ndarray) -> jaxlie.SE3:
    """Grasp-point (gripper fingertip) pose for joint config q. The gizmo lives here."""
    return ee_pose(q).multiply(TOOL0_T_GRASP)


def _grasp_to_tool0(pos: np.ndarray, wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a grasp-point target (gizmo/waypoint) back to a tool0 target for IK."""
    T_grasp = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(jnp.asarray(wxyz)), jnp.asarray(pos)
    )
    T_tool0 = T_grasp.multiply(TOOL0_T_GRASP.inverse())
    return np.asarray(T_tool0.translation()), np.asarray(T_tool0.rotation().wxyz)


def _warmup_ik() -> None:
    """Pre-compile the IK solver (JAX JIT, ~800 ms) so the first Plan is fast.
    Solves a no-op IK at the current pose; result discarded."""
    with state_lock:
        q = current_q.copy()
    T = ee_pose(q)
    try:
        pks.solve_ik_seeded(
            robot=robot,
            target_link_name=TARGET_LINK,
            target_position=np.asarray(T.translation()),
            target_wxyz=np.asarray(T.rotation().wxyz),
            q_seed=q,
            rest_weight=2.0,
        )
    except Exception as e:
        print(f"IK warmup skipped: {e}")


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

# Hand-E rides the wrist: a second ViserUrdf rooted at /world/gripper, whose frame
# we slave to the live tool0 pose every tick (see _update_gripper_viz). Both live
# under /world, so the display yaw above is inherited; IK is unaffected.
gripper_frame = server.scene.add_frame("/world/gripper", show_axes=False)
gripper_viser = ViserUrdf(server, gripper_urdf, root_node_name="/world/gripper")

with state_lock:
    q0 = current_q.copy()
T_grasp0 = grasp_pose(q0)
gizmo = server.scene.add_transform_controls(
    "/world/ee_target",
    scale=0.25,
    position=np.asarray(T_grasp0.translation()),
    wxyz=np.asarray(T_grasp0.rotation().wxyz),
)


def _update_gripper_viz(q: np.ndarray) -> None:
    """Glue the gripper meshes to the wrist and show the current finger opening."""
    T = ee_pose(q)
    gripper_frame.wxyz = np.asarray(T.rotation().wxyz)
    gripper_frame.position = np.asarray(T.translation())
    gripper_viser.update_cfg(np.array([gripper_finger]))

# ---------- GUI ----------
gui_status = server.gui.add_text("Status", initial_value="Idle", disabled=True)
gui_safety = server.gui.add_text("Safety", initial_value="NORMAL", disabled=True)
gui_wp_count = server.gui.add_text("Waypoints", initial_value="0", disabled=True)
gui_freedrive = server.gui.add_checkbox("Free-drive (hand-guide)", initial_value=False)
gui_add_wp = server.gui.add_button("Add waypoint (from gizmo)")
gui_capture = server.gui.add_button("Capture waypoint (current pose)")
gui_pop_wp = server.gui.add_button("Remove last waypoint")
gui_clear_wp = server.gui.add_button("Clear waypoints")
gui_plan = server.gui.add_button("Plan")
gui_play = server.gui.add_button("Play", disabled=True)
gui_stop = server.gui.add_button("Stop", disabled=True)
gui_reset = server.gui.add_button("Reset gizmo to current EE")
gui_snap = server.gui.add_button("Snap gizmo to nearest axis")
gui_speed = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.05, initial_value=1.0)
gui_execute = server.gui.add_checkbox("Execute on robot", initial_value=False)
gui_live = server.gui.add_checkbox("Live (drive robot)", initial_value=False)
gui_grip_open = server.gui.add_button("Open gripper")
gui_grip_close = server.gui.add_button("Close gripper")
gui_grip_status = server.gui.add_text(
    "Gripper", initial_value=("ready" if gripper is not None else "viz-only"), disabled=True
)
gui_traj_name = server.gui.add_text("Trajectory name", initial_value="traj1")
gui_save = server.gui.add_button("Save trajectory")
gui_load = server.gui.add_button("Load trajectory")


def _refresh_wp_count() -> None:
    gui_wp_count.value = str(len(waypoints))


def _add_waypoint(wp: dict) -> None:
    """Append a waypoint dict and draw its frame (label includes the grip action)."""
    i = len(waypoints)
    waypoints.append(wp)
    handle = server.scene.add_frame(
        f"/world/waypoints/{i}",
        position=np.asarray(wp["pos"]),
        wxyz=np.asarray(wp["wxyz"]),
        axes_length=0.12,
        axes_radius=0.005,
    )
    waypoint_frames.append(handle)
    _refresh_wp_count()


@gui_add_wp.on_click
def _(_):
    _add_waypoint({
        "q": None,  # filled by IK at Plan
        "pos": np.asarray(gizmo.position).tolist(),
        "wxyz": np.asarray(gizmo.wxyz).tolist(),
        "grip": gripper_state,
    })
    gui_status.value = f"Added waypoint {len(waypoints)} (gizmo)"


@gui_capture.on_click
def _(_):
    with state_lock:
        q = current_q.copy()
    T = grasp_pose(q)
    _add_waypoint({
        "q": q.tolist(),  # taught joints — replayed exactly, no IK
        "pos": np.asarray(T.translation()).tolist(),
        "wxyz": np.asarray(T.rotation().wxyz).tolist(),
        "grip": gripper_state,
    })
    gui_status.value = f"Captured waypoint {len(waypoints)} (joints)"


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
    if waypoints:
        targets = waypoints
    else:
        targets = [{"q": None, "pos": np.asarray(gizmo.position).tolist(),
                    "wxyz": np.asarray(gizmo.wxyz).tolist(), "grip": gripper_state}]
    with state_lock:
        q = current_q.copy()
    segments: list[tuple[np.ndarray, np.ndarray, str | None]] = []
    for i, wp in enumerate(targets):
        if wp.get("q") is not None:
            q_next = np.asarray(wp["q"], dtype=np.float64)  # taught joints, replay exactly
        else:
            pos_t, wxyz_t = _grasp_to_tool0(np.asarray(wp["pos"]), np.asarray(wp["wxyz"]))
            try:
                q_next = np.asarray(pks.solve_ik_seeded(
                    robot=robot,
                    target_link_name=TARGET_LINK,
                    target_position=pos_t,
                    target_wxyz=wxyz_t,
                    q_seed=q,
                    rest_weight=2.0,
                ))
            except Exception as e:
                gui_status.value = f"IK failed at waypoint {i + 1}: {e}"
                return
            wp["q"] = q_next.tolist()  # backfill so a save-after-plan carries joints too
        segments.append((q.copy(), q_next, wp.get("grip")))
        q = q_next
    plan_segments = segments
    total = sum(np.linalg.norm(b - a) for a, b, _ in segments)
    gui_play.disabled = False
    gui_status.value = f"Planned {len(segments)} segment(s), total |Δq|={total:.3f}"


@gui_reset.on_click
def _(_):
    with state_lock:
        q = current_q.copy()
    T = grasp_pose(q)
    gizmo.position = np.asarray(T.translation())
    gizmo.wxyz = np.asarray(T.rotation().wxyz)
    gui_status.value = "Gizmo reset to current EE"


def _nearest_axis_aligned_wxyz(wxyz: np.ndarray) -> np.ndarray:
    """Snap an orientation to the nearest of the 24 rotations whose axes are each
    parallel to a base-frame axis (each local axis -> +/- a world axis). Picks the
    one with the largest trace(R^T M), i.e. the smallest rotation away from R."""
    R = np.asarray(jaxlie.SO3(jnp.asarray(wxyz)).as_matrix())
    best_M, best_score = None, -np.inf
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for col, (row, s) in enumerate(zip(perm, signs)):
                M[row, col] = s
            if np.linalg.det(M) < 0:  # keep proper (right-handed) rotations only
                continue
            score = float(np.sum(R * M))
            if score > best_score:
                best_M, best_score = M, score
    return np.asarray(jaxlie.SO3.from_matrix(jnp.asarray(best_M)).wxyz)


def _set_gizmo_orientation(target_wxyz: np.ndarray) -> None:
    """Set the gizmo orientation, keeping position. Outside Live this is instant.
    In Live, slerp to the target so the live loop tracks a gradual orientation
    change (which holds the grasp point fixed) instead of chasing a jump — a jump
    slews in joint space and arcs the point out and back."""
    if not live.is_set():
        gizmo.wxyz = np.asarray(target_wxyz)
        return
    R0 = jaxlie.SO3(jnp.asarray(gizmo.wxyz))
    delta = jaxlie.SO3(jnp.asarray(target_wxyz)).multiply(R0.inverse()).log()
    steps = max(1, int(GIZMO_SNAP_TWEEN_S * LIVE_HZ))

    def _run() -> None:
        for i in range(1, steps + 1):
            if not live.is_set():
                break
            gizmo.wxyz = np.asarray(jaxlie.SO3.exp(delta * (i / steps)).multiply(R0).wxyz)
            time.sleep(1.0 / LIVE_HZ)

    threading.Thread(target=_run, daemon=True).start()


@gui_snap.on_click
def _(_):
    # In Live, re-anchor the gizmo to the robot's actual current grasp point first.
    # While dragging, the arm lags the gizmo (joint clamp), so gizmo.position sits
    # ahead of the real EE; snapping would then reorient AND chase that stale
    # position, which reads as a Cartesian jump. Re-anchoring leaves a pure in-place
    # reorient. (Outside Live the robot isn't moving, so leave position as dragged.)
    if live.is_set():
        with state_lock:
            q = current_q.copy()
        gizmo.position = np.asarray(grasp_pose(q).translation())
    # gizmo.wxyz is stored in the base frame, but the scene is *drawn* under a
    # VIZ_YAW_DEG display yaw, so snapping in the base frame looks tilted on
    # screen. Compose into the displayed (ground-grid) frame, snap there, then
    # compose back — so the gizmo lands parallel to the grid axes you eyeball.
    R_yaw = jaxlie.SO3(jnp.array([np.cos(_half_yaw), 0.0, 0.0, np.sin(_half_yaw)]))
    disp = R_yaw @ jaxlie.SO3(jnp.asarray(gizmo.wxyz))
    snapped = jaxlie.SO3(jnp.asarray(_nearest_axis_aligned_wxyz(np.asarray(disp.wxyz))))
    _set_gizmo_orientation(np.asarray((R_yaw.inverse() @ snapped).wxyz))
    gui_status.value = "Gizmo snapped to nearest axis-aligned orientation"


def _actuate_gripper(target_finger: float, action: str) -> None:
    """Command the real gripper (if connected), then tween the viz fingers to match."""
    global gripper_finger
    if gripper is not None:
        try:
            getattr(gripper, action)()
        except Exception as e:
            gui_grip_status.value = f"cmd failed: {e}"
            return
    start = gripper_finger
    steps = max(1, int(GRIPPER_TWEEN_S * PLAY_HZ))
    for i in range(1, steps + 1):
        gripper_finger = start + (target_finger - start) * i / steps
        time.sleep(1.0 / PLAY_HZ)
    gripper_finger = target_finger
    if gripper is not None:
        gui_grip_status.value = "object grasped" if gripper.status()["object"] else (
            "open" if target_finger > 0 else "closed"
        )


@gui_grip_open.on_click
def _(_):
    global gripper_state
    gripper_state = "open"
    threading.Thread(
        target=_actuate_gripper, args=(GRIPPER_FINGER_OPEN, "open"), daemon=True
    ).start()


@gui_grip_close.on_click
def _(_):
    global gripper_state
    gripper_state = "close"
    threading.Thread(
        target=_actuate_gripper, args=(0.0, "close_gripper"), daemon=True
    ).start()


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
    T = grasp_pose(q)
    gizmo.position = np.asarray(T.translation())
    gizmo.wxyz = np.asarray(T.rotation().wxyz)
    plan_segments = None
    gui_play.disabled = True


def _play() -> None:
    global gripper_finger, gripper_state
    assert plan_segments is not None
    execute = gui_execute.value   # latched at play-start
    if execute:
        gui_execute.value = False  # require explicit opt-in for every executed move
    dt = 1.0 / STREAM_HZ
    completed = False

    playing.set()
    stop_flag.clear()
    gui_play.disabled = True
    gui_live.disabled = True
    gui_freedrive.disabled = True
    gui_stop.disabled = False

    try:
        cur_grip = gripper_state   # running gripper state; only actuate when a waypoint differs
        for seg_idx, (q_start, q_goal, grip) in enumerate(plan_segments):
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
                _update_gripper_viz(q)
                if execute:
                    rtde_c.servoJ(q.tolist(), 0.0, 0.0, dt, SERVO_LOOKAHEAD, SERVO_GAIN)
                time.sleep(dt)
                alpha = min(1.0, alpha + dt * speed / seg_duration)

            # gripper action: only when this waypoint's state differs from the
            # running state. Settle the arm GRIP_PREDELAY_S first (keep streaming
            # the pose), then actuate. Real command only on an executed play.
            if not stop_flag.is_set() and grip in ("open", "close") and grip != cur_grip:
                gui_status.value = f"Settling before gripper: {grip}"
                for _ in range(int(GRIP_PREDELAY_S * STREAM_HZ)):
                    if stop_flag.is_set():
                        break
                    if execute:
                        rtde_c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, SERVO_LOOKAHEAD, SERVO_GAIN)
                    time.sleep(dt)

                gui_status.value = f"Gripper: {grip}"
                target_f = GRIPPER_FINGER_OPEN if grip == "open" else 0.0
                if execute and gripper is not None:
                    try:
                        getattr(gripper, "open" if grip == "open" else "close_gripper")()
                    except Exception as e:
                        gui_grip_status.value = f"cmd failed: {e}"
                start_f = gripper_finger
                steps = max(1, int(GRIPPER_TWEEN_S * STREAM_HZ))
                for k in range(1, steps + 1):
                    if stop_flag.is_set():
                        break
                    gripper_finger = start_f + (target_f - start_f) * k / steps
                    if execute:
                        rtde_c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, SERVO_LOOKAHEAD, SERVO_GAIN)
                    viser_urdf.update_cfg(q_goal)
                    _update_gripper_viz(q_goal)
                    time.sleep(dt)
                gripper_finger = target_f
                gripper_state = grip   # keep the global in sync with what we just commanded
                cur_grip = grip

            # dwell at the waypoint (still streaming to hold pose firmly)
            if not stop_flag.is_set() and seg_idx < len(plan_segments) - 1:
                hold = max(0.0, DWELL_S / max(0.1, float(gui_speed.value)))
                t = 0.0
                while t < hold and not stop_flag.is_set():
                    if execute:
                        rtde_c.servoJ(q_goal.tolist(), 0.0, 0.0, dt, SERVO_LOOKAHEAD, SERVO_GAIN)
                    time.sleep(dt)
                    t += dt

        # Final settle: servoJ trails its setpoint during motion, so keep
        # streaming the last target and WAIT until the measured joints actually
        # arrive (a held setpoint has ~0 steady-state error). Otherwise the arm
        # halts short of the end and the gizmo re-anchors to that short pose.
        if execute and not stop_flag.is_set():
            q_final = plan_segments[-1][1]
            deadline = time.monotonic() + SETTLE_MAX_S
            best, stalls = float("inf"), 0
            while not stop_flag.is_set():
                rtde_c.servoJ(q_final.tolist(), 0.0, 0.0, dt, SERVO_LOOKAHEAD, SETTLE_GAIN)
                time.sleep(dt)
                with state_lock:
                    err = float(np.max(np.abs(current_q - q_final)))
                if err < best - SETTLE_EPS_RAD:
                    best, stalls = err, 0
                else:
                    stalls += 1
                # stop when converged, when error plateaus (hardware floor), or at the cap
                if err < SETTLE_TOL_RAD or stalls >= SETTLE_STALL_TICKS or time.monotonic() > deadline:
                    if time.monotonic() > deadline and err >= SETTLE_TOL_RAD:
                        print(f"[settle] hit SETTLE_MAX_S cap at {np.degrees(err):.3f} deg")
                    break
        completed = not stop_flag.is_set()
    finally:
        if execute:
            rtde_c.servoStop(SERVO_STOP_DECEL)
        playing.clear()
        gui_stop.disabled = True
        gui_live.disabled = False
        gui_freedrive.disabled = False
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
    live.clear()
    if gui_live.value:
        gui_live.value = False
    if gui_freedrive.value:
        gui_freedrive.value = False   # triggers off-branch -> endTeachMode
    gui_plan.disabled = False


def _live_loop() -> None:
    """Continuously chase the gizmo: solve IK each tick and servoJ there, with a
    per-tick joint-step clamp so a fast drag or IK branch-flip can't make the arm
    lurch — it just rate-limits toward the target."""
    dt = 1.0 / LIVE_HZ
    max_step = MAX_JOINT_SPEED * dt
    with state_lock:
        q_cmd = current_q.copy()
    try:
        while live.is_set() and not stop_flag.is_set():
            pos_t, wxyz_t = _grasp_to_tool0(np.asarray(gizmo.position), np.asarray(gizmo.wxyz))
            try:
                q_target = np.asarray(pks.solve_ik_seeded(
                    robot=robot,
                    target_link_name=TARGET_LINK,
                    target_position=pos_t,
                    target_wxyz=wxyz_t,
                    q_seed=q_cmd,
                    rest_weight=2.0,
                ))
            except Exception:
                time.sleep(dt)
                continue
            q_cmd = q_cmd + np.clip(q_target - q_cmd, -max_step, max_step)
            viser_urdf.update_cfg(q_cmd)
            _update_gripper_viz(q_cmd)
            rtde_c.servoJ(q_cmd.tolist(), 0.0, 0.0, dt, SERVO_LOOKAHEAD, SERVO_GAIN)
            time.sleep(dt)
    finally:
        rtde_c.servoStop(SERVO_STOP_DECEL)
        gui_status.value = "Live off"


@gui_live.on_update
def _(_):
    if gui_live.value:
        # snap the gizmo to the current EE first, so the arm doesn't lurch toward
        # a stale gizmo pose, then start chasing it
        with state_lock:
            q = current_q.copy()
        T = grasp_pose(q)
        gizmo.position = np.asarray(T.translation())
        gizmo.wxyz = np.asarray(T.rotation().wxyz)
        stop_flag.clear()
        live.set()
        gui_plan.disabled = True
        gui_play.disabled = True
        gui_freedrive.disabled = True
        gui_stop.disabled = False
        gui_status.value = "LIVE: arm follows gizmo"
        threading.Thread(target=_live_loop, daemon=True).start()
    else:
        live.clear()
        gui_plan.disabled = False
        gui_play.disabled = plan_segments is None
        gui_freedrive.disabled = False
        gui_stop.disabled = True


@gui_freedrive.on_update
def _(_):
    if gui_freedrive.value:
        try:
            rtde_c.teachMode()   # zero-gravity hand-guiding
        except Exception as e:
            gui_status.value = f"teachMode failed: {e}"
            gui_freedrive.value = False
            return
        freedrive.set()
        gui_plan.disabled = True
        gui_play.disabled = True
        gui_live.disabled = True
        gui_stop.disabled = False
        gui_status.value = "FREE-DRIVE: move the arm by hand, then Capture waypoint"
    else:
        freedrive.clear()
        try:
            rtde_c.endTeachMode()
        except Exception:
            pass
        gui_plan.disabled = False
        gui_play.disabled = plan_segments is None
        gui_live.disabled = False
        gui_stop.disabled = True
        gui_status.value = "Free-drive off"


@gui_save.on_click
def _(_):
    os.makedirs(TRAJ_DIR, exist_ok=True)
    name = (gui_traj_name.value or "traj").strip()
    path = os.path.join(TRAJ_DIR, f"{name}.json")
    data = {
        "robot": "ur15",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "waypoints": waypoints,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    gui_status.value = f"Saved {len(waypoints)} waypoint(s) -> {name}.json"


@gui_load.on_click
def _(_):
    name = (gui_traj_name.value or "traj").strip()
    path = os.path.join(TRAJ_DIR, f"{name}.json")
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        gui_status.value = f"Load failed: {e}"
        return
    global plan_segments
    for h in waypoint_frames:
        h.remove()
    waypoint_frames.clear()
    waypoints.clear()
    for wp in data.get("waypoints", []):
        _add_waypoint(wp)
    plan_segments = None
    gui_play.disabled = True
    gui_status.value = f"Loaded {len(waypoints)} waypoint(s) from {name}.json — Plan to replay"


def viz_loop() -> None:
    period = 1.0 / PLAY_HZ
    while not shutdown.is_set():
        if not playing.is_set() and not live.is_set():
            with state_lock:
                q = current_q.copy()
            viser_urdf.update_cfg(q)
            _update_gripper_viz(q)
        time.sleep(period)


threading.Thread(target=viz_loop, daemon=True).start()

print("Warming up IK solver (JAX JIT compile)...")
_warmup_ik()

print("viser running. Open the URL printed above in a browser.")
try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    print("\nShutting down — disconnecting RTDE cleanly...")
    # Stop any active play/live and tell the daemon loops to stop touching RTDE,
    # then disconnect so ur_rtde joins its boost threads gracefully (otherwise
    # they get force-unwound at teardown -> "FATAL: exception not rethrown").
    shutdown.set()
    stop_flag.set()
    time.sleep(0.2)
    for fn in (
        lambda: rtde_c.endTeachMode() if freedrive.is_set() else None,
        lambda: rtde_c.servoStop(SERVO_STOP_DECEL),
        rtde_c.stopScript,
        rtde_c.disconnect,
        rtde_r.disconnect,
        (gripper.close if gripper is not None else (lambda: None)),
    ):
        try:
            fn()
        except Exception:
            pass
    print("Disconnected.")
    os._exit(0)
