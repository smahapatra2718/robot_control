"""
viser + pyroki teleop for ABB GoFa CRB 15000 using EGM (Externally Guided
Motion) for streaming joint control.

EGM streams joint targets at STREAM_HZ: the same q computed from the
trapezoidal alpha profile goes to BOTH viser AND the EGM target stream every
tick, so the viz and the robot move in lockstep (like teleop_ur15.py with
servoJ). The speed slider scales playback below a TCP-speed cap (MAX_TCP_SPEED).

Prerequisite: install_gofa_egm.py has been run successfully (PyEgm.mod loaded,
EGM_COMM.cfg + EGM_MOC.cfg loaded, controller rebooted, PP-to-Main + Play
done so the RAPID supervisor is parked at WaitUntil egm_go).

Run:
  ./robot_control/bin/python teleop_gofa_egm.py
"""

import datetime
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
from viser.extras import ViserUrdf

import abb_egm
import abb_rws

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import pyroki_snippets as pks  # noqa: E402

# ---------- config ----------
ROBOT_IP = "192.168.125.1"
RWS_USER = "Default User"
RWS_PASSWORD = "robotics"
URDF_PATH = os.path.join(_HERE, "crb15000_5_95.urdf")
URDF_MESH_DIR_PREFIX = os.path.join(_HERE, "abb_desc")
TARGET_LINK = "tool0"

RAPID_GO_FLAG_VAR = "egm_go"        # bool in PyEgm.mod
RAPID_MODULE = "PyEgm"

EGM_LOCAL_PORT = 6510                # must match RemotePortNumber in EGM_COMM.cfg

TRAJ_DIR = os.path.join(_HERE, "trajectories")  # saved teach trajectories (<name>.json)
POLL_HZ = 10                         # RWS state polling (idle viz only)
PLAY_HZ = 60                         # viz refresh when idle
STREAM_HZ = 100                      # EGM target stream rate (controller side runs ~250Hz)
LIVE_HZ = 30                         # live gizmo-follow: IK + EGM target update rate
MAX_JOINT_SPEED = 1.0                # rad/s peak per joint at slider=1.0
MAX_TCP_SPEED = 0.25                 # m/s — hard cap on real tool speed (ISO/TS 15066
                                     # collaborative limit). Enforced against the actual
                                     # kinematics per segment; slider can only slow below it.
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
# Each waypoint: {"q": [6 joints]|None, "pos": [x,y,z], "wxyz": [w,x,y,z]}
# q is filled by free-drive capture (real joints) or backfilled by IK at Plan.
waypoints: list[dict] = []
waypoint_frames: list = []
plan_segments: list[tuple[np.ndarray, np.ndarray]] | None = None
playing = threading.Event()
live = threading.Event()             # live gizmo-follow (continuous IK + EGM stream) active
stop_flag = threading.Event()
shutdown = threading.Event()         # set on Ctrl-C so daemon loops stop and EGM is closed

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


# Set once the GUI is built; the poll thread starts before then, so it stays
# None until assigned and the poller skips it until then.
gui_safety = None


def poll_loop() -> None:
    """RWS state polling at POLL_HZ. Used for idle visualization only — during
    an Execute Play, the play loop drives the URDF directly from the streamed
    target, so this loop's writes are skipped via `playing.is_set()` in viz_loop."""
    global current_q, last_poll_ok
    period = 1.0 / POLL_HZ
    last_state = None
    tick = 0
    while not shutdown.is_set():
        try:
            q = np.array(rws.get_joints(), dtype=np.float64)
            with state_lock:
                current_q = q
                last_poll_ok = True
        except Exception:
            with state_lock:
                last_poll_ok = False
        # ~1 Hz controller-state check; print only on a transition. guardstop
        # covers collision / joint-limit / safeguard; emergencystop = e-stop.
        tick += 1
        if tick % max(1, POLL_HZ) == 0:
            try:
                st = rws.get_controller_state()
            except Exception:
                st = last_state
            if st != last_state:
                if st in ("guardstop", "emergencystop", "sysfail"):
                    print(f"[safety] *** controller state: {st.upper()} *** "
                          "— guardstop = collision / joint limit / safeguard; "
                          "clear it on the pendant")
                    box = f"{st.upper()} — clear on pendant"
                else:
                    if last_state is not None:
                        print(f"[safety] controller state -> {st}")
                    box = st
                if gui_safety is not None:
                    gui_safety.value = box
                last_state = st
        time.sleep(period)


threading.Thread(target=poll_loop, daemon=True).start()
time.sleep(0.4)


def ee_pose(q: np.ndarray) -> jaxlie.SE3:
    Ts = robot.forward_kinematics(cfg=jnp.array(q))
    return jaxlie.SE3(Ts[TARGET_LINK_IDX])


def _warmup_ik() -> None:
    """Pre-compile the IK solver (JAX JIT, ~800 ms) so the first Plan is fast.
    Solves a no-op IK at the current pose; result discarded."""
    with state_lock:
        q = current_q.copy()
    T = ee_pose(q)
    try:
        pks.solve_ik_seeded(
            robot=robot, target_link_name=TARGET_LINK,
            target_position=np.asarray(T.translation()),
            target_wxyz=np.asarray(T.rotation().wxyz),
            q_seed=q, rest_weight=2.0,
        )
    except Exception as e:
        print(f"IK warmup skipped: {e}")


def _cap_seg_duration(q_start: np.ndarray, delta: np.ndarray,
                      seg_duration: float, dt: float) -> float:
    """Stretch seg_duration so the real TCP speed never exceeds MAX_TCP_SPEED.

    Walks the eased trajectory through forward kinematics at slider=1.0 and
    measures the peak instantaneous TCP speed. Because TCP speed scales as
    1/seg_duration for a fixed path, one measurement gives the exact stretch
    factor. The slider only scales below 1.0, so this is the worst case.
    """
    alpha, prev_p, peak = 0.0, np.asarray(ee_pose(q_start).translation()), 0.0
    while alpha < 1.0:
        alpha = min(1.0, alpha + dt / seg_duration)
        p = np.asarray(ee_pose(q_start + delta * _alpha_to_s(alpha)).translation())
        peak = max(peak, float(np.linalg.norm(p - prev_p)) / dt)
        prev_p = p
    if peak > MAX_TCP_SPEED:
        seg_duration *= peak / MAX_TCP_SPEED
    return seg_duration


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
gui_safety = server.gui.add_text("Safety", initial_value="motoron", disabled=True)
gui_rws_status = server.gui.add_text("RWS", initial_value="?", disabled=True)
gui_egm_status = server.gui.add_text("EGM", initial_value="idle", disabled=True)
gui_wp_count = server.gui.add_text("Waypoints", initial_value="0", disabled=True)
gui_add_wp = server.gui.add_button("Add waypoint (from gizmo)")
gui_capture = server.gui.add_button("Capture waypoint (current pose)")
gui_pop_wp = server.gui.add_button("Remove last waypoint")
gui_clear_wp = server.gui.add_button("Clear waypoints")
gui_plan = server.gui.add_button("Plan")
gui_play = server.gui.add_button("Play", disabled=True)
gui_stop = server.gui.add_button("Stop", disabled=True)
gui_reset = server.gui.add_button("Reset gizmo to current EE")
gui_speed = server.gui.add_slider("Speed (unified)", min=0.1, max=1.0, step=0.05, initial_value=1.0)
gui_execute = server.gui.add_checkbox("Execute on robot (EGM stream)", initial_value=False)
gui_live = server.gui.add_checkbox("Live (drive robot)", initial_value=False)
gui_traj_name = server.gui.add_text("Trajectory name", initial_value="traj1")
gui_save = server.gui.add_button("Save trajectory")
gui_load = server.gui.add_button("Load trajectory")


def _refresh_wp_count() -> None:
    gui_wp_count.value = str(len(waypoints))


def _add_waypoint(wp: dict) -> None:
    i = len(waypoints)
    waypoints.append(wp)
    handle = server.scene.add_frame(
        f"/waypoints/{i}",
        position=np.asarray(wp["pos"]), wxyz=np.asarray(wp["wxyz"]),
        axes_length=0.12, axes_radius=0.005,
    )
    waypoint_frames.append(handle)
    _refresh_wp_count()


@gui_add_wp.on_click
def _(_):
    _add_waypoint({
        "q": None,  # filled by IK at Plan
        "pos": np.asarray(gizmo.position).tolist(),
        "wxyz": np.asarray(gizmo.wxyz).tolist(),
    })
    gui_status.value = f"Added waypoint {len(waypoints)} (gizmo)"


@gui_capture.on_click
def _(_):
    with state_lock:
        q = current_q.copy()
    T = ee_pose(q)
    _add_waypoint({
        "q": q.tolist(),  # taught joints (hand-guide via the GoFa lead-through button) — replayed exactly
        "pos": np.asarray(T.translation()).tolist(),
        "wxyz": np.asarray(T.rotation().wxyz).tolist(),
    })
    gui_status.value = f"Captured waypoint {len(waypoints)} (joints)"


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
    if waypoints:
        targets = waypoints
    else:
        targets = [{"q": None, "pos": np.asarray(gizmo.position).tolist(),
                    "wxyz": np.asarray(gizmo.wxyz).tolist()}]
    with state_lock:
        q = current_q.copy()
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for i, wp in enumerate(targets):
        if wp.get("q") is not None:
            q_next = np.asarray(wp["q"], dtype=np.float64)  # taught joints, replay exactly
        else:
            try:
                q_next = np.asarray(pks.solve_ik_seeded(
                    robot=robot, target_link_name=TARGET_LINK,
                    target_position=np.asarray(wp["pos"]), target_wxyz=np.asarray(wp["wxyz"]),
                    q_seed=q, rest_weight=2.0,
                ))
            except Exception as e:
                gui_status.value = f"IK failed at waypoint {i + 1}: {e}"
                return
            wp["q"] = q_next.tolist()  # backfill so a save-after-plan carries joints too
        segments.append((q.copy(), q_next))
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
    gui_live.disabled = True
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
            seg_duration = _cap_seg_duration(q_start, delta, seg_duration, dt)
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
        gui_live.disabled = False
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
    live.clear()
    if gui_live.value:
        gui_live.value = False
    gui_plan.disabled = False


def _live_loop() -> None:
    """Continuously chase the gizmo over EGM: solve IK each tick and stream the
    target, with a per-tick joint clamp AND a TCP-speed clamp (the GoFa collaborative
    limit). Re-arms the EGM session if RAPID's CondTime drops it during a pause —
    so no supervisor change is needed."""
    dt = 1.0 / LIVE_HZ
    if not _start_egm_session():
        live.clear()
        gui_live.value = False
        return
    gui_egm_status.value = "streaming (live)"
    with state_lock:
        q_cmd = current_q.copy()
    try:
        while live.is_set() and not stop_flag.is_set():
            # re-arm if the controller dropped the session (CondTime exit on a pause)
            if not egm.is_fresh(max_age_s=0.3):
                gui_egm_status.value = "re-arming..."
                if not _start_egm_session():
                    break
                gui_egm_status.value = "streaming (live)"
                with state_lock:
                    q_cmd = current_q.copy()
            try:
                q_target = np.asarray(pks.solve_ik_seeded(
                    robot=robot, target_link_name=TARGET_LINK,
                    target_position=np.asarray(gizmo.position),
                    target_wxyz=np.asarray(gizmo.wxyz),
                    q_seed=q_cmd, rest_weight=2.0,
                ))
            except Exception:
                time.sleep(dt)
                continue
            dq = np.clip(q_target - q_cmd, -MAX_JOINT_SPEED * dt, MAX_JOINT_SPEED * dt)
            # cap real TCP speed (collaborative limit) by scaling the step down
            p0 = np.asarray(ee_pose(q_cmd).translation())
            p1 = np.asarray(ee_pose(q_cmd + dq).translation())
            d = float(np.linalg.norm(p1 - p0))
            if d > MAX_TCP_SPEED * dt and d > 1e-9:
                dq *= (MAX_TCP_SPEED * dt) / d
            q_cmd = q_cmd + dq
            viser_urdf.update_cfg(q_cmd)
            egm.set_target_rad(q_cmd.tolist())
            time.sleep(dt)
    finally:
        # hold the last pose so EGMRunJoint converges (CondTime) and the session exits
        for _ in range(int(HOLD_AFTER_PLAY_S * STREAM_HZ)):
            egm.set_target_rad(q_cmd.tolist())
            time.sleep(1.0 / STREAM_HZ)
        _wait_egm_clear()
        gui_egm_status.value = "idle"
        gui_status.value = "Live off"
        gui_plan.disabled = False
        gui_play.disabled = plan_segments is None
        gui_stop.disabled = True


@gui_live.on_update
def _(_):
    if gui_live.value:
        # snap gizmo to the current EE first (no lurch), then chase it
        with state_lock:
            q = current_q.copy()
        T = ee_pose(q)
        gizmo.position = np.asarray(T.translation())
        gizmo.wxyz = np.asarray(T.rotation().wxyz)
        stop_flag.clear()
        live.set()
        gui_plan.disabled = True
        gui_play.disabled = True
        gui_stop.disabled = False
        gui_status.value = "LIVE: GoFa follows gizmo"
        threading.Thread(target=_live_loop, daemon=True).start()
    else:
        live.clear()


@gui_save.on_click
def _(_):
    os.makedirs(TRAJ_DIR, exist_ok=True)
    name = (gui_traj_name.value or "traj").strip()
    path = os.path.join(TRAJ_DIR, f"{name}.json")
    data = {
        "robot": "gofa",
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
                ok = last_poll_ok
            viser_urdf.update_cfg(q)
            gui_rws_status.value = "OK" if ok else "DISCONNECTED"
            gui_egm_status.value = (
                f"rx={egm.packets_rx} tx={egm.packets_tx}"
                if egm.has_feedback() else "idle (no packets)"
            )
        time.sleep(period)


threading.Thread(target=viz_loop, daemon=True).start()

print("Warming up IK solver (JAX JIT compile)...")
_warmup_ik()

print("viser running. Open the URL printed above in a browser.")
print(f"RWS target: https://{ROBOT_IP}  user={RWS_USER!r}")
print(f"EGM local port: {EGM_LOCAL_PORT}")
try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    print("\nShutting down — stopping EGM and releasing mastership...")
    # Stop any active play/live, tell the daemon loops to quit, then deliberately
    # clear egm_go so the robot stops chasing the dead UDP stream (instead of
    # waiting out RAPID's \CondTime), and close the EGM socket. Mastership is
    # also released by abb_rws' atexit hook.
    shutdown.set()
    stop_flag.set()
    time.sleep(0.2)
    for fn in (
        lambda: rws.set_rapid_bool(RAPID_GO_FLAG_VAR, False, module=RAPID_MODULE),
        egm.stop,
        rws.release_mastership,
    ):
        try:
            fn()
        except Exception:
            pass
    print("EGM stopped.")
    os._exit(0)
