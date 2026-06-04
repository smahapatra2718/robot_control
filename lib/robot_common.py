"""Shared config + pure helpers for the teleop scaffold.

Single source of truth for the values and small functions that used to be
hand-mirrored across teleop_ur15.py, teleop_gofa_egm.py, play_trajectory.py and
teleop.py (each carried a "keep in sync" comment). Retune a number here and every
script picks it up.

Deliberately stdlib-only (os/json/datetime) so importing it never pulls in jax,
pyroki, yourdfpy or ur_rtde — the heavy/robot-specific imports stay in the
scripts that actually need them. Forward kinematics also stays per-script: it
needs jax/jaxlie/pyroki, and teleop.py / play_trajectory.py import those lazily
to defer the ~800 ms JIT cost, which a shared FK helper would undermine.
"""

from __future__ import annotations

import datetime
import json
import os

# Repo root = parent of lib/ (this module lives in lib/). Asset paths below
# resolve against the root from __file__, independent of the caller's CWD.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Saved teach trajectories: trajectories/<name>.json (shared by every script).
TRAJ_DIR = os.path.join(_ROOT, "trajectories")
TARGET_LINK = "tool0"            # FK link both arms target

# ---- trapezoidal play profile (shared by all four scripts) ----
RAMP_FRAC = 0.25                 # fraction of a segment spent ramping up (same for ramp-down)
MIN_SEG_DURATION_S = 0.5         # floor on segment time so tiny moves stay smooth
DWELL_S = 0.2                    # pause at each intermediate waypoint
GRIP_PREDELAY_S = 0.5            # settle the arm this long before actuating the gripper
GRIP_EPS = 0.02                  # min change in gripper fraction (2%) before a waypoint re-actuates

# ---- UR15 (RTDE / servoJ + Hand-E gripper) ----
UR_ROBOT_IP = "192.168.125.2"
UR_ROBOT_DESCRIPTION = "ur15_description"   # robot_descriptions loader key
UR_STREAM_HZ = 50                # servoJ + viz playback rate
UR_MAX_JOINT_SPEED = 1.0         # rad/s peak per joint at slider=1.0
UR_SERVO_LOOKAHEAD = 0.1         # servoJ lookahead_time (s)
UR_SERVO_GAIN = 300              # servoJ gain during motion
UR_SERVO_STOP_DECEL = 2.0        # rad/s^2 at end-of-trajectory servoStop (default 10 is harsh)
UR_SETTLE_GAIN = 600             # stiffer servoJ gain for the static end-of-play hold
UR_SETTLE_EPS_RAD = 0.00002      # min per-check improvement to count as "still converging"
UR_SETTLE_STALL_TICKS = 10       # consecutive non-improving checks => at the servoJ floor, stop
UR_SETTLE_MAX_S = 3.0            # hard cap on the final convergence hold
# Hand-E gripper geometry / payload
UR_GRIPPER_URDF_PATH = os.path.join(_ROOT, "urdf", "hande.urdf")
UR_MESH_DIR_PREFIX = _ROOT       # hande.urdf package://robotiq_hande_description meshes resolve under the project root
UR_GRASP_LINK = "robotiq_hande_end"   # grasp point; fixed offset from tool0 (gripper is rigid)
UR_GRIPPER_FINGER_OPEN = 0.025   # per-side finger travel (m) = URDF upper limit (open)
UR_GRIPPER_MASS = 1.0            # Hand-E payload (kg) told to the UR so it compensates gravity
UR_GRIPPER_COG = (0.0, 0.0, 0.06)  # payload center of gravity in the tool-flange frame (m)

# ---- GoFa CRB 15000 (EGM joint stream + RWS) ----
GOFA_ROBOT_IP = "192.168.125.1"
GOFA_RWS_USER = "Default User"
GOFA_RWS_PASSWORD = "robotics"
GOFA_URDF_PATH = os.path.join(_ROOT, "urdf", "crb15000_5_95.urdf")
GOFA_MESH_DIR_PREFIX = os.path.join(_ROOT, "abb_desc")
GOFA_RAPID_MODULE = "PyEgm"      # supervisor module loaded by install_gofa_egm.py
GOFA_RAPID_GO_FLAG = "egm_go"    # bool in PyEgm.mod: TRUE -> enter EGMRunJoint
GOFA_RAPID_LEAD_FLAG = "lead_go"  # bool in PyEgm.mod: TRUE -> SetLeadThrough (hand-guide)
GOFA_EGM_LOCAL_PORT = 6510       # UDP port; must match RemotePortNumber in EGM_COMM.cfg
GOFA_STREAM_HZ = 100             # EGM target stream rate
GOFA_MAX_JOINT_SPEED = 1.0       # rad/s peak per joint at slider=1.0 (before the TCP cap)
GOFA_MAX_TCP_SPEED = 0.25        # m/s hard cap on real tool speed (ISO/TS 15066 collaborative limit)
GOFA_HOLD_AFTER_PLAY_S = 1.5     # hold final target this long so RAPID's \CondTime triggers


def alpha_to_s(alpha: float, r: float = RAMP_FRAC) -> float:
    """Trapezoidal velocity profile: parabolic accel, linear cruise, parabolic decel.
    Maps alpha in [0,1] to traversed fraction s in [0,1] with s'(0)=s'(1)=0.
    r is the ramp fraction (0 < r <= 0.5)."""
    v_peak = 1.0 / (1.0 - r)  # cruise speed making total area = 1
    if alpha < r:
        return 0.5 * v_peak * alpha * alpha / r
    if alpha < 1.0 - r:
        return 0.5 * v_peak * r + v_peak * (alpha - r)
    return 1.0 - 0.5 * v_peak * (1.0 - alpha) ** 2 / r


def norm_grip(g):
    """Waypoint grip: legacy 'open'/'close'/None or a numeric fraction -> float or None.
    Fraction is 0.0=open .. 1.0=fully closed."""
    if g is None:
        return None
    if g == "open":
        return 0.0
    if g == "close":
        return 1.0
    return float(g)


def make_mesh_resolver(prefix: str):
    """Return a yourdfpy filename_handler that rewrites package://pkg/rest to
    prefix/pkg/rest (and passes anything else through unchanged)."""
    def _resolve(fname: str) -> str:
        if fname.startswith("package://"):
            pkg, rest = fname[len("package://"):].split("/", 1)
            return os.path.join(prefix, pkg, rest)
        return fname
    return _resolve


def save_trajectory(name: str, robot: str, waypoints: list, traj_dir: str = TRAJ_DIR) -> str:
    """Write trajectories/<name>.json in the shared format. Returns the path."""
    os.makedirs(traj_dir, exist_ok=True)
    path = os.path.join(traj_dir, f"{name}.json")
    data = {
        "robot": robot,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "waypoints": waypoints,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_trajectory(name: str, traj_dir: str = TRAJ_DIR) -> dict:
    """Read and parse trajectories/<name>.json (no validation)."""
    with open(os.path.join(traj_dir, f"{name}.json")) as f:
        return json.load(f)
