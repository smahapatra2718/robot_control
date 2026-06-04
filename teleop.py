"""teleop.py — CLI trajectory recorder for the UR15 or GoFa (no viser).

Hand-guide the arm in free-drive and capture waypoints with single keypresses,
saving in the same trajectories/<name>.json format play_trajectory.py replays.
This is the record-side counterpart to play_trajectory.py.

  ./robot_control/bin/python teleop.py [name] [--robot ur|gofa]

If name/robot are omitted you're prompted for them. Then the arm enters free-drive
(UR: teachMode; GoFa: software lead-through) and the key map is:

  c       capture a waypoint (joints + FK grasp pose + gripper fraction)
  up/down UR only: open / close the gripper one step (10%), commanded live
  Enter   end free-drive, save the trajectory, prompt for the next one
  Esc     soft stop: end free-drive cleanly and exit
  q       hard stop: protective stop (UR) / stop RAPID + drop lead-through (GoFa), exit

A blank name + Enter at the name prompt exits the script.
"""

import argparse
import datetime
import json
import os
import select
import sys
import termios
import time
import tty

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
TRAJ_DIR = os.path.join(_HERE, "trajectories")

GRIP_STEP = 0.10                # gripper fraction change per up/down press (UR)

# ---- UR15 (mirror play_trajectory.py / teleop_ur15.py) ----
UR_ROBOT_IP = "192.168.125.2"
UR_GRIPPER_FINGER_OPEN = 0.025  # per-side finger travel (m) = URDF "open" limit
UR_GRIPPER_MASS = 1.0
UR_GRIPPER_COG = (0.0, 0.0, 0.06)

# ---- GoFa (mirror play_trajectory.py / teleop_gofa_egm.py) ----
GOFA_ROBOT_IP = "192.168.125.1"
GOFA_RWS_USER = "Default User"
GOFA_RWS_PASSWORD = "robotics"
GOFA_RAPID_MODULE = "PyEgm"
GOFA_RAPID_LEAD_FLAG = "lead_go"
GOFA_RAPID_GO_FLAG = "egm_go"
GOFA_URDF_PATH = os.path.join(_HERE, "crb15000_5_95.urdf")
GOFA_MESH_DIR_PREFIX = os.path.join(_HERE, "abb_desc")


# ---------------- keyboard (raw single-key) ----------------
class raw_mode:
    """Put the TTY into cbreak (no line buffering, no echo) for the key loop;
    restore on exit. Ctrl-C still raises KeyboardInterrupt (ISIG stays on)."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


def read_key() -> str:
    """Block for one keypress; return 'c','q','enter','esc','up','down', or the
    raw character. Arrow keys arrive as ESC [ A/B; a bare Esc has nothing
    following within ~50 ms, which is how we tell them apart."""
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        r, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not r:
            return "esc"
        if sys.stdin.read(1) == "[":
            return {"A": "up", "B": "down"}.get(sys.stdin.read(1), "other")
        return "esc"
    if ch in ("\r", "\n"):
        return "enter"
    return ch


# ---------------- backends ----------------
class URBackend:
    robot_name = "ur15"

    def __init__(self):
        import jax.numpy as jnp
        import jaxlie
        import pyroki as pk
        import yourdfpy
        from robot_descriptions.loaders.yourdfpy import load_robot_description
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        import hande_gripper

        self._jnp, self._jaxlie = jnp, jaxlie

        urdf = load_robot_description("ur15_description")
        self._robot = pk.Robot.from_urdf(urdf)
        self._tcp = self._robot.links.names.index("tool0")

        # Fixed tool0 -> grasp-point offset, read from the Hand-E URDF (gripper rigid).
        def _resolve(fname):
            if fname.startswith("package://"):
                pkg, rest = fname[len("package://"):].split("/", 1)
                return os.path.join(_HERE, pkg, rest)
            return fname

        g_urdf = yourdfpy.URDF.load(os.path.join(_HERE, "hande.urdf"), filename_handler=_resolve)
        g_urdf.update_cfg(np.array([UR_GRIPPER_FINGER_OPEN]))
        self._tool0_T_grasp = jaxlie.SE3.from_matrix(
            jnp.asarray(g_urdf.get_transform("robotiq_hande_end", "tool0"))
        )

        print(f"Connecting to UR15 at {UR_ROBOT_IP} ...")
        self._r = RTDEReceiveInterface(UR_ROBOT_IP)
        self._c = RTDEControlInterface(UR_ROBOT_IP)
        try:
            self._c.setPayload(UR_GRIPPER_MASS, list(UR_GRIPPER_COG))
        except Exception as e:
            print(f"setPayload failed ({e}).")

        # Gripper best-effort: connect + activate + WAIT for calibration so the
        # arrow keys command a calibrated gripper. Unreachable -> arrows are no-ops.
        self.grip = 0.0
        try:
            self._gripper = hande_gripper.HandEGripper(UR_ROBOT_IP, hande_gripper.DEFAULT_PORT)
            self._gripper.connect()
            print("Activating Hand-E; waiting for calibration ...")
            self._gripper.reset(timeout=5.0)
            self._gripper.activate(timeout=20.0)
            self._gripper.open()
            self._gripper.wait_until_idle(timeout=10.0)
            print("Hand-E calibrated + open.")
        except Exception as e:
            self._gripper = None
            print(f"Hand-E unreachable ({e}); gripper keys disabled.")

        self._warmup()

    def _warmup(self):
        # Pay the JAX FK JIT compile now so the first capture isn't slow.
        try:
            self.grasp_pose(self.read_joints())
        except Exception:
            pass

    def grasp_pose(self, q):
        Ts = self._robot.forward_kinematics(cfg=self._jnp.array(q))
        T = self._jaxlie.SE3(Ts[self._tcp]).multiply(self._tool0_T_grasp)
        return np.asarray(T.translation()).tolist(), np.asarray(T.rotation().wxyz).tolist()

    def read_joints(self):
        return np.asarray(self._r.getActualQ(), dtype=np.float64)

    def capture(self):
        q = self.read_joints()
        pos, wxyz = self.grasp_pose(q)
        return {"q": q.tolist(), "pos": pos, "wxyz": wxyz, "grip": self.grip}

    def adjust_grip(self, delta):
        if self._gripper is None:
            return None
        self.grip = max(0.0, min(1.0, self.grip + delta))
        try:
            self._gripper.move(self.grip)
        except Exception as e:
            print(f"  gripper cmd failed: {e}")
        return self.grip

    def start_freedrive(self):
        self._c.teachMode()

    def stop_freedrive(self):
        try:
            self._c.endTeachMode()
        except Exception:
            pass

    def hard_stop(self):
        self.stop_freedrive()
        try:
            self._c.triggerProtectiveStop()
            print("  *** PROTECTIVE STOP — clear it on the pendant ***")
        except Exception as e:
            print(f"  triggerProtectiveStop failed ({e}); stopping script.")
        try:
            self._c.stopScript()
        except Exception:
            pass

    def close(self):
        self.stop_freedrive()
        for fn in (self._c.stopScript, self._c.disconnect, self._r.disconnect,
                   (self._gripper.close if self._gripper is not None else (lambda: None))):
            try:
                fn()
            except Exception:
                pass


class GoFaBackend:
    robot_name = "gofa"

    def __init__(self):
        import jax.numpy as jnp
        import jaxlie
        import pyroki as pk
        import yourdfpy

        import abb_rws

        self._jnp, self._jaxlie = jnp, jaxlie

        def _resolve(fname):
            if fname.startswith("package://"):
                pkg, rest = fname[len("package://"):].split("/", 1)
                return os.path.join(GOFA_MESH_DIR_PREFIX, pkg, rest)
            return fname

        urdf = yourdfpy.URDF.load(GOFA_URDF_PATH, filename_handler=_resolve)
        self._robot = pk.Robot.from_urdf(urdf)
        self._tcp = self._robot.links.names.index("tool0")

        print(f"Connecting to GoFa RWS at {GOFA_ROBOT_IP} ...")
        self._rws = abb_rws.RWSClient(
            host=GOFA_ROBOT_IP, user=GOFA_RWS_USER, password=GOFA_RWS_PASSWORD
        )
        self._rws.request_mastership()
        # Safety: both flags FALSE so a stray TRUE doesn't fire EGM or lead-through.
        self._rws.set_rapid_bool(GOFA_RAPID_GO_FLAG, False, module=GOFA_RAPID_MODULE)
        self._rws.set_rapid_bool(GOFA_RAPID_LEAD_FLAG, False, module=GOFA_RAPID_MODULE)
        self._warmup()

    def _warmup(self):
        try:
            self.grasp_pose(self.read_joints())
        except Exception:
            pass

    def grasp_pose(self, q):
        Ts = self._robot.forward_kinematics(cfg=self._jnp.array(q))
        T = self._jaxlie.SE3(Ts[self._tcp])
        return np.asarray(T.translation()).tolist(), np.asarray(T.rotation().wxyz).tolist()

    def read_joints(self):
        return np.asarray(self._rws.get_joints(), dtype=np.float64)

    def capture(self):
        q = self.read_joints()
        pos, wxyz = self.grasp_pose(q)
        return {"q": q.tolist(), "pos": pos, "wxyz": wxyz}

    def adjust_grip(self, delta):
        return None   # GoFa has no gripper

    def start_freedrive(self):
        # lead_go = TRUE -> PyEgm.mod calls SetLeadThrough \On (hand-guiding).
        self._rws.set_rapid_bool(GOFA_RAPID_GO_FLAG, False, module=GOFA_RAPID_MODULE)
        self._rws.set_rapid_bool(GOFA_RAPID_LEAD_FLAG, True, module=GOFA_RAPID_MODULE)

    def stop_freedrive(self):
        try:
            self._rws.set_rapid_bool(GOFA_RAPID_LEAD_FLAG, False, module=GOFA_RAPID_MODULE)
        except Exception:
            pass

    def hard_stop(self):
        # No reliable software motors-off over RWS; stopping the RAPID program
        # halts all motion and the supervisor, and clearing the flags drops
        # lead-through. Effective protective stop.
        for fn in (
            lambda: self._rws.set_rapid_bool(GOFA_RAPID_LEAD_FLAG, False, module=GOFA_RAPID_MODULE),
            lambda: self._rws.set_rapid_bool(GOFA_RAPID_GO_FLAG, False, module=GOFA_RAPID_MODULE),
            self._rws.stop_program,
        ):
            try:
                fn()
            except Exception:
                pass
        print("  *** STOPPED RAPID + dropped lead-through — re-run install/Play to resume ***")

    def close(self):
        self.stop_freedrive()
        try:
            self._rws.release_mastership()
        except Exception:
            pass


def make_backend(choice: str):
    return URBackend() if choice == "ur" else GoFaBackend()


# ---------------- prompts ----------------
def prompt_name() -> str:
    return input("Trajectory name (blank to exit): ").strip()


def prompt_robot() -> str | None:
    while True:
        sel = input("Robot?  [1] UR  [2] GoFa: ").strip()
        if sel == "":
            return None
        if sel == "1":
            return "ur"
        if sel == "2":
            return "gofa"
        print("  enter 1 or 2 (or blank to exit)")


def save_traj(name: str, robot: str, waypoints: list) -> None:
    os.makedirs(TRAJ_DIR, exist_ok=True)
    path = os.path.join(TRAJ_DIR, f"{name}.json")
    data = {
        "robot": robot,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "waypoints": waypoints,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  saved {len(waypoints)} waypoint(s) -> trajectories/{name}.json")


# ---------------- recording loop ----------------
def key_loop(backend, waypoints: list) -> str:
    """Free-drive key loop. Returns 'save', 'soft', or 'hard'."""
    with raw_mode():
        while True:
            k = read_key()
            if k == "c":
                waypoints.append(backend.capture())
                print(f"  captured waypoint {len(waypoints)}")
            elif k == "up":      # open
                f = backend.adjust_grip(-GRIP_STEP)
                if f is not None:
                    print(f"  gripper {int(round(f * 100))}% closed")
            elif k == "down":    # close
                f = backend.adjust_grip(+GRIP_STEP)
                if f is not None:
                    print(f"  gripper {int(round(f * 100))}% closed")
            elif k == "enter":
                return "save"
            elif k == "esc":
                return "soft"
            elif k == "q":
                backend.hard_stop()
                return "hard"


def record_session(backend, first_name: str) -> None:
    name = first_name
    try:
        while True:
            if name is None:
                name = prompt_name()
            if not name:
                print("No name — exiting.")
                return
            waypoints: list = []
            print(f"\nRecording '{name}' — FREE-DRIVE ON. Move the arm by hand.")
            print("  c=capture   up/down=gripper   Enter=save   Esc=stop   q=E-STOP")
            backend.start_freedrive()
            try:
                action = key_loop(backend, waypoints)
            finally:
                backend.stop_freedrive()

            if action in ("hard", "soft"):
                return
            if waypoints:
                save_traj(name, backend.robot_name, waypoints)
            else:
                print("  no waypoints captured — nothing saved.")
            name = None
    finally:
        backend.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="CLI trajectory recorder (UR15 / GoFa).")
    ap.add_argument("name", nargs="?", help="trajectory name (prompted if omitted)")
    ap.add_argument("--robot", choices=["ur", "gofa"], help="robot (prompted if omitted)")
    args = ap.parse_args()

    name = args.name
    if name is None:
        name = prompt_name()
    if not name:
        print("No name — exiting.")
        return

    robot_choice = args.robot or prompt_robot()
    if robot_choice is None:
        print("No robot selected — exiting.")
        return

    backend = make_backend(robot_choice)
    try:
        record_session(backend, name)
    except KeyboardInterrupt:
        print("\nInterrupted — releasing free-drive and disconnecting.")
        backend.close()
    sys.stdout.flush()
    os._exit(0)   # avoid ur_rtde boost-thread teardown crash on a clean exit


if __name__ == "__main__":
    main()
