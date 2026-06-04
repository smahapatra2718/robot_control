"""teleop.py — CLI trajectory recorder for the UR15 or GoFa (no viser).

Hand-guide the arm in free-drive and capture waypoints with single keypresses,
saving in the same trajectories/<name>.json format play_trajectory.py replays.
This is the record-side counterpart to play_trajectory.py.

  ./robot_control/bin/python teleop.py [name] [--robot ur|gofa]

If name/robot are omitted you're prompted for them. Then the arm enters free-drive
(UR: teachMode; GoFa: software lead-through) and the key map is:

  c       capture a waypoint (joints + FK grasp pose + gripper fraction)
  o / p   UR only: open / close the gripper one step (10%), commanded live
  Enter   end free-drive, save the trajectory, prompt for the next one
  w       soft stop: end free-drive cleanly and exit
  q       hard stop: protective stop (UR) / stop RAPID + drop lead-through (GoFa), exit

A live terminal dashboard shows the grasp pose, joint angles, and gripper % while
recording. A blank name + Enter at the name prompt exits the script.
"""

import argparse
import os
import select
import sys
import termios
import time
import tty

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import robot_common as rc  # noqa: E402
from robot_common import (  # noqa: E402
    TRAJ_DIR, TARGET_LINK,
    UR_ROBOT_IP, UR_ROBOT_DESCRIPTION, UR_GRASP_LINK, UR_GRIPPER_URDF_PATH,
    UR_MESH_DIR_PREFIX, UR_GRIPPER_FINGER_OPEN, UR_GRIPPER_MASS, UR_GRIPPER_COG,
    GOFA_ROBOT_IP, GOFA_RWS_USER, GOFA_RWS_PASSWORD, GOFA_RAPID_MODULE,
    GOFA_RAPID_LEAD_FLAG, GOFA_RAPID_GO_FLAG, GOFA_URDF_PATH, GOFA_MESH_DIR_PREFIX,
)

GRIP_STEP = 0.10                # gripper fraction change per o/p press (UR)
DASH_HZ = 10                    # live dashboard refresh + key-poll rate


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


def quat_to_rpy(wxyz) -> np.ndarray:
    """Quaternion (w,x,y,z) -> roll/pitch/yaw in degrees (for the dashboard)."""
    w, x, y, z = wxyz
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


# ---------------- live dashboard ----------------
def build_lines(backend, name, q, pos, wxyz, count, status) -> list[str]:
    rpy = quat_to_rpy(wxyz)
    jdeg = np.degrees(np.asarray(q))
    joints = "  ".join(f"{v:+6.1f}" for v in jdeg)
    sep = "─" * 56
    return [
        f" teleop · {backend.robot_name} · '{name}'  —  FREE-DRIVE, hand-guide the arm",
        f" {sep}",
        f" Pose    X {pos[0] * 1000:+8.1f}   Y {pos[1] * 1000:+8.1f}   Z {pos[2] * 1000:+8.1f}   mm",
        f"         R {rpy[0]:+8.1f}   P {rpy[1]:+8.1f}   Y {rpy[2]:+8.1f}   deg",
        f" Joints  {joints}   deg",
        f" Gripper {backend.grip_text()}",
        f" Points  {count} captured",
        f" {sep}",
        f" c capture    o/p open/close    Enter save    w stop    q E-STOP",
        f" > {status}",
    ]


def render(lines: list[str], first: bool) -> None:
    """Redraw the block in place: move the cursor up to its top (except the first
    time), then rewrite each line clearing to end-of-line. Line count is constant."""
    buf = [] if first else [f"\033[{len(lines)}A"]
    for ln in lines:
        buf.append("\r\033[K" + ln + "\n")
    sys.stdout.write("".join(buf))
    sys.stdout.flush()


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

        urdf = load_robot_description(UR_ROBOT_DESCRIPTION)
        self._robot = pk.Robot.from_urdf(urdf)
        self._tcp = self._robot.links.names.index(TARGET_LINK)

        # Fixed tool0 -> grasp-point offset, read from the Hand-E URDF (gripper rigid).
        g_urdf = yourdfpy.URDF.load(
            UR_GRIPPER_URDF_PATH, filename_handler=rc.make_mesh_resolver(UR_MESH_DIR_PREFIX)
        )
        g_urdf.update_cfg(np.array([UR_GRIPPER_FINGER_OPEN]))
        self._tool0_T_grasp = jaxlie.SE3.from_matrix(
            jnp.asarray(g_urdf.get_transform(UR_GRASP_LINK, TARGET_LINK))
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
        return np.asarray(T.translation()), np.asarray(T.rotation().wxyz)

    def read_joints(self):
        return np.asarray(self._r.getActualQ(), dtype=np.float64)

    def make_waypoint(self, q):
        pos, wxyz = self.grasp_pose(q)
        return {"q": q.tolist(), "pos": pos.tolist(), "wxyz": wxyz.tolist(), "grip": self.grip}

    def grip_text(self):
        s = f"{int(round(self.grip * 100))}% closed"
        return s if self._gripper is not None else s + "   (gripper offline)"

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

        urdf = yourdfpy.URDF.load(
            GOFA_URDF_PATH, filename_handler=rc.make_mesh_resolver(GOFA_MESH_DIR_PREFIX)
        )
        self._robot = pk.Robot.from_urdf(urdf)
        self._tcp = self._robot.links.names.index(TARGET_LINK)

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
        return np.asarray(T.translation()), np.asarray(T.rotation().wxyz)

    def read_joints(self):
        return np.asarray(self._rws.get_joints(), dtype=np.float64)

    def make_waypoint(self, q):
        pos, wxyz = self.grasp_pose(q)
        return {"q": q.tolist(), "pos": pos.tolist(), "wxyz": wxyz.tolist()}

    def grip_text(self):
        return "n/a (no gripper)"

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
    rc.save_trajectory(name, robot, waypoints)
    print(f"  saved {len(waypoints)} waypoint(s) -> trajectories/{name}.json")


# ---------------- recording loop ----------------
def key_loop(backend, name: str, waypoints: list) -> str:
    """Free-drive loop: refresh the live dashboard at DASH_HZ and poll for a key
    each tick (non-blocking, so the dashboard keeps updating). Returns 'save',
    'soft', or 'hard'."""
    status = "ready"
    first = True
    last_q = backend.read_joints()
    sys.stdout.write("\033[?25l")   # hide cursor while the dashboard owns the screen
    try:
        with raw_mode():
            while True:
                try:
                    q = backend.read_joints()
                    last_q = q
                except Exception:
                    q = last_q          # network/RTDE blip: hold the last reading
                pos, wxyz = backend.grasp_pose(q)
                render(build_lines(backend, name, q, pos, wxyz, len(waypoints), status), first)
                first = False

                r, _, _ = select.select([sys.stdin], [], [], 1.0 / DASH_HZ)
                if not r:
                    continue
                k = sys.stdin.read(1)
                if k == "c":
                    waypoints.append(backend.make_waypoint(q))
                    status = f"captured waypoint {len(waypoints)}"
                elif k == "o":      # open one step (live % shows on the Gripper line)
                    status = "no gripper" if backend.adjust_grip(-GRIP_STEP) is None else "opening gripper"
                elif k == "p":      # close one step
                    status = "no gripper" if backend.adjust_grip(+GRIP_STEP) is None else "closing gripper"
                elif k in ("\r", "\n"):
                    return "save"
                elif k == "w":
                    return "soft"
                elif k == "q":
                    backend.hard_stop()
                    return "hard"
    finally:
        sys.stdout.write("\033[?25h\n")   # restore cursor, drop below the dashboard
        sys.stdout.flush()


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
            backend.start_freedrive()
            try:
                action = key_loop(backend, name, waypoints)
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
