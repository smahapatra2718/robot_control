"""teleop.py — CLI trajectory recorder for the UR15 or GoFa (no viser).

Hand-guide the arm in free-drive and capture waypoints with single keypresses,
saving in the same trajectories/<name>.json format play_trajectory.py replays.
This is the record-side counterpart to play_trajectory.py.

  ./robot_control/bin/python scripts/teleop.py [name] [--robot ur|gofa]

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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "lib")):  # repo root (pyroki_snippets) + lib/ (our modules)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import robot_common as rc  # noqa: E402
from control import make_controller  # noqa: E402

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


# ---------------- backend ----------------
class Backend:
    def __init__(self, choice: str):
        self.robot_name = "ur15" if choice == "ur" else "gofa"
        print(f"Connecting to {self.robot_name} ...")
        self._c = make_controller(self.robot_name)
        self._c.connect()

    def read_joints(self):
        return np.asarray(self._c.get_state().q, dtype=np.float64)

    def grasp_pose(self, q):
        pos, wxyz = self._c.grasp_pose(q)
        return np.asarray(pos), np.asarray(wxyz)

    def make_waypoint(self, q):
        pos, wxyz = self.grasp_pose(q)
        wp = {"q": q.tolist(), "pos": pos.tolist(), "wxyz": wxyz.tolist()}
        frac = self._c.get_state().gripper_frac
        if frac is not None:
            wp["grip"] = frac
        return wp

    def grip_text(self):
        frac = self._c.get_state().gripper_frac
        if frac is None:
            return "n/a (no gripper)"
        return f"{int(round(frac * 100))}% closed"

    def adjust_grip(self, delta):
        return self._c.adjust_grip(delta)

    def start_freedrive(self):
        self._c.start_freedrive()

    def stop_freedrive(self):
        self._c.stop_freedrive()

    def hard_stop(self):
        self.stop_freedrive()
        self._c.estop()
        # close() (always run by record_session's finally) follows with stopScript/teardown
        if self.robot_name == "gofa":
            print("  *** STOPPED RAPID + dropped lead-through — re-run install/Play to resume ***")
        else:
            print("  *** PROTECTIVE STOP — clear it on the pendant ***")

    def close(self):
        self.stop_freedrive()
        self._c.close()


def make_backend(choice: str):
    return Backend(choice)


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
