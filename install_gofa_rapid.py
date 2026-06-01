"""
One-time setup: upload PyExec.mod to the GoFa OmniCore controller via RWS and
start it. Run this once. After that, just run teleop_gofa.py.

What it does, step by step:
  1. Connect to RWS at ROBOT_IP.
  2. Print controller state, operation mode, current execution state.
  3. Stop any running program (safe — robot won't move).
  4. Upload PyExec.mod into the controller's HOME/ directory.
  5. Load PyExec.mod as a module in task T_ROB1.
  6. Reset program pointer to main.
  7. Turn motors on.
  8. Start the program. It will park at WaitUntil py_go.

Prerequisites you do on the GoFa side BEFORE running this:
  A. Controller powered on, robot in Normal state.
  B. Key switch on the cabinet in AUTO (top position). Motors-on attempts
     from RWS will fail in Manual mode.
  C. Find the GoFa's IP address: pendant Settings -> Network. Set ROBOT_IP
     below (or in teleop_gofa.py + abb_rws.py).
  D. RWS services enabled on the controller (typically default on OmniCore).
     If you get connect refused, check pendant Settings -> Security ->
     Services and enable "Robot Web Services".

After this script succeeds, `teleop_gofa.py` will be able to move the robot
when Execute is checked.
"""

import sys
import time

import abb_rws

ROBOT_IP = "192.168.125.1"          # GoFa MGMT port (direct Mac connection)
RWS_USER = "Default User"
RWS_PASSWORD = "robotics"

MODULE_NAME = "PyExec"
MODULE_FILENAME = "PyExec.mod"

PYEXEC_MOD = """\
MODULE PyExec
  VAR jointtarget py_target := [[0,0,0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];
  PERS bool       py_go     := FALSE;
  CONST speeddata v_tcp     := v200;

  PROC main()
    ! AccSet <accel %>, <ramp %>. Lower = smoother start/stop, slower overall.
    ! 50/50 is a calm research-lab default; bump to 80/80 for snappier moves.
    AccSet 50, 50;
    WHILE TRUE DO
      WaitUntil py_go = TRUE;
      ! `fine` = exact stop at every waypoint (no blending). Robot pauses
      ! briefly between segments. Swap to `z10` / `z50` / `z100` if you want
      ! blended motion at the cost of precision through the waypoints.
      MoveAbsJ py_target, v_tcp, fine, tool0;
      py_go := FALSE;
    ENDWHILE
  ENDPROC
ENDMODULE
"""

# Modules in T_ROB1 that also declare PROC main() — we unload these first to
# avoid the "duplicate entry point" semantic error. The GoFa ships with
# MainModule by default; safe to unload (it has no required logic).
MODULES_TO_UNLOAD = ("MainModule", "PyEgm")


def step(n: int, msg: str) -> None:
    print(f"\n[{n}] {msg}")


def main() -> int:
    print(f"Connecting to RWS at http://{ROBOT_IP}  user={RWS_USER!r}")
    rws = abb_rws.RWSClient(host=ROBOT_IP, user=RWS_USER, password=RWS_PASSWORD)

    # ----- 1. probe -----
    step(1, "Probing controller state")
    try:
        ctrl = rws.get_controller_state()
        opmode = rws.get_operation_mode()
        execstate = rws.get_execution_state()
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Check ROBOT_IP, RWS user/password, that the controller is powered on,")
        print("  and that RWS services are enabled on the pendant.")
        return 1
    print(f"  ctrl-state={ctrl}  opmode={opmode}  exec={execstate}")
    if opmode.lower() not in ("auto", "auto_ch", "automatic"):
        print(f"  ERROR: controller must be in AUTO mode (it's in {opmode!r}).")
        print("  Turn the key switch on the cabinet to AUTO and rerun.")
        return 1

    # ----- 2. request mastership FIRST (OmniCore needs it for stop/load/PP) -----
    step(2, "Requesting RAPID mastership")
    try:
        rws.request_mastership()
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Another client may hold mastership. From the pendant, log out other")
        print("  RWS sessions or reboot the controller.")
        return 1

    try:
        # ----- 3. stop any running program -----
        step(3, "Stopping any running program (safe)")
        if execstate == "running":
            try:
                rws.stop_program()
                time.sleep(0.5)
            except Exception as e:
                print(f"  stop_program: {e} (continuing)")
        else:
            print(f"  already {execstate}, nothing to stop")

        # ----- 4. upload module file -----
        step(4, f"Uploading {MODULE_FILENAME} to HOME/")
        url = f"{rws.base}/fileservice/$home/{MODULE_FILENAME}"
        r = rws._session.put(
            url,
            data=PYEXEC_MOD.encode("utf-8"),
            headers={"Content-Type": "application/octet-stream;v=2.0"},
            timeout=10.0,
        )
        if not r.ok:
            print(f"  upload returned HTTP {r.status_code}: {r.text[:300]}")
            return 1
        print("  uploaded")

        # ----- 4b. unload conflicting modules (anything else that has PROC main) -----
        for mod in MODULES_TO_UNLOAD:
            try:
                rws.unload_module(mod)
                print(f"  unloaded conflicting module: {mod}")
            except Exception as e:
                print(f"  unload {mod}: {e} (continuing)")

        # ----- 5. load module into T_ROB1 -----
        step(5, f"Loading {MODULE_FILENAME} into task T_ROB1")
        # OmniCore RWS 2.0: task-scoped loadmod
        load_url = f"{rws.base}/rw/rapid/tasks/T_ROB1/loadmod"
        r = rws._session.post(
            load_url,
            data={"modulepath": f"$home/{MODULE_FILENAME}", "replace": "true"},
            headers={"Content-Type": "application/x-www-form-urlencoded;v=2.0"},
            timeout=10.0,
        )
        if not r.ok:
            print(f"  loadmodule returned HTTP {r.status_code}: {r.text[:300]}")
            return 1
        print("  loaded")

        # ----- 6. motors on (in case they're off) -----
        step(6, "Turning motors on")
        try:
            rws.set_motors_on()
            time.sleep(0.5)
            ctrl = rws.get_controller_state()
            print(f"  ctrl-state -> {ctrl}")
        except Exception as e:
            print(f"  set_motors_on: {e} (you can also do this on the pendant)")

        # ----- 7. try resetpp + start; fall back to pendant if they fail -----
        step(7, "Resetting PP and starting program")
        auto_started = False
        try:
            rws.reset_pp()
            time.sleep(0.3)
            rws.start_program()
            time.sleep(0.5)
            exec_now = rws.get_execution_state()
            if exec_now == "running":
                print(f"  exec-state -> {exec_now} (program running)")
                auto_started = True
            else:
                print(f"  exec-state -> {exec_now} (not running, need pendant)")
        except Exception as e:
            print(f"  auto start failed: {e}")

        if not auto_started:
            print()
            print("  Releasing RAPID mastership so the pendant can take over...")
            try:
                rws.release_mastership()
            except Exception as e:
                print(f"  release_mastership: {e}")
            print()
            print("  ===== Pendant steps (mastership now free) =====")
            print("  1. On the sidebar, tap 'PP to Main' (write access should no longer be blocked).")
            print("  2. Press the green Play / Start button (or white motors-on if needed).")
            print("  The program should enter 'running' and park at WaitUntil py_go.")
            print("  ===============================================")
            input("  Press Enter once the pendant shows 'running'...")
            exec_now = rws.get_execution_state()
            if exec_now != "running":
                print(f"  still {exec_now}; aborting.")
                return 1
            print(f"  exec-state -> {exec_now}, ok.")
            return 0  # already released above, skip the finally release

    finally:
        try:
            rws.release_mastership()
        except Exception:
            pass

    print("\nDONE. PyExec is loaded and parked at WaitUntil py_go.")
    print("You can now run teleop_gofa.py and use 'Execute on robot' freely.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
