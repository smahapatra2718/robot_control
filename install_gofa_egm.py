"""
One-time setup: install the EGM RAPID supervisor + UC device config on the
GoFa controller, so teleop_gofa_egm.py can stream joint targets at 250 Hz.

Run this once. After that, just run teleop_gofa_egm.py.

What it does:
  1. Connect to RWS at ROBOT_IP.
  2. Print controller state + opmode.
  3. Grab RAPID mastership.
  4. Stop any running program.
  5. Unload the MoveAbsJ supervisor (PyExec) and MainModule if loaded.
  6. Upload EGM_COMM.cfg + EGM_MOC.cfg to HOME/ on the controller.
  7. Upload PyEgm.mod to HOME/ on the controller.
  8. Load PyEgm.mod into task T_ROB1.
  9. Walk the user through loading the .cfg files via FlexPendant (config
     load via RWS is unreliable on OmniCore — same pattern as PP-to-Main).
 10. Reset PP + start, fall back to pendant Play if RWS can't.

Prerequisites:
  A. Controller in a known-good state (we override the supervisor regardless).
  B. Your Mac's IP is what's in EGM_COMM.cfg under RemoteAddress. Default
     assumes 192.168.125.50; edit EGM_COMM.cfg and PC_IP below to match.
  C. Controller in AUTO, motors on (or we'll turn them on), robot in Normal.
"""

import sys
import time

import abb_rws

# ---- match what EGM_COMM.cfg / PyEgm.mod expect ----
ROBOT_IP = "192.168.125.1"          # GoFa MGMT port
RWS_USER = "Default User"
RWS_PASSWORD = "robotics"

# Sanity-check value. Must equal the RemoteAddress field in EGM_COMM.cfg.
# Used only to remind you to keep them in sync; the controller doesn't see it.
PC_IP_EXPECTED = "192.168.125.50"
EGM_PORT = 6510

MODULE_NAME = "PyEgm"
MODULE_FILENAME = "PyEgm.mod"

# Anything else with a PROC main() that would collide. We unload these first.
MODULES_TO_UNLOAD = ("MainModule", "PyExec")

# The EGM endpoint config lives in SIO:UDPUC_HOST. The controller ships with
# an entry already named "UCdevice" that just needs RemoteAddress changed
# from 127.0.0.1 to the PC's IP. RWS doesn't let "Default User" write this
# (every body shape returns 0xc004841a / -EACCES), so we walk the user
# through editing the single field via RobotStudio or FlexPendant.
UDPUC_INSTANCE_NAME = "UCdevice"

PYEGM_MOD = """\
MODULE PyEgm
  ! Flipped TRUE by Python (via RWS) to start an EGM streaming session.
  ! After the session ends (1s of convergence) RAPID clears it back to FALSE.
  PERS bool egm_go := FALSE;

  ! Flipped TRUE by Python (via RWS) to engage software lead-through (hand-guiding).
  ! Cleared back to FALSE to release. Mutually exclusive with egm_go.
  PERS bool lead_go := FALSE;

  ! Config name referenced by EGMSetupUC. MUST match the Name of a UDPUC_HOST
  ! entry on the controller. UCdevice's RemoteAddress is set via RobotStudio
  ! (Configuration -> Communication -> UDP Unicast Communication Host) to
  ! point at the PC running teleop_gofa_egm.py.
  CONST string EGM_EXT_NAME := "default";
  CONST string EGM_UC_NAME  := "UCdevice";

  VAR egmident egm_id;

  PROC main()
    AccSet 50, 50;

    WHILE TRUE DO
      WaitUntil egm_go = TRUE OR lead_go = TRUE;

      IF lead_go = TRUE THEN
        ! Software lead-through (hand-guiding). SetLeadThrough \\On orders a
        ! StopMove by default, so the arm goes compliant even though this task
        ! keeps executing; \\Off resumes (default ClearPath + StartMove).
        SetLeadThrough \\On;
        WaitUntil lead_go = FALSE;
        SetLeadThrough \\Off;
      ELSE
        EGMReset egm_id;
        EGMGetId egm_id;
        EGMSetupUC ROB_1, egm_id, EGM_EXT_NAME, EGM_UC_NAME \\Joint;

        !   \\LpFilter         : low-pass cutoff in Hz (lower = more smoothing)
        !   \\MaxSpeedDeviation: cap on per-joint speed during EGM, deg/s.
        !     20 deg/s ~= 0.35 rad/s; at the 0.95 m reach that bounds the TCP
        !     near the 250 mm/s collaborative limit as a controller-side backstop
        !     to the Python MAX_TCP_SPEED cap. Raise both together if you raise speed.
        EGMActJoint egm_id \\LpFilter := 20 \\MaxSpeedDeviation := 20;

        EGMRunJoint egm_id, EGM_STOP_HOLD
          \\J1 \\J2 \\J3 \\J4 \\J5 \\J6
          \\CondTime := 1 \\RampInTime := 0.1 \\RampOutTime := 0.2;

        EGMReset egm_id;
        egm_go := FALSE;
      ENDIF
    ENDWHILE
  ENDPROC
ENDMODULE
"""


def step(n: int, msg: str) -> None:
    print(f"\n[{n}] {msg}")


def main() -> int:
    print(f"Connecting to RWS at https://{ROBOT_IP}  user={RWS_USER!r}")
    print(f"Expected PC IP (must match EGM_COMM.cfg RemoteAddress): {PC_IP_EXPECTED}")
    print(f"EGM UDP port: {EGM_PORT}")
    rws = abb_rws.RWSClient(host=ROBOT_IP, user=RWS_USER, password=RWS_PASSWORD)

    # ----- 1. probe -----
    step(1, "Probing controller state")
    try:
        ctrl = rws.get_controller_state()
        opmode = rws.get_operation_mode()
        execstate = rws.get_execution_state()
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1
    print(f"  ctrl-state={ctrl}  opmode={opmode}  exec={execstate}")
    if opmode.lower() not in ("auto", "auto_ch", "automatic"):
        print(f"  ERROR: controller must be in AUTO mode (it's in {opmode!r}).")
        return 1

    # ----- 2. mastership -----
    step(2, "Requesting RAPID mastership")
    try:
        rws.request_mastership()
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1

    try:
        # ----- 3. stop -----
        step(3, "Stopping any running program")
        if execstate == "running":
            try:
                rws.stop_program()
                time.sleep(0.5)
            except Exception as e:
                print(f"  stop_program: {e} (continuing)")
        else:
            print(f"  already {execstate}")

        # ----- 4. inspect existing UCdevice config -----
        step(4, f"Checking existing SIO:UDPUC_HOST/{UDPUC_INSTANCE_NAME} config")
        try:
            r = rws._session.get(
                f"{rws.base}/rw/cfg/SIO/UDPUC_HOST/instances/{UDPUC_INSTANCE_NAME}",
                timeout=5.0,
            )
            r.raise_for_status()
            existing = r.json()
            attribs = {
                a["_title"]: a["value"]
                for inst in existing.get("state", [])
                for a in inst.get("attrib", [])
            }
            current_ip = attribs.get("RemoteAddress", "?")
            current_port = attribs.get("RemotePortNumber", "?")
            print(f"  Found: RemoteAddress={current_ip}  RemotePortNumber={current_port}")
            if current_ip != PC_IP_EXPECTED:
                print(f"  ERROR: RemoteAddress is {current_ip!r}, expected {PC_IP_EXPECTED!r}.")
                print(f"  EGM packets will not reach this PC. Edit UCdevice via RobotStudio:")
                print(f"  Configuration -> Communication -> UDP Unicast Communication Host ->")
                print(f"  UCdevice -> RemoteAddress = {PC_IP_EXPECTED}, then warm-restart.")
                return 1
            print(f"  RemoteAddress matches expected ({current_ip}). Continuing.")
        except Exception as e:
            print(f"  WARNING: could not inspect UCdevice config: {e}")
            print(f"  Continuing anyway; EGM will fail if RemoteAddress is wrong.")

        # ----- 5. unload conflicting modules -----
        step(5, "Unloading conflicting modules")
        for mod in MODULES_TO_UNLOAD:
            try:
                rws.unload_module(mod)
                print(f"  unloaded {mod}")
            except Exception as e:
                print(f"  unload {mod}: {e} (continuing)")

        # ----- 6. upload PyEgm.mod -----
        step(6, f"Uploading {MODULE_FILENAME} to HOME/")
        url = f"{rws.base}/fileservice/$home/{MODULE_FILENAME}"
        r = rws._session.put(
            url, data=PYEGM_MOD.encode("utf-8"),
            headers={"Content-Type": "application/octet-stream;v=2.0"},
            timeout=10.0,
        )
        if not r.ok:
            print(f"  upload HTTP {r.status_code}: {r.text[:200]}")
            return 1
        print("  uploaded")

        # ----- 7. load PyEgm.mod -----
        step(7, f"Loading {MODULE_FILENAME} into T_ROB1")
        load_url = f"{rws.base}/rw/rapid/tasks/T_ROB1/loadmod"
        r = rws._session.post(
            load_url,
            data={"modulepath": f"$home/{MODULE_FILENAME}", "replace": "true"},
            headers={"Content-Type": "application/x-www-form-urlencoded;v=2.0"},
            timeout=10.0,
        )
        if not r.ok:
            print(f"  loadmodule HTTP {r.status_code}: {r.text[:200]}")
            return 1
        print("  loaded")

        # ----- 8. motors + start -----
        step(8, "Turning motors on")
        try:
            rws.set_motors_on()
            time.sleep(0.5)
            print(f"  ctrl-state -> {rws.get_controller_state()}")
        except Exception as e:
            print(f"  set_motors_on: {e}")

        step(9, "Resetting PP + starting program")
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
                print(f"  exec-state -> {exec_now} (not running)")
        except Exception as e:
            print(f"  auto start failed: {e}")

        if not auto_started:
            print()
            print("  Releasing mastership so the pendant can take over...")
            try:
                rws.release_mastership()
            except Exception:
                pass
            print()
            print("  ===== Pendant steps =====")
            print("  1. Tap 'PP to Main' on the sidebar.")
            print("  2. Press the green Play button.")
            print("  The program should park at WaitUntil egm_go.")
            print("  =========================")
            input("  Press Enter when pendant shows 'running'...")
            exec_now = rws.get_execution_state()
            if exec_now != "running":
                print(f"  still {exec_now}; aborting.")
                return 1
            print(f"  exec-state -> {exec_now}, ok.")
            return 0

    finally:
        try:
            rws.release_mastership()
        except Exception:
            pass

    print("\nDONE. PyEgm is loaded and parked at WaitUntil egm_go.")
    print("Run teleop_gofa_egm.py next; toggle 'Execute on robot' to stream.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
