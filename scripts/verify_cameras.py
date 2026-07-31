"""
One-shot USB camera probe. Run this BEFORE wiring cameras to arms, to find out
which device is which and what to put in UR_CAMERA_INDEX / GOFA_CAMERA_INDEX.

Enumerates every plausible capture device, opens each with the same backend
selection lib/camera.py uses, grabs a frame, and reports what came back. With
`--save` it writes one JPEG per working device so you can *look* at them and tell
which camera is pointed at which arm -- the only reliable way to map them.

  uv run scripts/verify_cameras.py            # probe and report
  uv run scripts/verify_cameras.py --save     # also write /tmp/camera-probe/*.jpg

Two UVC cameras usually expose FOUR /dev/video* nodes: each claims a capture node
and a metadata node, and only the capture node yields frames. That is why the
report distinguishes "delivered a frame" from "opened" -- configure only the
former. Prefer the printed /dev/v4l/by-id/... path over a bare index: indices are
assigned in enumeration order and move when devices are replugged or reboot, and
pointing an arm at the wrong camera is a silent, confusing failure.

WSL note: if this finds nothing and /dev/video* does not exist, the camera is not
visible to Linux at all. usbipd-win forwards the USB device, but the stock WSL2
kernel ships no uvcvideo driver to bind it -- see API.md "WSL2 has no USB cameras
by default".
"""

import argparse
import glob
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "lib")):  # repo root + lib/ (our modules)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import robot_common as rc  # noqa: E402
from camera import CV2_AVAILABLE, CameraSource  # noqa: E402

SAVE_DIR = os.path.join("/tmp", "camera-probe")
MAX_INDEX = 10          # how far to scan when there are no /dev nodes to enumerate


def candidates() -> list:
    """Devices worth trying: real /dev nodes on Linux, otherwise plain indices."""
    if sys.platform.startswith("linux"):
        by_id = sorted(glob.glob("/dev/v4l/by-id/*"))     # stable across replug
        nodes = sorted(glob.glob("/dev/video*"))
        return by_id + nodes if (by_id or nodes) else []
    return list(range(MAX_INDEX))


def probe(dev, save: bool) -> dict:
    """Open one device exactly as the server would, and report what happened."""
    src = CameraSource(dev, rc.CAMERA_WIDTH, rc.CAMERA_HEIGHT, rc.CAMERA_FPS,
                       rc.CAMERA_JPEG_QUALITY, name="probe")
    if not src.open():
        return {"dev": dev, "ok": False, "error": src.error}
    try:
        f = src.latest()
        out = {"dev": dev, "ok": f is not None, "backend": getattr(src, "backend", None),
               "bytes": len(f.jpeg) if f else 0, "error": None}
        if f and save:
            os.makedirs(SAVE_DIR, exist_ok=True)
            name = str(dev).strip("/").replace("/", "_") + ".jpg"
            path = os.path.join(SAVE_DIR, name)
            with open(path, "wb") as fh:
                fh.write(f.jpeg)
            out["saved"] = path
        return out
    finally:
        src.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe USB cameras for the remote API.")
    ap.add_argument("--save", action="store_true",
                    help=f"write one JPEG per working device to {SAVE_DIR}")
    args = ap.parse_args()

    if not CV2_AVAILABLE:
        print("opencv is not installed — run `uv sync` (opencv-python-headless).")
        raise SystemExit(1)

    devs = candidates()
    if sys.platform.startswith("linux") and not devs:
        print("No /dev/video* devices exist.\n"
              "  The camera is not visible to Linux at all. Under WSL2 that is the default:\n"
              "  usbipd-win forwards the USB device, but the stock WSL2 kernel has no uvcvideo\n"
              "  driver to bind it, so no node is ever created. Check:\n"
              "    usbipd.exe list           # is the device attached?\n"
              "    lsmod | grep uvcvideo     # empty => kernel lacks the driver\n"
              "  See API.md -> Cameras -> 'WSL2 has no USB cameras by default'.")
        raise SystemExit(1)

    print(f"Probing {len(devs)} candidate device(s) at "
          f"{rc.CAMERA_WIDTH}x{rc.CAMERA_HEIGHT} @{rc.CAMERA_FPS} ...\n")
    working = []
    for dev in devs:
        r = probe(dev, args.save)
        if r["ok"]:
            working.append(r)
            extra = f"  -> {r['saved']}" if r.get("saved") else ""
            print(f"  WORKS   {dev!r:44} {r['backend']:<12} {r['bytes']:>7} B/frame{extra}")
        else:
            # the common, uninteresting case: a node that exists but yields nothing
            print(f"  no      {dev!r:44} {r['error']}")

    print()
    if not working:
        print("No device delivered a frame. Nothing to configure.")
        raise SystemExit(1)

    print(f"{len(working)} device(s) delivered frames.")
    if not args.save:
        print("Re-run with --save to write a JPEG per device — comparing the images is the\n"
              "only dependable way to tell which camera is aimed at which arm.")
    print("\nOnce you know which is which, set them per host (a path beats an index —\n"
          "indices move when devices re-enumerate):\n")
    picks = [w["dev"] for w in working]
    ur = picks[0]
    gofa = picks[1] if len(picks) > 1 else "-1   # -1 disables"
    print(f"  export UR_CAMERA_INDEX={ur}")
    print(f"  export GOFA_CAMERA_INDEX={gofa}")
    print("\nThen: uv run scripts/real.py api     (startup prints what each arm resolved to)")


if __name__ == "__main__":
    main()
