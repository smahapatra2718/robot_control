"""USB camera capture for the remote API — one background grab thread per arm.

A `CameraSource` owns one `cv2.VideoCapture`, grabs continuously into a single latest-frame
slot, and hands out JPEG bytes. Endpoints read the slot; they never touch the device, so a
slow client can't stall capture and N viewers cost one decode.

**Timestamps.** Each frame carries two clocks, stamped together right after the grab:

  ts       time.monotonic()  — the SAME clock as RobotState.ts, so `frame.ts - state.ts`
                               is a real interval. This is what makes a frame pairable with
                               telemetry. Only meaningful inside this process.
  ts_unix  time.time()       — wall clock, for logging and cross-machine correlation.

Carrying both is what lets a client anchor the monotonic clock: read one frame, and the
`ts_unix - ts` offset converts any RobotState.ts to absolute time. That's why RobotState
itself needs no new field.

Capture is best-effort, like the Hand-E gripper: if cv2 is missing or the device won't open,
`open_camera()` returns None and the API reports the camera unavailable rather than failing.
"""
from __future__ import annotations

import glob
import os
import sys
import threading
import time
from dataclasses import dataclass

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:                                # noqa: BLE001 - optional dependency
    CV2_AVAILABLE = False


@dataclass(frozen=True)
class Frame:
    """One captured frame. `jpeg` is encoded bytes; see the module docstring on clocks."""
    jpeg: bytes
    ts: float          # time.monotonic() at grab — same clock as RobotState.ts
    ts_unix: float     # time.time() at grab — wall clock
    seq: int           # monotonic counter from 1, so a client can spot dropped/stale frames


class CameraSource:
    """Background grab loop over one capture device, exposing the latest JPEG frame."""

    def __init__(self, index, width: int, height: int, fps: int, quality: int,
                 name: str = "camera") -> None:
        self.index, self.name = index, name      # int index or a "/dev/videoN" path
        self.width, self.height, self.fps, self.quality = width, height, fps, quality
        self.error: str | None = None            # why open() failed, surfaced by /camera/info
        self._cap = None
        self._frame: Frame | None = None
        self._seq = 0
        self._cv = threading.Condition()         # guards _frame/_seq; notifies stream waiters
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle ----
    def _backends(self):
        """Backends to try, best first. Naming one matters: with no /dev/video* present,
        cv2 silently falls through to CAP_FFMPEG and reports 'configure with libavdevice',
        which points at the wrong problem entirely (it's a missing device, not a build
        option). Trying V4L2 explicitly makes the real failure the one you see."""
        if isinstance(self.index, str):
            return [("V4L2", cv2.CAP_V4L2), ("default", cv2.CAP_ANY)]
        if sys.platform.startswith("linux"):
            return [("V4L2", cv2.CAP_V4L2), ("default", cv2.CAP_ANY)]
        if sys.platform == "darwin":
            return [("AVFoundation", cv2.CAP_AVFOUNDATION), ("default", cv2.CAP_ANY)]
        return [("default", cv2.CAP_ANY)]        # Windows: MSMF/DSHOW auto-select is fine

    @staticmethod
    def _linux_hint() -> str:
        """Explain an open failure on Linux when there are simply no capture devices.

        This is the WSL2 case and it's worth naming: the default WSL kernel ships no
        uvcvideo/V4L2, so a USB camera never appears as /dev/video* even after
        `usbipd attach`, and OpenCV's fallback to CAP_FFMPEG then reports 'configure with
        libavdevice' — which points at a build option rather than the real cause.

        Checked here, after the open attempt, rather than as a precondition: the sim
        shadows cv2 with a synthetic device that has no /dev node, and gating on one
        would break offline runs on Linux.
        """
        if not sys.platform.startswith("linux") or glob.glob("/dev/video*"):
            return ""
        return (". No /dev/video* devices exist on this host, so no camera is visible to "
                "Linux at all. Under WSL2 this is the default: its kernel has no uvcvideo/"
                "V4L2 support, so USB cameras never appear even after `usbipd attach` — "
                "run the server on Windows, or build a WSL kernel with UVC")

    def open(self) -> bool:
        """Open the device and start grabbing. False if unavailable (never raises);
        `self.error` then says why, so the server can print something actionable."""
        if not CV2_AVAILABLE:
            self.error = "opencv is not installed (pip/uv add opencv-python-headless)"
            return False
        if isinstance(self.index, int) and self.index < 0:
            self.error = "disabled (index -1)"
            return False
        if isinstance(self.index, str) and not os.path.exists(self.index):
            self.error = f"{self.index} does not exist"
            return False
        tried = []
        for label, backend in self._backends():
            try:
                cap = cv2.VideoCapture(self.index, backend)
                if not cap.isOpened():
                    cap.release()
                    tried.append(f"{label}: would not open")
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                # isOpened() lies on some backends — a device can open and never deliver.
                # Require one real frame before declaring the camera up, so /camera/info
                # can't say available on a camera that will only ever 503.
                ok, img = cap.read()
                if not ok or img is None:
                    cap.release()
                    tried.append(f"{label}: opened but delivered no frame")
                    continue
            except Exception as e:               # noqa: BLE001 - any capture failure = no camera
                tried.append(f"{label}: {e}")
                continue
            self.error = None
            self.backend = label
            break
        else:
            self.error = f"could not open {self.index!r} — " + "; ".join(tried) + self._linux_hint()
            return False
        self._cap = cap
        # publish the validation frame rather than dropping it: otherwise `latest()` is None
        # until the grab thread's first tick and a request in that window 503s on a camera
        # we just declared available.
        self._encode_publish(img)
        self._thread = threading.Thread(target=self._grab_loop, daemon=True,
                                        name=f"camera-{self.name}")
        self._thread.start()
        return True

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:                    # noqa: BLE001 - best effort on shutdown
                pass
            self._cap = None

    # ---- capture ----
    def _encode_publish(self, img) -> None:
        """JPEG-encode a grabbed image and publish it, stamping both clocks together."""
        ts, ts_unix = time.monotonic(), time.time()
        ok, buf = cv2.imencode(".jpg", img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(self.quality)])
        if ok:
            self._publish(Frame(buf.tobytes(), ts, ts_unix, self._seq + 1))

    def _grab_loop(self) -> None:
        period = 1.0 / max(1, self.fps)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                ok, img = self._cap.read()
                if ok and img is not None:
                    self._encode_publish(img)
            except Exception:                    # noqa: BLE001 - a bad read shouldn't kill the thread
                pass
            # pace to fps; read() often blocks that long already, so this is usually a no-op
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _publish(self, frame: Frame) -> None:
        with self._cv:
            self._seq = frame.seq
            self._frame = frame
            self._cv.notify_all()

    # ---- reads ----
    def latest(self) -> Frame | None:
        with self._cv:
            return self._frame

    def wait_for_next(self, after_seq: int, timeout: float = 5.0) -> Frame | None:
        """Block until a frame newer than `after_seq` lands. None on timeout/shutdown.
        This is what makes the MJPEG stream push at the capture rate instead of polling."""
        with self._cv:
            if self._seq > after_seq and self._frame is not None:
                return self._frame
            self._cv.wait(timeout)
            return self._frame if self._seq > after_seq else None

    def info(self) -> dict:
        f = self.latest()
        return {"available": self._cap is not None, "index": self.index,
                "backend": getattr(self, "backend", None), "error": self.error,
                "width": self.width, "height": self.height, "fps": self.fps,
                "seq": self._seq, "ts": f.ts if f else None,
                "ts_unix": f.ts_unix if f else None}


def open_camera(robot: str, index=None) -> CameraSource:
    """Build and start the camera for an arm. Always returns a CameraSource — check
    `.error` / `info()["available"]`, don't test for None.

    A failed open still returns the object so the *reason* survives to `/camera/info` and
    the startup log. Best-effort by design: a missing camera degrades the API (endpoints
    503) instead of stopping the server, the same way an unreachable Hand-E socket leaves
    the UR viz-only.
    """
    import robot_common as rc
    idx = rc.camera_index(robot) if index is None else index
    src = CameraSource(idx, rc.CAMERA_WIDTH, rc.CAMERA_HEIGHT, rc.CAMERA_FPS,
                       rc.CAMERA_JPEG_QUALITY, name=robot)
    src.open()
    return src
