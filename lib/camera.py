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

    def __init__(self, index: int, width: int, height: int, fps: int, quality: int,
                 name: str = "camera") -> None:
        self.index, self.name = index, name
        self.width, self.height, self.fps, self.quality = width, height, fps, quality
        self._cap = None
        self._frame: Frame | None = None
        self._seq = 0
        self._cv = threading.Condition()         # guards _frame/_seq; notifies stream waiters
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle ----
    def open(self) -> bool:
        """Open the device and start grabbing. False if unavailable (never raises)."""
        if not CV2_AVAILABLE or self.index < 0:
            return False
        try:
            cap = cv2.VideoCapture(self.index)
            if not cap.isOpened():
                cap.release()
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            # isOpened() lies on some backends — a device can open and never deliver. Require
            # one real frame before declaring the camera up, so /camera/info can't say
            # available on a camera that will only ever 503.
            ok, img = cap.read()
            if not ok or img is None:
                cap.release()
                return False
        except Exception:                        # noqa: BLE001 - any capture failure = no camera
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
                "width": self.width, "height": self.height, "fps": self.fps,
                "seq": self._seq, "ts": f.ts if f else None,
                "ts_unix": f.ts_unix if f else None}


def open_camera(robot: str, index: int | None = None) -> CameraSource | None:
    """Build and start the camera for an arm, or None if there isn't a usable one.

    Best-effort by design: a missing camera degrades the API (endpoints 503) instead of
    stopping the server, the same way an unreachable Hand-E socket leaves the UR viz-only.
    """
    import robot_common as rc
    idx = rc.camera_index(robot) if index is None else index
    src = CameraSource(idx, rc.CAMERA_WIDTH, rc.CAMERA_HEIGHT, rc.CAMERA_FPS,
                       rc.CAMERA_JPEG_QUALITY, name=robot)
    return src if src.open() else None
