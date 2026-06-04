"""
Minimal Externally Guided Motion (EGM) client for ABB OmniCore.

Pairs with `abb_rws.py`: RWS handles state/control (motors on, load module,
mastership, start program). EGM handles the high-rate UDP streaming once the
RAPID side calls EGMRunJoint.

Wire format: protobuf, defined in egm.proto. Joints on the wire are in
**degrees** (ABB convention); we convert to radians at the boundary.

Usage:
    egm = EGMSession(local_port=6510)
    egm.start()
    # ... once RAPID enters EGMRunJoint, packets start arriving:
    while True:
        if egm.has_feedback():
            q = egm.get_feedback_rad()        # current robot joints
            egm.set_target_rad(my_target)     # gets echoed back at ~250 Hz
        time.sleep(0.004)
    egm.stop()

The background thread is the *only* writer for `_feedback` and the *only*
reader for `_target` — both protected by `_lock`. Set/get from any thread.
"""

from __future__ import annotations

import math
import socket
import threading
import time
from dataclasses import dataclass, field

import egm_pb2


@dataclass
class EGMSession:
    local_port: int = 6510
    bind_host: str = "0.0.0.0"
    recv_buf_size: int = 4096

    _sock: socket.socket | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _stop_flag: threading.Event = field(default_factory=threading.Event, init=False)

    # Latest feedback from controller (degrees on the wire, stored as rad here).
    _feedback_rad: list[float] | None = field(default=None, init=False)
    _feedback_seq: int = field(default=0, init=False)
    _feedback_time: float = field(default=0.0, init=False)

    # Latest target from the user (rad).
    _target_rad: list[float] | None = field(default=None, init=False)
    _target_seq: int = field(default=0, init=False)

    # Remote address learned from incoming packets (we reply to the source).
    _remote_addr: tuple[str, int] | None = field(default=None, init=False)

    # Counters for diagnostics.
    packets_rx: int = field(default=0, init=False)
    packets_tx: int = field(default=0, init=False)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.bind_host, self.local_port))
        self._sock.settimeout(0.5)
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def set_target_rad(self, joints_rad: list[float]) -> None:
        assert len(joints_rad) == 6
        with self._lock:
            self._target_rad = list(joints_rad)
            self._target_seq += 1

    def get_feedback_rad(self) -> list[float] | None:
        with self._lock:
            return list(self._feedback_rad) if self._feedback_rad is not None else None

    def has_feedback(self) -> bool:
        with self._lock:
            return self._feedback_rad is not None

    def is_fresh(self, max_age_s: float = 0.2) -> bool:
        with self._lock:
            return (
                self._feedback_rad is not None
                and (time.time() - self._feedback_time) < max_age_s
            )

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "rx": self.packets_rx,
                "tx": self.packets_tx,
                "age_s": (time.time() - self._feedback_time) if self._feedback_rad else -1.0,
                "remote": self._remote_addr or "(none)",
            }

    # ---- internal ----
    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop_flag.is_set():
            try:
                data, addr = self._sock.recvfrom(self.recv_buf_size)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                msg = egm_pb2.EgmRobot.FromString(data)
            except Exception:
                continue  # malformed packet; skip

            with self._lock:
                self.packets_rx += 1
                self._remote_addr = addr
                fb = msg.feedBack
                if fb.HasField("joints") and len(fb.joints.joints) >= 6:
                    # ABB sends degrees; we work in radians everywhere else.
                    self._feedback_rad = [
                        math.radians(fb.joints.joints[i]) for i in range(6)
                    ]
                    self._feedback_time = time.time()
                # Mirror seqno so the controller knows we're paired with it.
                in_seq = msg.header.seqno if msg.header.HasField("seqno") else 0

            # Build and send reply outside the lock (UDP send is fast but
            # blocks the GIL on the network call).
            reply = self._build_reply(in_seq)
            if reply is not None:
                try:
                    self._sock.sendto(reply, addr)
                    with self._lock:
                        self.packets_tx += 1
                except OSError:
                    pass

    def _build_reply(self, mirror_seq: int) -> bytes | None:
        with self._lock:
            target = list(self._target_rad) if self._target_rad is not None else None
            target_seq = self._target_seq

        if target is None:
            # Nothing useful to say yet. The controller's EGMRunJoint will
            # hold position while we have no target.
            return None

        sensor = egm_pb2.EgmSensor()
        sensor.header.seqno = target_seq
        sensor.header.tm = int((time.time() * 1000) % (1 << 32))
        sensor.header.mtype = egm_pb2.EgmHeader.MSGTYPE_CORRECTION
        joints_deg = [math.degrees(q) for q in target]
        sensor.planned.joints.joints.extend(joints_deg)
        return sensor.SerializeToString()
