"""
Minimal Robotiq Hand-E gripper client.

Talks to the Robotiq *Grippers* URCap's background socket server, which the
URCap runs on the controller at <robot_ip>:63352. That daemon owns the wrist
RS-485 and runs independently of whatever program is playing -- so it coexists
with ur_rtde's resident control script, and the pendant gripper buttons keep
working too (both route through the same daemon).

No external deps and no Modbus: the URCap server speaks a simple newline-
terminated ASCII protocol (same one the well-known standalone robotiq_gripper.py
driver uses):

    SET <VAR> <val> [<VAR> <val> ...]\\n   -> "ack"
    GET <VAR>\\n                           -> "<VAR> <val>"

Variables we use: ACT (activate), GTO (go-to), POS (request 0=open..255=closed),
SPE (speed 0..255), FOR (force 0..255), STA (status; 3=activation complete),
OBJ (object detection; 1/2 = stopped on an object), FLT (fault).
"""

from __future__ import annotations

import socket
import time

DEFAULT_PORT = 63352     # Robotiq Grippers URCap socket server

DEFAULT_SPEED = 255      # SPE: full speed
DEFAULT_FORCE = 150      # FOR: moderate force (collaborative)


class HandEGripper:
    def __init__(self, host: str, port: int = DEFAULT_PORT, timeout: float = 2.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None

    # ---- transport ----
    def connect(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), self.timeout)
        self._sock.settimeout(self.timeout)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _cmd(self, line: str) -> str:
        """Send one command line and return its reply (stripped).

        The URCap socket replies are short and not reliably newline-terminated,
        so we do a single recv like the canonical robotiq_gripper.py driver
        rather than waiting for a '\\n' that may never arrive.
        """
        assert self._sock is not None, "call connect() first"
        self._sock.sendall(line.encode("ascii") + b"\n")
        return self._sock.recv(1024).decode("ascii").strip()

    def _set(self, **vars_: int) -> None:
        parts = " ".join(f"{k} {int(v)}" for k, v in vars_.items())
        reply = self._cmd(f"SET {parts}")
        if reply.lower() != "ack":
            raise IOError(f"SET {parts!r} -> unexpected reply {reply!r}")

    def _get(self, var: str) -> int:
        reply = self._cmd(f"GET {var}")
        # reply is like "POS 255"
        toks = reply.split()
        if len(toks) != 2 or toks[0] != var:
            raise IOError(f"GET {var} -> unexpected reply {reply!r}")
        return int(toks[1])

    # ---- gripper protocol ----
    def activate(self, timeout: float = 5.0) -> None:
        """Activate the gripper; blocks until STA reports activation complete."""
        self._set(ACT=1, GTO=1)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._get("STA") == 3:
                return
            time.sleep(0.2)
        raise TimeoutError("Hand-E activation did not complete (STA != 3)")

    def _move(self, position: int, speed: int, force: int) -> None:
        position = max(0, min(255, position))
        self._set(POS=position, SPE=speed, FOR=force, GTO=1)

    def open(self, speed: int = DEFAULT_SPEED, force: int = DEFAULT_FORCE) -> None:
        self._move(0, speed, force)

    def close_gripper(self, speed: int = DEFAULT_SPEED, force: int = DEFAULT_FORCE) -> None:
        self._move(255, speed, force)

    def status(self) -> dict[str, int | bool]:
        return {
            "activated": self._get("STA") == 3,
            "object": self._get("OBJ") in (1, 2),
            "fault": self._get("FLT"),
            "position": self._get("POS"),
        }
