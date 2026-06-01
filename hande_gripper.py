"""
Minimal Robotiq Hand-E gripper client.

Talks Modbus RTU to the gripper over a TCP socket exposed by the UR's tool
RS-485 (the Robotiq "RS485" URCap / Tool Communication Interface forwards the
wrist RS-485 connector to a TCP port as a background daemon -- independent of
whatever program the controller is running, so it coexists with ur_rtde's
resident control script).

No external deps: the Robotiq command set is one write (FC16) + one status
read (FC04), so we frame Modbus RTU by hand over a stdlib socket, same spirit
as abb_rws.py / abb_egm.py.

Register map (Robotiq generic, shared by Hand-E; slave id 9):
  Robot OUTPUT (command) registers, written with FC16 at 0x03E8 as 3x16-bit:
    byte0 ACTION REQUEST  bit0 rACT, bit3 rGTO, bit4 rATR, bit5 rARD
    byte1 reserved
    byte2 reserved
    byte3 rPR  position request  0 = fully OPEN ... 255 = fully CLOSED
    byte4 rSP  speed   0..255
    byte5 rFR  force   0..255
  Robot INPUT (status) registers, read with FC04 at 0x07D0 as 3x16-bit:
    byte0 GRIPPER STATUS  bit0 gACT, bit3 gGTO, bits4-5 gSTA, bits6-7 gOBJ
    byte1 reserved
    byte2 FAULT STATUS gFLT
    byte3 gPR  echo of position request
    byte4 gPO  actual position 0..255
    byte5 gCU  motor current

gSTA == 3 means activation is complete. gOBJ in {1,2} means the fingers
stopped on an object (something is grasped).
"""

from __future__ import annotations

import socket
import time

CMD_ADDR = 0x03E8        # robot output (command) registers
STATUS_ADDR = 0x07D0     # robot input (status) registers
DEFAULT_PORT = 54321     # Robotiq RS485 URCap / Tool Comm socket

DEFAULT_SPEED = 255      # rSP: full speed
DEFAULT_FORCE = 150      # rFR: moderate force (collaborative)


def _crc16(data: bytes) -> bytes:
    """Modbus RTU CRC16 (poly 0xA001), returned low byte first."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return bytes((crc & 0xFF, (crc >> 8) & 0xFF))


class HandEGripper:
    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        slave_id: int = 9,
        timeout: float = 2.0,
    ) -> None:
        self.host = host
        self.port = port
        self.slave_id = slave_id
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

    def _txn(self, payload: bytes, expected_len: int) -> bytes:
        """Send one RTU frame (payload + CRC) and read back expected_len bytes."""
        assert self._sock is not None, "call connect() first"
        frame = payload + _crc16(payload)
        self._sock.sendall(frame)
        buf = b""
        deadline = time.monotonic() + self.timeout
        while len(buf) < expected_len and time.monotonic() < deadline:
            chunk = self._sock.recv(expected_len - len(buf))
            if not chunk:
                break
            buf += chunk
        if len(buf) < expected_len:
            raise IOError(f"short Modbus reply: got {len(buf)}/{expected_len} bytes")
        return buf

    def _write_regs(self, addr: int, regs: list[int]) -> None:
        body = bytes(
            (self.slave_id, 0x10, addr >> 8, addr & 0xFF, 0x00, len(regs), 2 * len(regs))
        )
        for r in regs:
            body += bytes((r >> 8, r & 0xFF))
        self._txn(body, expected_len=8)  # echo of slave,fc,addr,qty + CRC

    def _read_regs(self, addr: int, count: int) -> list[int]:
        body = bytes((self.slave_id, 0x04, addr >> 8, addr & 0xFF, 0x00, count))
        reply = self._txn(body, expected_len=5 + 2 * count)  # slave,fc,bytecount,data,CRC
        n = reply[2]
        data = reply[3 : 3 + n]
        return [(data[i] << 8) | data[i + 1] for i in range(0, n, 2)]

    # ---- gripper protocol ----
    def _command(self, position: int, speed: int, force: int) -> None:
        position = max(0, min(255, position))
        # byte0 = rACT|rGTO (0x09), byte1=0; byte2=0, byte3=position; byte4=speed, byte5=force
        regs = [0x0900, position & 0xFF, ((speed & 0xFF) << 8) | (force & 0xFF)]
        self._write_regs(CMD_ADDR, regs)

    def status(self) -> dict[str, int | bool]:
        regs = self._read_regs(STATUS_ADDR, 3)
        b0 = regs[0] >> 8          # gripper status byte
        fault = regs[1] >> 8       # gFLT
        pos = regs[2] >> 8         # gPO actual position 0..255
        return {
            "activated": bool(b0 & 0x01) and ((b0 >> 4) & 0x03) == 3,
            "object": ((b0 >> 6) & 0x03) in (1, 2),
            "fault": fault,
            "position": pos,
        }

    def activate(self, timeout: float = 5.0) -> None:
        """Run the activation handshake; blocks until gSTA reports complete."""
        # Clear then set rACT so a previously-activated gripper re-arms cleanly.
        self._write_regs(CMD_ADDR, [0x0000, 0x0000, 0x0000])
        time.sleep(0.1)
        self._write_regs(CMD_ADDR, [0x0100, 0x0000, 0x0000])  # rACT=1, rGTO=0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            regs = self._read_regs(STATUS_ADDR, 1)
            gsta = (regs[0] >> 8 >> 4) & 0x03
            if gsta == 3:
                return
            time.sleep(0.2)
        raise TimeoutError("Hand-E activation did not complete (gSTA != 3)")

    def open(self, speed: int = DEFAULT_SPEED, force: int = DEFAULT_FORCE) -> None:
        self._command(0, speed, force)

    def close_gripper(self, speed: int = DEFAULT_SPEED, force: int = DEFAULT_FORCE) -> None:
        self._command(255, speed, force)
