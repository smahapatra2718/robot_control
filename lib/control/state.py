"""RobotState: a JSON-serializable snapshot of the robot, produced by the
controller's state-poll thread and consumed by every surface (viser, API)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RobotState:
    ts: float                       # monotonic timestamp of the snapshot
    robot: str                      # "ur15" | "gofa"
    q: list[float]                  # 6 joint angles (rad)
    pose: dict[str, list[float]]    # {"pos": [x,y,z], "wxyz": [w,x,y,z]} grasp/EE pose
    gripper_frac: float | None      # 0=open..1=closed; None if no gripper
    safety_state: str               # robot-reported safety state
    controller_state: str           # robot-reported controller/exec state
    activity: str                   # "idle"|"moving"|"playing"|"stopped"
    active_command: dict | None     # {"id","kind","status","progress","error"} or None
    conn_ok: bool                   # last hardware read succeeded
    health: dict = field(default_factory=dict)   # transport-specific extras

    def to_dict(self) -> dict:
        return asdict(self)
