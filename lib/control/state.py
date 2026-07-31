"""RobotState: a JSON-serializable snapshot of the robot, produced by the
controller's state-poll thread and consumed by every surface (viser, API)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Keys of the command object (RobotState.active_command / GET /command/{id}).
COMMAND_KEYS = ("id", "kind", "status", "progress", "error")


def empty_command() -> dict:
    """A command object with every key present and None — what `active_command` holds
    whenever nothing is running. Keeping the sub-keys present means a client can read
    `active_command.status` without null-checking the parent, and the served shape never
    changes underneath it."""
    return {k: None for k in COMMAND_KEYS}


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
    active_command: dict            # the *running* command, else all-None. Sub-keys always
                                    #   present: {"id","kind","status","progress","error"}.
                                    #   Terminal status is never shown here — see GET /command/{id}
    conn_ok: bool                   # last hardware read succeeded
    health: dict = field(default_factory=dict)   # transport-specific extras

    def to_dict(self) -> dict:
        return asdict(self)

    def to_flat_dict(self) -> dict:
        """The same snapshot with every value a scalar — no arrays, no sub-objects.

        For clients that can't walk nested JSON. Served by `GET /state?flat=1` and
        `WS /telemetry?flat=1`; `to_dict()` stays the default shape. Types are kept as-is
        (floats stay floats, bools stay bools), and an absent value is None rather than a
        missing key, so the key set is identical on every frame and for both arms — with
        one exception: `health_*` mirrors `health`, whose keys are transport-specific."""
        out: dict = {"ts": self.ts, "robot": self.robot}
        for i, v in enumerate(self.q):
            out[f"q_{i}"] = v
        for axis, v in zip("xyz", self.pose["pos"]):
            out[f"pose_pos_{axis}"] = v
        for axis, v in zip("wxyz", self.pose["wxyz"]):
            out[f"pose_wxyz_{axis}"] = v
        out["gripper_frac"] = self.gripper_frac
        out["safety_state"] = self.safety_state
        out["controller_state"] = self.controller_state
        out["activity"] = self.activity
        cmd = self.active_command or {}
        for k in COMMAND_KEYS:
            out[f"command_{k}"] = cmd.get(k)
        out["conn_ok"] = self.conn_ok
        for k, v in self.health.items():
            out[f"health_{k}"] = v
        return out
