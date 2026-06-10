"""RobotController core: one motion implementation behind the teleop scripts and the API."""
from .base import Busy, RobotController, Unsupported
from .state import RobotState

__all__ = ["RobotController", "RobotState", "Busy", "Unsupported", "make_controller"]


def make_controller(robot: str) -> RobotController:
    if robot == "ur15":
        from .ur import URController
        return URController()
    if robot == "gofa":
        from .gofa import GoFaController
        return GoFaController()
    raise ValueError(f"unknown robot {robot!r} (expected 'ur15' or 'gofa')")
