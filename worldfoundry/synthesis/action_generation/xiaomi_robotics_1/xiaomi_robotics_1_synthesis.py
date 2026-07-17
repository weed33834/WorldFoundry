"""WorldFoundry synthesis facade for Xiaomi-Robotics-1."""

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class XiaomiRobotics1Synthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed bimanual action generation."""

    MODEL_ID = "xiaomi-robotics-1"


__all__ = ["XiaomiRobotics1Synthesis"]
