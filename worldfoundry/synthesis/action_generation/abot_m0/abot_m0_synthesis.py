"""WorldFoundry synthesis facade for ABot-M0."""

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class ABotM0Synthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed, local-only ABot-M0 action generation."""

    MODEL_ID = "abot-m0"


__all__ = ["ABotM0Synthesis"]
