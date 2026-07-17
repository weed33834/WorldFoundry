"""WorldFoundry synthesis surface for EventVLA."""

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class EventVLASynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed event-driven visual-memory action generation."""

    MODEL_ID = "eventvla"


__all__ = ["EventVLASynthesis"]
