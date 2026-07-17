"""WorldFoundry synthesis surface for AHA-WAM."""

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class AHAWAMSynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed, history-aware world-action generation."""

    MODEL_ID = "ahawam"


__all__ = ["AHAWAMSynthesis"]
