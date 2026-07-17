"""WorldFoundry synthesis facade for GigaWorld-Policy-0.5."""

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class GigaWorldPolicySynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed action-centered world-action policy."""

    MODEL_ID = "giga-world-policy-0.5"


__all__ = ["GigaWorldPolicySynthesis"]
