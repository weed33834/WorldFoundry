"""WorldFoundry synthesis facade for RISE."""

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class RiseSynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed RISE Pi0.5 action policy."""

    MODEL_ID = "rise"


__all__ = ["RiseSynthesis"]
