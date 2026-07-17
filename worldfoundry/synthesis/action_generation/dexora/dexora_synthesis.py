"""WorldFoundry synthesis facade for Dexora."""

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class DexoraSynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed direct 36-DoF action generation."""

    MODEL_ID = "dexora-1b"


__all__ = ["DexoraSynthesis"]
