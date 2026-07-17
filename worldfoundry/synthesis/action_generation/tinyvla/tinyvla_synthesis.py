"""WorldFoundry synthesis facade for TinyVLA."""

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class TinyVLASynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed LLaVA-Pythia action policy."""

    MODEL_ID = "tinyvla"


__all__ = ["TinyVLASynthesis"]
