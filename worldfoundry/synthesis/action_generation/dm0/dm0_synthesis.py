"""WorldFoundry synthesis facade for DM0."""

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class DM0Synthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    """Profile-backed triple-view flow-policy inference."""

    MODEL_ID = "dm0"


__all__ = ["DM0Synthesis"]
