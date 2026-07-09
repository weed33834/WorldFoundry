from __future__ import annotations

from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.official_policy import OfficialPolicySynthesis


class DBCogACTSynthesis(OfficialPolicySynthesis, ActionModelSynthesis):
    MODEL_ID = "db-cogact"
