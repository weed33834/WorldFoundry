"""First-class Studio entries for the in-tree OpenPI policy variants."""

from __future__ import annotations

from worldfoundry.synthesis.action_generation.base_action_synthesis import (
    ActionModelSynthesis,
)

from .openpi_synthesis import OpenPISynthesis


class Pi0Synthesis(OpenPISynthesis, ActionModelSynthesis):
    """Pi0 policy using the shared OpenPI inference implementation."""

    MODEL_ID = "pi0"


class Pi05Synthesis(OpenPISynthesis, ActionModelSynthesis):
    """Pi0.5 policy using the shared OpenPI inference implementation."""

    MODEL_ID = "pi05"


class Pi0FastSynthesis(OpenPISynthesis, ActionModelSynthesis):
    """Pi0-FAST policy using the shared OpenPI inference implementation."""

    MODEL_ID = "pi0-fast"


__all__ = ["Pi0FastSynthesis", "Pi0Synthesis", "Pi05Synthesis"]
