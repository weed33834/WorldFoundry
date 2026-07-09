"""Defines the ActionModelSynthesis class, a specialized runtime profile for action models.

This module provides the base structure for integrating action models into the evaluation
framework by inheriting from `RuntimeProfileSynthesis`. It establishes the model's
identifier and sets the stage for defining its architecture, runtime, and inference path.
"""
from __future__ import annotations

from worldfoundry.evaluation.models.runtime.profiles import RuntimeProfileSynthesis


class ActionModelSynthesis(RuntimeProfileSynthesis):
    """Profile-backed action-model surface.

    This class serves as a base for action models within the evaluation framework.
    Thin subclasses define identity and artifact contracts only. A model becomes
    a full integration after its architecture/runtime and real inference path
    are present in-tree and the runtime profile is promoted accordingly.
    """

    MODEL_ID = "action-model"