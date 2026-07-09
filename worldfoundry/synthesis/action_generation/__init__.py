"""Action, policy, and latent-action synthesis packages.

Concrete synthesis classes are exposed from their model-specific packages.
"""

from .base_action_synthesis import ActionModelSynthesis

__all__ = ["ActionModelSynthesis"]
