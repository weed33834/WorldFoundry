"""Synthesis facade for the in-tree RollingForcing runtime."""

from worldfoundry.synthesis.visual_generation.forcing.forcing_synthesis import _BaseForcingSynthesis

from .runtime import RollingForcingRuntime


class RollingForcingSynthesis(_BaseForcingSynthesis):
    """Inference-only RollingForcing text-to-video synthesis."""

    MODEL_ID = "rolling-forcing"
    DISPLAY_NAME = "RollingForcing"
    RUNTIME_CLS = RollingForcingRuntime


__all__ = ["RollingForcingSynthesis"]
