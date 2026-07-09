"""Compatibility namespace for ViPE prior modules.

Canonical owners are split by capability:
``three_dimensions.depth``, ``three_dimensions.general_3d.geocalib``, and
``perception_core.tracking.track_anything``.
"""

from pathlib import Path

_THREE_DIMENSIONS = Path(__file__).resolve().parents[3]

__path__ = [
    str(_THREE_DIMENSIONS),
    str(_THREE_DIMENSIONS / "general_3d"),
    str(_THREE_DIMENSIONS / "optical_flow"),
]
