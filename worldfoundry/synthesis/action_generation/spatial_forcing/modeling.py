"""Inference model surface for Spatial-Forcing's released OpenVLA checkpoints.

Spatial-Forcing changes training by aligning VLA features to VGGT features.  Its
released LIBERO policies use the same inference graph as OpenVLA-OFT; neither
VGGT nor the alignment projector is consulted at inference time.  Re-exporting
the already vendored, inference-pruned Prismatic implementation keeps this
integration flat and avoids a second copy of identical model code.
"""

from worldfoundry.synthesis.action_generation.openvla_oft.config import OpenVLAConfig
from worldfoundry.synthesis.action_generation.openvla_oft.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NormalizationType,
)
from worldfoundry.synthesis.action_generation.openvla_oft.modeling.action_heads import (
    L1RegressionActionHead,
)
from worldfoundry.synthesis.action_generation.openvla_oft.modeling.model import (
    OpenVLAForActionPrediction,
)
from worldfoundry.synthesis.action_generation.openvla_oft.modeling.projectors import (
    ProprioProjector,
)
from worldfoundry.synthesis.action_generation.openvla_oft.preprocessing import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)

__all__ = [
    "ACTION_DIM",
    "ACTION_PROPRIO_NORMALIZATION_TYPE",
    "NUM_ACTIONS_CHUNK",
    "PROPRIO_DIM",
    "L1RegressionActionHead",
    "NormalizationType",
    "OpenVLAConfig",
    "OpenVLAForActionPrediction",
    "PrismaticImageProcessor",
    "PrismaticProcessor",
    "ProprioProjector",
]
