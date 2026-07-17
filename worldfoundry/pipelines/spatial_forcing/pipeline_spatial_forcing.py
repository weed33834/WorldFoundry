"""Independent action pipeline for Spatial-Forcing policies."""

from __future__ import annotations

from worldfoundry.operators.vla_native_operator import OpenVLAOFTOperator
from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class SpatialForcingOperator(OpenVLAOFTOperator):
    """Official two-camera LIBERO observation contract for Spatial-Forcing."""

    MODEL_ID = "spatial-forcing"
    POLICY_FAMILY = "spatial_forcing_openvla_action_chunk_policy"


class SpatialForcingPipeline(ComponentPipeline):
    """WorldFoundry VLA pipeline for Spatial-Forcing action generation."""

    MODEL_ID = "spatial-forcing"
    MODEL_PATH_OPTION = "checkpoint_path"
    OPERATOR_CLS = SpatialForcingOperator
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.spatial_forcing:SpatialForcingSynthesis"
    generation_type = "vla_policy"


__all__ = ["SpatialForcingOperator", "SpatialForcingPipeline"]
