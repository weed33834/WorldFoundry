"""WorldFoundry pipeline for Uni3C."""

from __future__ import annotations

from ..component_pipelines import ComponentPipeline


class Uni3CPipeline(ComponentPipeline):
    """Run Uni3C camera control or unified camera-and-human control."""

    MODEL_ID = "uni3c"
    OPERATOR_TARGET = "worldfoundry.operators.uni3c_operator:Uni3COperator"
    MEMORY_TARGET = "worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.visual_generation.uni3c:Uni3CSynthesis"
    generation_type = "camera_human_motion_controlled_video"


__all__ = ["Uni3CPipeline"]
