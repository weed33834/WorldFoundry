"""Runtime visual generation pipeline module."""

from __future__ import annotations

from ...synthesis.visual_generation.memory.runtime import RuntimeMemory
from ...operators.three_d_four_d_runtime_operator import ThreeDFourDRuntimeOperator
from ...synthesis.visual_generation.three_d_four_d.runtime import (
    ThreeDFourDRuntimeSynthesis,
)
from ..pipeline_utils import PipelineABC


class ThreeDFourDRuntimePipeline(PipelineABC):
    """WorldFoundry pipeline for 3D/4D model runtime entrypoints."""

    MODEL_ID = "three-d-four-d-runtime"
    OPERATOR_CLS = ThreeDFourDRuntimeOperator
    MEMORY_CLS = RuntimeMemory
    SYNTHESIS_CLS = ThreeDFourDRuntimeSynthesis
    MEMORY_RECORD_TYPE = "three_d_four_d_runtime"
    generation_type = "three_d_four_d"


class FourDGSPipeline(ThreeDFourDRuntimePipeline):
    """Pipeline implementation for FourDGS visual generation."""
    MODEL_ID = "4d-gs"


class LagrNVSPipeline(ThreeDFourDRuntimePipeline):
    """Pipeline implementation for LagrNVS visual generation."""
    MODEL_ID = "lagernvs"


class MonST3RPipeline(ThreeDFourDRuntimePipeline):
    """Pipeline implementation for MonST3R visual generation."""
    MODEL_ID = "monst3r"


class MVDiffusionPipeline(ThreeDFourDRuntimePipeline):
    """Pipeline implementation for MVDiffusion visual generation."""
    MODEL_ID = "mvdiffusion"


class ShapeOfMotionPipeline(ThreeDFourDRuntimePipeline):
    """Pipeline implementation for ShapeOfMotion visual generation."""
    MODEL_ID = "shape-of-motion"


class StableVirtualCameraPipeline(ThreeDFourDRuntimePipeline):
    """Pipeline implementation for StableVirtualCamera visual generation."""
    MODEL_ID = "stable-virtual-camera"


class WonderJourneyPipeline(ThreeDFourDRuntimePipeline):
    """Pipeline implementation for WonderJourney visual generation."""
    MODEL_ID = "wonderjourney"


class WonderWorldPipeline(ThreeDFourDRuntimePipeline):
    """Pipeline implementation for WonderWorld visual generation."""
    MODEL_ID = "wonderworld"


class WorldGenPipeline(ThreeDFourDRuntimePipeline):
    """Pipeline implementation for WorldGen visual generation."""
    MODEL_ID = "worldgen"


__all__ = [
    "FourDGSPipeline",
    "LagrNVSPipeline",
    "MonST3RPipeline",
    "MVDiffusionPipeline",
    "ShapeOfMotionPipeline",
    "StableVirtualCameraPipeline",
    "ThreeDFourDRuntimePipeline",
    "WonderJourneyPipeline",
    "WonderWorldPipeline",
    "WorldGenPipeline",
]
