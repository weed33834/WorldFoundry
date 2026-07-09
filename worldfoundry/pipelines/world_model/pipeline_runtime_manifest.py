"""Runtime Manifest visual generation pipeline module."""

from __future__ import annotations

from ...synthesis.visual_generation.memory.runtime import RuntimeMemory
from ...operators.world_model_runtime_operator import WorldModelRuntimeOperator
from ...synthesis.visual_generation.world_model.runtime_manifest import WorldModelRuntimeSynthesis
from ..pipeline_utils import PipelineABC


class WorldModelRuntimePipeline(PipelineABC):
    """Pipeline surface for vendored world-model runtimes with asset-gated execution."""

    MODEL_ID = "world-model-runtime"
    OPERATOR_CLS = WorldModelRuntimeOperator
    MEMORY_CLS = RuntimeMemory
    SYNTHESIS_CLS = WorldModelRuntimeSynthesis
    MEMORY_RECORD_TYPE = "world_model_runtime_plan"
    generation_type = "world_model"

    def __call__(self, *args, **kwargs):
        """Execute the complete pipeline generation flow."""
        kwargs.setdefault("return_dict", True)
        return super().__call__(*args, **kwargs)


class AdaWorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for AdaWorld visual generation."""
    MODEL_ID = "adaworld"


class CtrlWorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for CtrlWorld visual generation."""
    MODEL_ID = "ctrl-world"


class DIAMONDPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for DIAMOND visual generation."""
    MODEL_ID = "diamond"


class DinoWMPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for DinoWM visual generation."""
    MODEL_ID = "dino-wm"


class GenieEnvisionerPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for GenieEnvisioner visual generation."""
    MODEL_ID = "genie-envisioner"


class GigaWorld0Pipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for GigaWorld0 visual generation."""
    MODEL_ID = "giga-world-0"


class LeWorldModelPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for LeWorldModel visual generation."""
    MODEL_ID = "leworldmodel"


class MineWorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for MineWorld visual generation."""
    MODEL_ID = "mineworld"


class NWMPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for NWM visual generation."""
    MODEL_ID = "nwm"


class Oasis500MPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for Oasis500M visual generation."""
    MODEL_ID = "oasis-500m"


class SanaWMPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for SanaWM visual generation."""
    MODEL_ID = "sana-wm"


class StarWMPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for StarWM visual generation."""
    MODEL_ID = "starwm"


class TesserActPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for TesserAct visual generation."""
    MODEL_ID = "tesseract"


class DreamXWorld5BCamPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for DreamXWorld5BCam visual generation."""
    MODEL_ID = "dreamx-world-5b-cam"


class DROIDWPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for DROIDW visual generation."""
    MODEL_ID = "droid-w"


class EgoWMPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for EgoWM visual generation."""
    MODEL_ID = "egowm"


class HappyOysterPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for HappyOyster visual generation."""
    MODEL_ID = "happyoyster"


class HMAPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for HMA visual generation."""
    MODEL_ID = "hma"


class HunyuanWorld1Pipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for HunyuanWorld1 visual generation."""
    MODEL_ID = "hunyuanworld-1"


class MosaicMemPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for MosaicMem visual generation."""
    MODEL_ID = "mosaicmem"


class MotionBricksPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for MotionBricks visual generation."""
    MODEL_ID = "motionbricks"


class OmniForcingPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for OmniForcing visual generation."""
    MODEL_ID = "omniforcing"


class PointWorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for PointWorld visual generation."""
    MODEL_ID = "pointworld"


class ShotStreamPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for ShotStream visual generation."""
    MODEL_ID = "shotstream"


class SimWorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for SimWorld visual generation."""
    MODEL_ID = "simworld"


class UWMPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for UWM visual generation."""
    MODEL_ID = "uwm"


class VGGTWorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for VGGTWorld visual generation."""
    MODEL_ID = "vggt-world"


class Vid2WorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for Vid2World visual generation."""
    MODEL_ID = "vid2world"


class ViewCrafterPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for ViewCrafter visual generation."""
    MODEL_ID = "viewcrafter"


class WildDet3DPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for WildDet3D visual generation."""
    MODEL_ID = "wilddet3d"


class WildWorldPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for WildWorld visual generation."""
    MODEL_ID = "wildworld"


class WorldGrowPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for WorldGrow visual generation."""
    MODEL_ID = "worldgrow"


class WorldMemPipeline(WorldModelRuntimePipeline):
    """Pipeline implementation for WorldMem visual generation."""
    MODEL_ID = "worldmem"


__all__ = [
    "AdaWorldPipeline",
    "CtrlWorldPipeline",
    "DIAMONDPipeline",
    "DinoWMPipeline",
    "DROIDWPipeline",
    "DreamXWorld5BCamPipeline",
    "EgoWMPipeline",
    "GenieEnvisionerPipeline",
    "GigaWorld0Pipeline",
    "HMAPipeline",
    "HappyOysterPipeline",
    "HunyuanWorld1Pipeline",
    "LeWorldModelPipeline",
    "MineWorldPipeline",
    "MosaicMemPipeline",
    "MotionBricksPipeline",
    "NWMPipeline",
    "OmniForcingPipeline",
    "Oasis500MPipeline",
    "PointWorldPipeline",
    "SanaWMPipeline",
    "ShotStreamPipeline",
    "SimWorldPipeline",
    "StarWMPipeline",
    "TesserActPipeline",
    "UWMPipeline",
    "VGGTWorldPipeline",
    "Vid2WorldPipeline",
    "ViewCrafterPipeline",
    "WildDet3DPipeline",
    "WildWorldPipeline",
    "WorldModelRuntimePipeline",
    "WorldGrowPipeline",
    "WorldMemPipeline",
]
