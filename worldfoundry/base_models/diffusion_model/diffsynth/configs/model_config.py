"""Module for base_models -> diffusion_model -> diffsynth -> configs -> model_config.py functionality."""

from typing_extensions import TypeAlias

from ..models.wan_video_dit import WanModel
from ..models.wan_video_pusa import WanModelPusa
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_vace import VaceWanModel
from worldfoundry.core.model_loading import load_model_loader_registry


_MODEL_CLASSES = {
    "WanModel": WanModel,
    "WanModelPusa": WanModelPusa,
    "WanTextEncoder": WanTextEncoder,
    "WanImageEncoder": WanImageEncoder,
    "WanVideoVAE": WanVideoVAE,
    "WanMotionControllerModel": WanMotionControllerModel,
    "VaceWanModel": VaceWanModel,
}

_registry = load_model_loader_registry("models/runtime/configs/diffsynth/wan_model_loader.yaml", _MODEL_CLASSES)

model_loader_configs = _registry.model_loader_configs
huggingface_model_loader_configs = _registry.huggingface_model_loader_configs
patch_model_loader_configs = _registry.patch_model_loader_configs
preset_models_on_huggingface = _registry.preset_models_on_huggingface
preset_models_on_modelscope = _registry.preset_models_on_modelscope
preset_model_ids = _registry.preset_model_ids
preset_model_websites = _registry.preset_model_websites

Preset_model_id: TypeAlias = str
Preset_model_website: TypeAlias = str
