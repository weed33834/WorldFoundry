from __future__ import annotations

from .models.wan_video_dit import WanModel
from worldfoundry.base_models.diffusion_model.diffsynth.models.wan_video_text_encoder import WanTextEncoder
from worldfoundry.base_models.diffusion_model.diffsynth.models.wan_video_image_encoder import WanImageEncoder
from worldfoundry.base_models.diffusion_model.diffsynth.models.wan_video_vae import WanVideoVAE
from worldfoundry.core.model_loading import load_model_loader_registry


_CONFIG_RESOURCE = "models/runtime/configs/kling/astra/model_loader.yaml"

_MODEL_CLASSES = {
    "WanModel": WanModel,
    "WanTextEncoder": WanTextEncoder,
    "WanImageEncoder": WanImageEncoder,
    "WanVideoVAE": WanVideoVAE,
}


_registry = load_model_loader_registry(_CONFIG_RESOURCE, _MODEL_CLASSES)
model_loader_configs = _registry.model_loader_configs
huggingface_model_loader_configs = _registry.huggingface_model_loader_configs
patch_model_loader_configs = _registry.patch_model_loader_configs

__all__ = [
    "huggingface_model_loader_configs",
    "model_loader_configs",
    "patch_model_loader_configs",
]
