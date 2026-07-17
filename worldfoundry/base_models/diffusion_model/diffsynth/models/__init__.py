"""Module for base_models -> diffusion_model -> diffsynth -> models -> __init__.py functionality."""

from importlib import import_module

from worldfoundry.core.model_loading import load_state_dict as load_state_dict

_EXPORTS = {
    "ModelManager": ".model_manager",
    "CameraPoseEncoder": ".pose_adaptor_ac3d",
    "DynamicRetrievalAttention": ".hydra_wan_video_dit",
    "HyDRAAttentionConfig": ".hydra_wan_video_dit",
    "MemoryTokenizer": ".hydra_wan_video_dit",
    "SimpleAdapter": ".wan_video_camera_controller",
    "WanS2VAudioEncoder": ".wav2vec",
    "WanS2VModel": ".wan_video_dit_s2v",
    "generate_camera_coordinates": ".wan_video_camera_controller",
    "get_sample_indices": ".wav2vec",
    "process_pose_file": ".wan_video_camera_controller",
    "rope_precompute": ".wan_video_dit_s2v",
    "configure_hydra_model": ".hydra_wan_video_dit",
}

__all__ = sorted((*_EXPORTS, "load_state_dict"))


def __getattr__(name):
    """Getattr.

    Args:
        name: The name.
    """
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
