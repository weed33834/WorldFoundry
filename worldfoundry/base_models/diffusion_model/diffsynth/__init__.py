"""Module for base_models -> diffusion_model -> diffsynth -> __init__.py functionality."""

from importlib import import_module

_EXPORTS = {
    "FlowMatchScheduler": ".schedulers",
    "ModelManager": ".models",
    "ModelManagerWan22": ".models.model_manager_wan22",
    "WanPrompter": ".prompters",
    "Wan22VideoPusaMultiFramesPipeline": ".pipelines.wan22_video_pusa_multi_frames",
    "Wan22VideoPusaPipeline": ".pipelines.wan22_video_pusa",
    "WanUniAnimateVideoPipeline": ".pipelines",
    "WanVideoPipeline": ".pipelines",
    "load_state_dict": ".models",
}

__all__ = sorted(_EXPORTS)


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
