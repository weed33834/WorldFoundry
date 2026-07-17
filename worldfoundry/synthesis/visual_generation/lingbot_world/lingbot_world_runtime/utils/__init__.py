"""Lazy exports for the LingBot utilities reused from WorldFoundry base models."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "FlowDPMSolverMultistepScheduler",
    "FlowUniPCMultistepScheduler",
    "HuggingfaceTokenizer",
    "compute_relative_poses",
    "get_plucker_embeddings",
    "get_sampling_sigmas",
    "interpolate_camera_poses",
    "retrieve_timesteps",
]

_EXPORTS = {
    "FlowDPMSolverMultistepScheduler": (
        "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.utils.fm_solvers",
        "FlowDPMSolverMultistepScheduler",
    ),
    "get_sampling_sigmas": (
        "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.utils.fm_solvers",
        "get_sampling_sigmas",
    ),
    "retrieve_timesteps": (
        "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.utils.fm_solvers",
        "retrieve_timesteps",
    ),
    "FlowUniPCMultistepScheduler": (
        "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.utils.fm_solvers_unipc",
        "FlowUniPCMultistepScheduler",
    ),
    "HuggingfaceTokenizer": (
        "worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.tokenizers",
        "HuggingfaceTokenizer",
    ),
    "compute_relative_poses": (__name__ + ".cam_utils", "compute_relative_poses"),
    "interpolate_camera_poses": (__name__ + ".cam_utils", "interpolate_camera_poses"),
    "get_plucker_embeddings": (__name__ + ".cam_utils", "get_plucker_embeddings"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
