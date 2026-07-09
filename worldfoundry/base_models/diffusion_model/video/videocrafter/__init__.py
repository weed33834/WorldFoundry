"""Module for base_models -> diffusion_model -> video -> videocrafter -> __init__.py functionality."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoCrafterComponents:
    """Video crafter components implementation."""
    batch_ddim_sampling: object
    load_image_batch: object
    load_model_checkpoint: object
    instantiate_from_config: object


def load_videocrafter_components() -> VideoCrafterComponents:
    """Load packaged VideoCrafter inference components lazily."""

    from .videocrafter_runtime.inference import (
        batch_ddim_sampling,
        load_image_batch,
        load_model_checkpoint,
    )
    from worldfoundry.base_models.diffusion_model.video.lvdm.utils import instantiate_from_config

    return VideoCrafterComponents(
        batch_ddim_sampling=batch_ddim_sampling,
        load_image_batch=load_image_batch,
        load_model_checkpoint=load_model_checkpoint,
        instantiate_from_config=instantiate_from_config,
    )


__all__ = [
    "VideoCrafter",
    "VideoCrafterComponents",
    "load_videocrafter_components",
    "resolve_runtime_config",
]


def __getattr__(name: str):
    """Getattr.

    Args:
        name: The name.
    """
    if name in {"VideoCrafter", "resolve_runtime_config"}:
        from .worldfoundry_runtime import VideoCrafter, resolve_runtime_config

        return {"VideoCrafter": VideoCrafter, "resolve_runtime_config": resolve_runtime_config}[name]
    raise AttributeError(name)
