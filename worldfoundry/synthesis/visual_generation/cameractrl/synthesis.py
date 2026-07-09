"""
Adapter module for the CameraCtrl synthesis model.

This module provides the `CameraCtrlSynthesis` class, which implements the `BaseSynthesis`
interface by wrapping the internal `CameraCtrlRuntime`. It facilitates interaction
with the CameraCtrl model for tasks like generating camera control sequences or videos
based on prompts and other inputs.

It also exposes various default configuration constants and paths relevant to
CameraCtrl models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ...base_synthesis import BaseSynthesis
from .runtime import (
    DEFAULT_CAMERACTRL_CKPT,
    DEFAULT_CAMERACTRL_CONFIG,
    DEFAULT_CAMERACTRL_IMAGE_LORA,
    DEFAULT_SD15_ROOT,
    CameraCtrlRuntime,
)


class CameraCtrlSynthesis(BaseSynthesis):
    """
    A synthesis adapter for the CameraCtrl model, implementing the `BaseSynthesis` interface.

    This class provides a high-level API to interact with the underlying `CameraCtrlRuntime`
    for generating content based on prompts, images, or videos, abstracting away
    the low-level runtime details.
    """

    MODEL_ID = CameraCtrlRuntime.MODEL_ID
    DISPLAY_NAME = CameraCtrlRuntime.DISPLAY_NAME

    def __init__(self, runtime: CameraCtrlRuntime) -> None:
        """
        Initializes the CameraCtrl synthesis adapter with a given runtime instance.

        The adapter copies essential configuration attributes from the provided runtime
        for convenient access and maintains a reference to the runtime for all core operations.

        Args:
            runtime: An instance of `CameraCtrlRuntime` that handles the actual model operations.
        """
        super().__init__()
        self.runtime = runtime
        self.model_id = runtime.model_id
        self.model_name = runtime.model_name
        self.generation_type = runtime.generation_type
        self.device = runtime.device
        self.sd15_path = runtime.sd15_path
        self.pose_adaptor_ckpt = runtime.pose_adaptor_ckpt
        self.model_config = runtime.model_config
        self.motion_module_ckpt = runtime.motion_module_ckpt
        self.image_lora_ckpt = runtime.image_lora_ckpt
        self.image_lora_rank = runtime.image_lora_rank
        self.unet_subfolder = runtime.unet_subfolder
        self.personalized_base_model = runtime.personalized_base_model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "CameraCtrlSynthesis":
        """
        Creates a `CameraCtrlSynthesis` instance from a pretrained model.

        This class method acts as a factory, delegating the model loading and
        initialization to the underlying `CameraCtrlRuntime`.

        Args:
            pretrained_model_path: Path to the pretrained model or a model identifier.
            args: Deprecated argument; it will be ignored if provided.
            device: The device to load the model on (e.g., 'cuda', 'cpu').
            model_id: An optional identifier for the model.
            **kwargs: Additional keyword arguments passed directly to `CameraCtrlRuntime.from_pretrained`.

        Returns:
            An initialized `CameraCtrlSynthesis` instance.
        """
        del args  # Discard the 'args' parameter as it is deprecated and not used by the runtime.
        runtime = CameraCtrlRuntime.from_pretrained(
            pretrained_model_path,
            device=device,
            model_id=model_id,
            **kwargs,
        )
        return cls(runtime)

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generates content using the CameraCtrl model based on the provided inputs.

        This method acts as a proxy, delegating the actual prediction task to
        the underlying `CameraCtrlRuntime` instance with the given parameters.

        Args:
            prompt: The text prompt to guide the generation. Defaults to an empty string.
            images: Input images for the generation, if applicable. Can be a path or loaded image data.
            video: Input video for the generation, if applicable. Can be a path or loaded video data.
            interactions: A sequence of interaction instructions (e.g., camera movements)
                          to guide the generation. Defaults to an empty sequence.
            output_path: The file path to save the generated output (e.g., a video file).
                         If None, the runtime determines the default or returns in memory.
            fps: Frames per second for video generation. If None, uses a default from the runtime.
            **kwargs: Additional keyword arguments passed directly to `CameraCtrlRuntime.predict`.

        Returns:
            A dictionary containing the results of the prediction, typically including
            the path to the generated output and other metadata.
        """
        return self.runtime.predict(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            **kwargs,
        )


__all__ = [
    "CameraCtrlSynthesis",
    "DEFAULT_CAMERACTRL_CKPT",
    "DEFAULT_CAMERACTRL_CONFIG",
    "DEFAULT_CAMERACTRL_IMAGE_LORA",
    "DEFAULT_SD15_ROOT",
]