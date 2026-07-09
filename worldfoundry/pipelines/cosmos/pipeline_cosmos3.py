"""Cosmos3 visual generation pipeline module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..pipeline_utils import PipelineABC
from ...base_models.diffusion_model.video.cosmos3 import DEFAULT_COSMOS3_REPO_ID
from ...synthesis.visual_generation.cosmos.cosmos3_synthesis import Cosmos3Synthesis


def _video_to_thwc_uint8(video: Any) -> np.ndarray:
    """Normalize Cosmos3 decoded video output to Studio's THWC uint8 contract."""

    if isinstance(video, (list, tuple)):
        if len(video) != 1:
            raise ValueError(f"Expected one Cosmos3 video sample, got {len(video)} samples.")
        video = video[0]

    if isinstance(video, torch.Tensor):
        arr = video.detach().cpu().float().numpy()
    else:
        arr = np.asarray(video)

    if arr.ndim == 5:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for Cosmos3 video output, got shape {arr.shape}")
        arr = arr[0]

    if arr.ndim != 4:
        raise ValueError(f"Expected Cosmos3 video output in 4D format, got shape {arr.shape}")

    if arr.shape[-1] in {1, 3, 4}:
        pass
    elif arr.shape[0] in {1, 3, 4}:
        arr = np.transpose(arr, (1, 2, 3, 0))
    elif arr.shape[1] in {1, 3, 4}:
        arr = np.transpose(arr, (0, 2, 3, 1))
    else:
        raise ValueError(f"Expected Cosmos3 video output in THWC-compatible shape, got {arr.shape}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if not np.isfinite(arr).all():
                raise ValueError("Cosmos3 video output contains NaN or infinite values.")
            arr = np.clip(arr * 255.0 if arr.max(initial=0.0) <= 1.0 else arr, 0, 255)
        else:
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)

    return arr


class Cosmos3Pipeline(PipelineABC):
    """WorldFoundry pipeline wrapper for the in-tree Cosmos3 base runtime."""

    MODEL_ID = "cosmos3"

    def __init__(
        self,
        synthesis_model: Cosmos3Synthesis | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        super().__init__(
            model_id=model_id or self.MODEL_ID,
            synthesis_model=synthesis_model,
            device=device,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "Cosmos3Pipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        options = dict(required_components or {})
        options.update(kwargs)
        if model_path is None:
            model_path = options.pop("pretrained_model_path", DEFAULT_COSMOS3_REPO_ID)

        synthesis_model = Cosmos3Synthesis.from_pretrained(
            model_path=model_path,
            device=device,
            **options,
        )
        return cls(
            synthesis_model=synthesis_model,
            device=device,
            model_id=model_id or cls.MODEL_ID,
        )

    @classmethod
    def plan(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Plan for Cosmos3Pipeline."""
        options = dict(required_components or {})
        options.update(kwargs)
        if model_path is None:
            model_path = options.pop("pretrained_model_path", None)
        return Cosmos3Synthesis.plan(model_path=model_path, **options)

    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        image: Any = None,
        images: Any = None,
        image_path: str | None = None,
        input_path: str | None = None,
        output_path: str | None = None,
        output_dir: str | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float | None = None,
        seed: int | None = None,
        output_type: str = "video",
        **kwargs: Any,
    ) -> Any:
        """Execute the complete pipeline generation flow."""
        del output_path, output_dir
        input_image = image if image is not None else images
        if input_image is None:
            input_image = image_path or input_path
        if isinstance(input_image, (str, Path)):
            from PIL import Image

            input_image = Image.open(input_image).convert("RGB")
        generation_kwargs = dict(kwargs)
        generation_kwargs.pop("task", None)
        generation_kwargs.pop("task_type", None)
        for key, value in {
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "fps": fps,
            "seed": seed,
        }.items():
            if value is not None:
                generation_kwargs[key] = value
        result = self.synthesis_model.predict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            output_type=output_type,
            **generation_kwargs,
        )
        if output_type == "video":
            return _video_to_thwc_uint8(result)
        return result


__all__ = ["Cosmos3Pipeline"]
