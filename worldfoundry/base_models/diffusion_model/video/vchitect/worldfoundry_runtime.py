"""Module for base_models -> diffusion_model -> video -> vchitect -> worldfoundry_runtime.py functionality."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

from .vchitect_runtime import DEFAULT_CHECKPOINT_DIR, load_vchitect_components


class Vchitect:
    """Vchitect implementation."""
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        generation_type: Literal["t2v", "i2v"] = "t2v",
        num_frames: int = 40,
        width: int = 768,
        height: int = 432,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 100,
        negative_prompt: str = "",
        checkpoint_root: str | None = None,
        device: str = "cuda",
    ):
        """Build the in-tree Vchitect-2 T2V runtime."""
        if generation_type != "t2v":
            raise ValueError("Vchitect runtime currently supports only t2v generation.")

        self.model_name = model_name
        self.model_path = checkpoint_root or model_path or DEFAULT_CHECKPOINT_DIR
        self.generation_type = generation_type
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.negative_prompt = negative_prompt
        self.device = device

        components = load_vchitect_components()
        self.pipe = components.pipeline_cls(self.model_path, device=device)

    def generate_video(self, prompt: str, image_path: str | None = None):
        """Generate a text-to-video sample using the vendored Vchitect pipeline."""
        if image_path is not None:
            raise ValueError("Vchitect is a text-to-video model.")

        import torch

        use_cuda_autocast = str(self.device).startswith("cuda") and torch.cuda.is_available()
        autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_cuda_autocast else nullcontext()
        with autocast:
            return self.pipe(
                prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                width=self.width,
                height=self.height,
                frames=self.num_frames,
            )


__all__ = ["Vchitect"]
