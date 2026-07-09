"""A wrapper for the LTX-Video image-to-video generation pipeline.

This module provides a class `LTXVideo` to easily interact with the LTX-Video model
for generating videos from a text prompt and an initial conditioning image.
It handles model loading, device selection, seeding, and video generation,
abstracting away the underlying Diffusers pipeline details.
"""

from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image


def _load_image(image_path: str, height: int, width: int) -> Image.Image:
    """Load an RGB conditioning image resized for LTX image-to-video.

    Args:
        image_path: Local image file path used as the first video frame.
        height: Target frame height in pixels.
        width: Target frame width in pixels.
    """
    return Image.open(Path(image_path).expanduser()).convert("RGB").resize((width, height))


def _seed_everything(seed: int) -> None:
    """Seed CPU and accelerator RNGs before a deterministic generation call.

    Args:
        seed: Integer seed propagated to Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _select_device() -> str:
    """Choose the best available local inference device.

    Args:
        None.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LTXVideo:
    """A wrapper class for the LTX-Video image-to-video generation pipeline.

    This class simplifies the process of loading the LTX-Video model and generating
    videos from a text prompt and an initial conditioning image. It encapsulates
    configuration parameters and the Diffusers pipeline for a streamlined user experience.
    """

    def __init__(
        self,
        model_name: str,
        generation_type: Literal["t2v", "i2v"],
        num_images_per_prompt: int,
        image_cond_noise_scale: float,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: int,
        pipeline_config: str | None = None,
        negative_prompt: str = "",
        conditioning_start_frames: int = 0,
        model_path: str = "Lightricks/LTX-Video",
        seed: int = 171198,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        device: Optional[str] = None,
        torch_dtype: str = "bfloat16",
    ) -> None:
        """Create the in-tree LTX image-to-video runtime wrapper.

        Args:
            model_name: Registry name for the WorldFoundry video model.
            generation_type: Generation mode, currently only image-to-video is supported.
            num_images_per_prompt: Number of generated samples per prompt.
            image_cond_noise_scale: Image conditioning noise scale passed to compatible pipelines.
            height: Output video height in pixels.
            width: Output video width in pixels.
            num_frames: Number of frames to generate.
            frame_rate: Output frame rate metadata.
            pipeline_config: Deprecated external LTX-Video config path kept for config compatibility.
            negative_prompt: Negative prompt used by the pipeline.
            conditioning_start_frames: Expected conditioning frame index; only zero is supported.
            model_path: Hugging Face repo ID or local checkpoint directory for weights.
            seed: Deterministic seed for generation.
            num_inference_steps: Denoising step count.
            guidance_scale: Classifier-free guidance scale.
            device: Optional explicit PyTorch device.
            torch_dtype: Floating point dtype name used when loading weights.
        """
        del pipeline_config  # Remove deprecated pipeline_config argument as it's no longer used.
        if generation_type != "i2v":
            raise ValueError("LTXVideo only supports image-to-video generation in WorldFoundry.")
        if conditioning_start_frames != 0:
            raise ValueError("LTXVideo only supports conditioning_start_frames=0.")
        self.model_name = model_name
        self.generation_type = generation_type
        self.num_images_per_prompt = num_images_per_prompt
        self.image_cond_noise_scale = image_cond_noise_scale
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.negative_prompt = negative_prompt
        self.model_path = model_path
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        # Select device automatically if not explicitly provided.
        self.device = device or _select_device()
        # Convert string dtype (e.g., "bfloat16") to actual torch.dtype object.
        self.torch_dtype = getattr(torch, torch_dtype)
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """Load the packaged Diffusers LTX image-to-video pipeline.

        Returns:
            The initialized and device-mapped LTXImageToVideoPipeline instance.
        """
        from diffusers import LTXImageToVideoPipeline

        pipeline = LTXImageToVideoPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
        )
        return pipeline.to(self.device)

    def generate_video(self, prompt: str, image_path: Optional[str] = None):
        """Generate a video tensor from a prompt and initial image.

        Args:
            prompt: Text prompt describing the target video.
            image_path: Local conditioning image path required for image-to-video generation.

        Returns:
            A PyTorch tensor containing the generated video frames.
        """
        if image_path is None:
            raise ValueError("LTX image-to-video generation requires image_path.")
        _seed_everything(self.seed)
        image = _load_image(image_path, self.height, self.width)
        # Create a PyTorch generator for deterministic random number generation on the specified device.
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        output = self.pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            width=self.width,
            height=self.height,
            num_frames=self.num_frames,
            frame_rate=self.frame_rate,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            num_videos_per_prompt=self.num_images_per_prompt,
            generator=generator,
            output_type="pt",
        )
        # Extract the first video (assuming num_images_per_prompt typically results in one video).
        frames = output.frames[0]
        if torch.cuda.is_available():
            # Clear CUDA memory and synchronize streams to free up resources after generation.
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Explicitly run garbage collection to release CPU memory.
        gc.collect()
        return frames