"""
This module provides utilities and a class for generating video using LTX-2.3 models.

It includes functions for managing device selection, seeding random generators,
resolving model checkpoints, and preparing input parameters. The core
functionality is encapsulated in the LTX2Video class, which supports
both 'diffusers' and 'distilled' pipeline variants for image-to-video generation.
"""

from __future__ import annotations

import gc
import os
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np

# Upstream LTX-2.3 low-memory demos launch with expandable CUDA segments.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from PIL import Image  # noqa: E402


# LTX-2 distilled inference schedule published by Lightricks and mirrored by
# Diffusers. Keep the values in-tree so inference does not depend on a private
# helper in a particular Diffusers release.
_LTX2_DISTILLED_SIGMA_VALUES = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
)


def _is_distilled_single_file(checkpoint_path: str) -> bool:
    """Return whether a standalone checkpoint is an LTX-2 distilled model."""
    path = Path(checkpoint_path).expanduser()
    filename = path.name.lower()
    return path.is_file() and "distilled" in filename and "lora" not in filename


def _load_image(image_path: str, height: int, width: int) -> Image.Image:
    """
    Loads an image from the specified path, converts it to RGB, and resizes it.

    Args:
        image_path: The path to the image file.
        height: The target height for the image.
        width: The target width for the image.

    Returns:
        A PIL Image object, resized and converted to RGB.
    """
    return Image.open(Path(image_path).expanduser()).convert("RGB").resize((width, height))


def _seed_everything(seed: int) -> None:
    """
    Sets the seed for various random number generators to ensure reproducibility.

    This includes Python's built-in `random`, NumPy, and PyTorch (for CPU, CUDA, and MPS).

    Args:
        seed: The integer seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _select_device() -> str:
    """
    Selects the best available compute device (CUDA, MPS, or CPU).

    Returns:
        A string representing the selected device ("cuda", "mps", or "cpu").
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_pipeline_source(checkpoint_path: str) -> tuple[str, bool]:
    """
    Resolves the source path for loading a pipeline, determining if it's a directory
    containing a `model_index.json` or a single file.

    Args:
        checkpoint_path: The initial path to the checkpoint.

    Returns:
        A tuple containing:
        - The resolved path suitable for `from_pretrained` or `from_single_file`.
        - A boolean indicating if the source is a single file (True) or a directory (False).
    """
    path = Path(checkpoint_path).expanduser()
    if path.is_dir():
        # If the path is a directory, assume it's a standard `from_pretrained` source.
        return str(path), False
    # An explicitly selected checkpoint file must be honored even when its
    # parent also contains a Diffusers model_index.json. Falling back to the
    # parent here can silently load a different checkpoint (for example, the
    # dev model instead of ltx-2-19b-distilled.safetensors).
    return str(path), True


def _as_existing_path(value: str, label: str) -> str:
    """
    Validates if a given string path exists on the filesystem.

    Args:
        value: The string path to check.
        label: A descriptive label for the path, used in error messages.

    Returns:
        The absolute string path if it exists.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return str(path)


def _normalize_optional_string(value: Any) -> str | None:
    """
    Normalizes a value to a string or None, treating various falsey-like strings as None.

    Args:
        value: The input value to normalize.

    Returns:
        The normalized string, or None if the input is None, empty, or a common "falsey" string.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"0", "false", "none", "null"}:
        return None
    return text


def _build_quantization_policy(value: Any, checkpoint_path: str) -> Any:
    """
    Builds a quantization policy object based on the provided value and checkpoint path.

    Args:
        value: The quantization policy name (e.g., "fp8", "int8") or an existing policy object.
        checkpoint_path: The path to the model checkpoint, used by the quantization factory.

    Returns:
        A quantization policy object, or None if `value` is None or a "falsey" string.

    Raises:
        ValueError: If an unsupported quantization policy name is provided.
    """
    if value is not None and not isinstance(value, (str, int, float, bool)):
        # If value is already an object that's not a basic type, return it as-is.
        return value

    normalized = _normalize_optional_string(value)
    if normalized is None:
        return None

    # Defer import to avoid circular dependencies and only load when needed.
    from worldfoundry.synthesis.visual_generation.ltx2.ltx_pipelines.utils.quantization_factory import QuantizationKind

    policy_name = normalized.replace("_", "-")
    try:
        # Attempt to convert the normalized policy name to a QuantizationKind and then to a policy object.
        return QuantizationKind(policy_name).to_policy(checkpoint_path=checkpoint_path)
    except ValueError as exc:
        # If conversion fails, list valid quantization kinds for a helpful error message.
        valid = ", ".join(kind.value for kind in QuantizationKind)
        raise ValueError(f"Unsupported LTX-2.3 quantization policy {normalized!r}; expected one of: {valid}") from exc


def _tiling_value(config: dict[str, Any], *keys: str, default: int) -> int:
    """
    Retrieves an integer value from a dictionary using a list of potential keys,
    falling back to a default if none of the keys are found.

    Args:
        config: The dictionary to search within.
        *keys: A variable number of string keys to try in order.
        default: The default integer value to return if no keys are found.

    Returns:
        The integer value corresponding to the first found key, or the default value.
    """
    for key in keys:
        if key in config:
            return int(config[key])
    return default


class LTX2Video:
    """
    A class for generating video frames using LTX-2.3 models.

    This class supports two main pipeline variants: 'diffusers' (for standard Diffusers-based
    LTX-2 models) and 'distilled' (for the LTX-2.3 distilled pipeline). It handles model loading,
    configuration, and video generation from an initial image and a text prompt.
    """

    def __init__(
        self,
        model_name: str,
        generation_type: Literal["i2v"],
        version_hint: str,
        checkpoint_path: str,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: int,
        num_inference_steps: int,
        image_frame_index: int = 0,
        image_strength: float = 1.0,
        enhance_prompt: bool = False,
        negative_prompt: str = "",
        seed: int = 171198,
        guidance_scale: float = 4.0,
        cpu_offload: bool = True,
        device: str | None = None,
        torch_dtype: str = "bfloat16",
        pipeline_variant: str | None = None,
        unsupported_reason: str | None = None,
        spatial_upsampler_path: str | None = None,
        gemma_root: str | None = None,
        offload_mode: str = "cpu",
        quantization: str | None = None,
        vae_tiling: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Initializes the LTX2Video generator.

        Args:
            model_name: The name of the model being used.
            generation_type: The type of generation, currently fixed to "i2v" (image-to-video).
            version_hint: A hint for the model version, e.g., "2.3".
            checkpoint_path: Path to the model checkpoint or a directory containing it.
            height: The height of the generated video frames in pixels.
            width: The width of the generated video frames in pixels.
            num_frames: The total number of frames to generate.
            frame_rate: The frame rate of the generated video.
            num_inference_steps: The number of inference steps for the diffusion process.
            image_frame_index: The frame index where the input image is applied (currently 0).
            image_strength: The strength of the initial image conditioning.
            enhance_prompt: Whether to enhance the prompt for better generation.
            negative_prompt: A prompt describing what should NOT be in the video.
            seed: The random seed for reproducibility.
            guidance_scale: Classifier-free guidance scale.
            cpu_offload: Whether to offload parts of the model to CPU to save GPU memory.
            device: The compute device to use (e.g., "cuda", "mps", "cpu"). If None, automatically selected.
            torch_dtype: The torch data type to use for model weights (e.g., "bfloat16", "float16").
            pipeline_variant: Specifies the pipeline variant: "distilled" or "diffusers".
                               If None, inferred from `version_hint`.
            unsupported_reason: (Deprecated/Ignored) Reason if the model is unsupported.
            spatial_upsampler_path: Required for the 'distilled' pipeline variant, path to the spatial upsampler.
            gemma_root: Required for the 'distilled' pipeline variant, path to the Gemma root directory.
            offload_mode: Offload mode for the 'distilled' pipeline (e.g., "cpu").
            quantization: Quantization policy for the model (e.g., "fp8", "int8").
            vae_tiling: Dictionary configuring VAE tiling for memory optimization.
            **kwargs: Additional keyword arguments (ignored).

        Raises:
            ValueError: If `generation_type` is not "i2v" or `image_frame_index` is not 0.
        """
        del unsupported_reason, kwargs  # Ignore unused parameters
        if generation_type != "i2v":
            raise ValueError("LTX-2 runtime declarations must use i2v generation.")
        if image_frame_index != 0:
            raise ValueError("LTX-2 image-to-video currently supports image_frame_index=0.")

        self.model_name = model_name
        self.generation_type = generation_type
        self.version_hint = version_hint
        self.checkpoint_path = _as_existing_path(checkpoint_path, "checkpoint_path")
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.frame_rate = int(frame_rate)
        self.num_inference_steps = int(num_inference_steps)
        self.image_strength = float(image_strength)
        self.enhance_prompt = bool(enhance_prompt)
        self.negative_prompt = negative_prompt
        self.seed = int(seed)
        self.guidance_scale = float(guidance_scale)
        self.cpu_offload = bool(cpu_offload)
        self.device = device or _select_device()
        self.torch_dtype = getattr(torch, torch_dtype)
        # Determine pipeline variant if not explicitly provided
        self.pipeline_variant = pipeline_variant or ("distilled" if str(version_hint).startswith("2.3") else "diffusers")
        self.is_distilled_diffusers_checkpoint = (
            self.pipeline_variant != "distilled" and _is_distilled_single_file(self.checkpoint_path)
        )
        if self.is_distilled_diffusers_checkpoint:
            # LTX-2 distilled checkpoints are trained for this exact schedule.
            # Applying dev defaults (40 steps and CFG=4) is both slower and can
            # substantially reduce motion quality.
            self.num_inference_steps = len(_LTX2_DISTILLED_SIGMA_VALUES)
            self.guidance_scale = 1.0
        self.spatial_upsampler_path = spatial_upsampler_path
        self.gemma_root = gemma_root
        self.offload_mode = offload_mode
        self.quantization = quantization
        self.vae_tiling = dict(vae_tiling or {})

        # Load the appropriate pipeline based on the determined variant
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """
        Loads the appropriate pipeline ('distilled' or 'diffusers') based on `self.pipeline_variant`.

        Returns:
            The loaded pipeline object.
        """
        if self.pipeline_variant == "distilled":
            return self._load_distilled_pipeline()
        return self._load_diffusers_pipeline()

    def _load_diffusers_pipeline(self):
        """
        Loads the LTX2ImageToVideoPipeline from the `diffusers` library.

        Handles loading from either a directory or a single file checkpoint and
        applies CPU offloading if enabled and supported.

        Returns:
            An instance of `LTX2ImageToVideoPipeline`.
        """
        from diffusers import LTX2ImageToVideoPipeline, LTX2VideoTransformer3DModel

        source, single_file = _resolve_pipeline_source(self.checkpoint_path)
        load_kwargs = {"torch_dtype": self.torch_dtype}
        if single_file:
            # Reuse the adjacent Diffusers component configs and invariant
            # components. This keeps loading fully local while the selected
            # single-file checkpoint supplies the transformer weights.
            config_dir = Path(source).parent
            if (config_dir / "model_index.json").is_file():
                from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

                text_encoder_dir = config_dir / "text_encoder"
                tokenizer_dir = config_dir / "tokenizer"
                if not text_encoder_dir.is_dir():
                    raise FileNotFoundError(
                        "LTX-2 single-file loading requires the adjacent local text_encoder directory: "
                        f"{text_encoder_dir}"
                    )
                if not tokenizer_dir.is_dir():
                    raise FileNotFoundError(
                        "LTX-2 single-file loading requires the adjacent local tokenizer directory: "
                        f"{tokenizer_dir}"
                    )

                # Diffusers' single-file component probe only recognizes an
                # unsharded model.safetensors file. Gemma is sharded here, so
                # load it explicitly from the adjacent local component. This
                # also guarantees that no Hub fallback is attempted.
                text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                    str(text_encoder_dir),
                    torch_dtype=self.torch_dtype,
                    local_files_only=True,
                )
                tokenizer = GemmaTokenizerFast.from_pretrained(
                    str(tokenizer_dir),
                    local_files_only=True,
                )
                # Pipeline-level from_single_file currently fails after loading
                # the LTX-2 transformer because its subsequent VAE conversion
                # incorrectly reports the VAE-prefixed weights as missing.
                # Load only the selected transformer from the single file, then
                # assemble the remaining invariant components from the adjacent
                # local Diffusers directory. Supplying `transformer` prevents
                # from_pretrained from ever reading the parent dev transformer.
                transformer = LTX2VideoTransformer3DModel.from_single_file(
                    source,
                    config=str(config_dir),
                    subfolder="transformer",
                    torch_dtype=self.torch_dtype,
                    local_files_only=True,
                )
                pipeline = LTX2ImageToVideoPipeline.from_pretrained(
                    str(config_dir),
                    transformer=transformer,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    torch_dtype=self.torch_dtype,
                    local_files_only=True,
                )
            else:
                pipeline = LTX2ImageToVideoPipeline.from_single_file(source, **load_kwargs)
        else:
            pipeline = LTX2ImageToVideoPipeline.from_pretrained(source, **load_kwargs)

        # Apply CPU offload if enabled, device is CUDA, and the pipeline supports it.
        # Otherwise, move the entire pipeline to the target device.
        if self.cpu_offload and self.device.startswith("cuda") and hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload(device=self.device)
            return pipeline
        return pipeline.to(self.device)

    def _load_distilled_pipeline(self):
        """
        Loads the LTX-2.3 distilled pipeline.

        Requires `spatial_upsampler_path` and `gemma_root` to be set.
        Applies quantization and offload mode as configured.

        Returns:
            An instance of `DistilledPipeline`.

        Raises:
            ValueError: If `spatial_upsampler_path` or `gemma_root` are not provided.
        """
        if not self.spatial_upsampler_path:
            raise ValueError("LTX-2.3 distilled runtime requires spatial_upsampler_path.")
        if not self.gemma_root:
            raise ValueError("LTX-2.3 distilled runtime requires gemma_root.")

        # Defer imports to only load distilled pipeline components when needed.
        from worldfoundry.synthesis.visual_generation.ltx2.ltx_pipelines.distilled import DistilledPipeline
        from worldfoundry.synthesis.visual_generation.ltx2.ltx_pipelines.utils.types import OffloadMode

        quantization = _build_quantization_policy(self.quantization, self.checkpoint_path)
        return DistilledPipeline(
            distilled_checkpoint_path=self.checkpoint_path,
            spatial_upsampler_path=_as_existing_path(self.spatial_upsampler_path, "spatial_upsampler_path"),
            gemma_root=_as_existing_path(self.gemma_root, "gemma_root"),
            loras=[],  # Currently no LoRAs for LTX-2.3 distilled
            device=torch.device(self.device),
            quantization=quantization,
            offload_mode=OffloadMode(self.offload_mode),
        )

    def _distilled_tiling_config(self):
        """
        Constructs a `TilingConfig` object for the distilled pipeline based on
        the `vae_tiling` configuration in `self.vae_tiling`.

        Returns:
            A `TilingConfig` instance if tiling is enabled and configured,
            or `TilingConfig.default()` if `vae_tiling` is empty,
            or `None` if tiling is explicitly disabled.
        """
        # Defer imports to only load tiling config components when needed.
        from worldfoundry.base_models.diffusion_model.video.ltx2.ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig

        if not self.vae_tiling:
            # If no vae_tiling config is provided, return the default TilingConfig.
            return TilingConfig.default()
        if _normalize_optional_string(self.vae_tiling.get("enabled", "true")) is None:
            # If "enabled" is explicitly set to a falsey value, return None to disable tiling.
            return None

        return TilingConfig(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=_tiling_value(
                    self.vae_tiling,
                    "spatial_tile_size",
                    "tile_size_in_pixels",
                    default=768,
                ),
                tile_overlap_in_pixels=_tiling_value(
                    self.vae_tiling,
                    "spatial_overlap",
                    "tile_overlap_in_pixels",
                    default=64,
                ),
            ),
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=_tiling_value(
                    self.vae_tiling,
                    "temporal_tile_size",
                    "tile_size_in_frames",
                    default=80,
                ),
                tile_overlap_in_frames=_tiling_value(
                    self.vae_tiling,
                    "temporal_overlap",
                    "tile_overlap_in_frames",
                    default=24,
                ),
            ),
        )

    def _generate_distilled_video(self, prompt: str, image_path: str):
        """
        Generates a video using the LTX-2.3 distilled pipeline.

        Args:
            prompt: The text prompt for video generation.
            image_path: The path to the input image.

        Returns:
            A `torch.Tensor` representing the generated video frames.

        Raises:
            RuntimeError: If the distilled pipeline returns no video chunks.
        """
        # Defer imports to only load distilled video generation components when needed.
        from worldfoundry.base_models.diffusion_model.video.ltx2.ltx_core.model.video_vae import get_video_chunks_number
        from worldfoundry.synthesis.visual_generation.ltx2.ltx_pipelines.utils.args import ImageConditioningInput

        # Prepare image conditioning input for the distilled pipeline.
        images = [
            ImageConditioningInput(
                path=_as_existing_path(image_path, "image_path"),
                frame_idx=0,
                strength=self.image_strength,
            )
        ]
        tiling_config = self._distilled_tiling_config()
        # Calculate video chunks number (currently unused but part of API).
        video_chunks_number = get_video_chunks_number(self.num_frames, tiling_config)
        del video_chunks_number  # The variable is computed but not used further in this function.

        with torch.no_grad():
            # Perform video generation using the distilled pipeline.
            video, _audio = self.pipeline(
                prompt=prompt,
                seed=self.seed,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                frame_rate=self.frame_rate,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=self.enhance_prompt,
            )
        chunks = list(video)
        if not chunks:
            raise RuntimeError("LTX-2.3 distilled runtime returned no video chunks.")
        # Concatenate individual video chunks into a single tensor.
        return torch.cat(chunks, dim=0)

    def generate_video(self, prompt: str, image_path: str | None = None):
        """
        Generates a video based on a text prompt and an input image.

        Args:
            prompt: The text prompt to guide video generation.
            image_path: The path to the input image for image-to-video generation.
                        Required for 'i2v' generation type.

        Returns:
            A `torch.Tensor` containing the generated video frames (batch_size, num_frames, C, H, W).

        Raises:
            ValueError: If `image_path` is None for an image-to-video generation task.
        """
        if image_path is None:
            raise ValueError("LTX-2 image-to-video generation requires image_path.")

        _seed_everything(self.seed)

        if self.pipeline_variant == "distilled":
            frames = self._generate_distilled_video(prompt, image_path)
            # Clear CUDA memory and synchronize after generation for resource management.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            return frames

        # Load and resize the input image for the diffusers pipeline.
        image = _load_image(image_path, self.height, self.width)
        # Create a torch generator for reproducible random noise.
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        pipeline_kwargs = dict(
            image=image,
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            width=self.width,
            height=self.height,
            num_frames=self.num_frames,
            frame_rate=self.frame_rate,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            output_type="pt",  # Request output as PyTorch tensors.
        )
        if self.is_distilled_diffusers_checkpoint:
            pipeline_kwargs["sigmas"] = list(_LTX2_DISTILLED_SIGMA_VALUES)
        output = self.pipeline(**pipeline_kwargs)
        frames = output.frames[0]
        # Clear CUDA memory and synchronize after generation for resource management.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        return frames
