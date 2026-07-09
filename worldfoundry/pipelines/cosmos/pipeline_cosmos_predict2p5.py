"""Cosmos Predict2P5 visual generation pipeline module."""

from ..pipeline_utils import PipelineABC
import math
import numpy as np
import torch
from PIL import Image
from typing import Any, Dict, List, Optional, Union

from ...operators.cosmos_predict2p5_operator import CosmosPredict2p5Operator
from ...synthesis.visual_generation.cosmos.cosmos_predict2p5_synthesis import CosmosPredict2p5Synthesis
from ...synthesis.visual_generation.memory.stream import VisualFrameMemory


COSMOS_PREDICT2P5_DEFAULT_FPS = 16
COSMOS_PREDICT2P5_DEFAULT_NUM_FRAMES = 93
COSMOS_PREDICT2P5_DEFAULT_NUM_INFERENCE_STEPS = 35
_STUDIO_LEGACY_VALIDATION_DEFAULTS = {
    "fps": 28,
    "num_frames": 17,
    "num_inference_steps": 4,
}
_SUPPORTED_WORLD_MODES = {
    "img2world": "img2world",
    "image2world": "img2world",
    "image-to-world": "img2world",
    "text2world": "img2world",
    "text-to-world": "img2world",
}


def _normalize_generation_defaults(
    *,
    fps: int,
    num_frames: int,
    num_inference_steps: int,
) -> tuple[int, int, int]:
    """Promote legacy compact Studio defaults to usable Cosmos defaults."""
    if (
        fps == _STUDIO_LEGACY_VALIDATION_DEFAULTS["fps"]
        and num_frames == _STUDIO_LEGACY_VALIDATION_DEFAULTS["num_frames"]
        and num_inference_steps == _STUDIO_LEGACY_VALIDATION_DEFAULTS["num_inference_steps"]
    ):
        return (
            COSMOS_PREDICT2P5_DEFAULT_FPS,
            COSMOS_PREDICT2P5_DEFAULT_NUM_FRAMES,
            COSMOS_PREDICT2P5_DEFAULT_NUM_INFERENCE_STEPS,
        )
    return fps, num_frames, num_inference_steps


def _video_to_thwc_uint8(video: Any) -> np.ndarray:
    """Video to thwc uint8 helper function."""
    if isinstance(video, torch.Tensor):
        arr = video.detach().cpu().float().numpy()
    else:
        arr = np.asarray(video)

    if arr.ndim == 5:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for Cosmos video output, got shape {arr.shape}")
        arr = arr[0]

    if arr.ndim == 4:
        if arr.shape[-1] in {1, 3, 4}:
            pass
        elif arr.shape[0] in {1, 3, 4}:
            arr = np.transpose(arr, (1, 2, 3, 0))
        elif arr.shape[1] in {1, 3, 4}:
            arr = np.transpose(arr, (0, 2, 3, 1))

    if arr.ndim != 4 or arr.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"Expected Cosmos video output in THWC-compatible shape, got {arr.shape}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        if arr.size:
            min_value = float(arr.min())
            max_value = float(arr.max())
        else:
            min_value = 0.0
            max_value = 0.0
        if min_value < 0.0 and max_value <= 1.0:
            arr = (arr + 1.0) * 127.5
        elif max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


class CosmosPredict2p5Pipeline(PipelineABC):

    """Pipeline implementation for CosmosPredict2p5 visual generation."""
    def __init__(
        self,
        operator: Optional[CosmosPredict2p5Operator] = None,
        synthesis_model: Optional[CosmosPredict2p5Synthesis] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the pipeline and configure runtime components."""
        self.synthesis_model = synthesis_model
        self.operator = operator
        self.memory_module = memory_module
        self.device = device
        self.weight_dtype = weight_dtype
        self.current_image = None

    @classmethod
    def from_pretrained(
        cls, 
        model_path: Optional[str | Dict[str, Any]] = None,
        required_components: Optional[Dict] = None,
        token: Optional[str] = None,
        mode: str = 'img2world',
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype: Optional[torch.dtype] = torch.bfloat16,
        **kwargs,
    ) -> "CosmosPredict2p5Pipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        component_options: Dict[str, Any] = dict(required_components or {})
        if isinstance(model_path, dict):
            component_options.update(model_path)
            model_path = (
                component_options.pop("model_path", None)
                or component_options.pop("pretrained_model_path", None)
                or component_options.pop("repo_id", None)
            )
        component_options.update(kwargs)
        component_options = cls._strip_framework_loading_options(component_options)
        mode = component_options.pop("mode", mode)
        token = component_options.pop("token", token)
        normalized_mode = str(mode).strip().lower().replace("_", "-")
        if normalized_mode == "video2world" or normalized_mode == "video-to-world":
            raise NotImplementedError(
                "Cosmos Predict2.5 video2world loading is not wired in this in-tree "
                "WorldFoundry wrapper yet."
            )
        mode = _SUPPORTED_WORLD_MODES.get(normalized_mode, mode)
        if model_path is None:
            model_path = "nvidia/Cosmos-Predict2.5"
        if required_components is None:
            required_components = {
                "text_encoder_model_path": "nvidia/Cosmos-Reason1-7B",
                "vae_model_path": "Wan-AI/Wan2.1-T2V-1.3B",
            }
        required_components = {**required_components, **component_options}

        synthesis_model = CosmosPredict2p5Synthesis.from_pretrained(
            mode=mode,
            transformer_model_path=model_path,
            text_encoder_model_path=required_components["text_encoder_model_path"],
            vae_model_path=required_components["vae_model_path"],
            token=token,
            device=torch.device(device),
            weight_dtype=weight_dtype,
        )
        operator = CosmosPredict2p5Operator()
        memory_module = VisualFrameMemory(model_id="cosmos-predict2.5")

        pipeline = cls(
            operator=operator,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device,
            weight_dtype=weight_dtype
        )
        return pipeline

    def set_negative_prompt(self, neg_prompt: Optional[str] = None):
        """Set negative prompt for CosmosPredict2p5Pipeline."""
        if neg_prompt is not None:
            self.negative_prompt = neg_prompt
        else:
            self.negative_prompt = (
                'The video captures a series of frames showing ugly scenes, static with no motion, '
                'motion blur, over-saturation, shaky footage, low resolution, grainy texture,'
                ' pixelated images, poorly lit areas, underexposed and overexposed scenes, '
                'poor color balance, washed out colors, choppy sequences, jerky movements, '
                'low frame rate, artifacting, color banding, unnatural transitions, '
                'outdated special effects, fake elements, unconvincing visuals, '
                'poorly edited content, jump cuts, visual noise, and flickering. '
                'Overall, the video is of poor quality.'
            )

    def process(
        self,
        prompt: str,
        images: Optional[Image.Image] = None,
        image_path: Optional[str] = None,
        input_path: Optional[str] = None,
        height: int = 704,
        width: int = 1280,
    ) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        input_for_perception = images if images is not None else image_path or input_path
        perception = self.operator.process_perception(
            input_path=input_for_perception,
            height=height,
            width=width,
        )
        
        image = perception["input_image"]
        height = perception["height"]
        width = perception["width"]

        self.operator.get_interaction(prompt)
        interaction = self.operator.process_interaction()
        
        prompt = interaction["input_prompt"]

        return {
            "prompt": prompt,
            "image": image,
            "height": height,
            "width": width,
        }

    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        images: Any = None,
        image_path: Optional[str] = None,
        input_path: Optional[str] = None,
        video: Any = None,
        video_path: Optional[str] = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = COSMOS_PREDICT2P5_DEFAULT_NUM_INFERENCE_STEPS,
        fps: int = COSMOS_PREDICT2P5_DEFAULT_FPS,
        num_frames: int = COSMOS_PREDICT2P5_DEFAULT_NUM_FRAMES,
        height: int = 704,
        width: int = 1280,
        action_latents: Optional[torch.Tensor] = None,
        control_video: Optional[Union[List[Image.Image], Dict[str, List[Image.Image]]]] = None,
        control_scale: Optional[Union[float, Dict[str, float]]] = 1.0,
        cond_timestep: float = 0,
        timestep_scale: float = 0.001,
        seed: int = -1,
        use_kerras_sigma: bool = True,
        pad_mode: str = 'repeat',
        output_type: Optional[str] = 'pt',
        _normalize_output: bool = True,
    ) -> Any:
        """Execute the complete pipeline generation flow."""
        if prompt is None:
            raise ValueError("prompt must be provided either in initialization or call().")
        if video is not None or video_path is not None:
            raise NotImplementedError(
                "Cosmos Predict2.5 video2world conditioning is not wired in this in-tree "
                "WorldFoundry wrapper yet. Use text2world/image2world here, or route the "
                "official video2world runtime with an input video."
            )
        if negative_prompt is None:
            negative_prompt = getattr(self, "negative_prompt", None)
        fps, num_frames, num_inference_steps = _normalize_generation_defaults(
            fps=fps,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
        )

        processed_input = self.process(
            prompt=prompt,
            images=images,
            image_path=image_path,
            input_path=input_path,
            height=height,
            width=width,
        )

        video = self.synthesis_model.predict(
            **processed_input,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            fps=fps,
            num_frames=num_frames,
            action_latents=action_latents,
            control_video=control_video,
            control_scale=control_scale,
            cond_timestep=cond_timestep,
            timestep_scale=timestep_scale,
            seed=seed,
            use_kerras_sigma=use_kerras_sigma,
            pad_mode=pad_mode,
            output_type=output_type,
        )

        if not _normalize_output:
            return video
        return _video_to_thwc_uint8(video)

    def stream(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        images: Any = None,
        image_path: Optional[str] = None,
        input_path: Optional[str] = None,
        video: Any = None,
        video_path: Optional[str] = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = COSMOS_PREDICT2P5_DEFAULT_NUM_INFERENCE_STEPS,
        fps: int = COSMOS_PREDICT2P5_DEFAULT_FPS,
        num_frames: int = COSMOS_PREDICT2P5_DEFAULT_NUM_FRAMES,
        height: int = 704,
        width: int = 1280,
        action_latents: Optional[torch.Tensor] = None,
        control_video: Optional[Union[List[Image.Image], Dict[str, List[Image.Image]]]] = None,
        control_scale: Optional[Union[float, Dict[str, float]]] = 1.0,
        cond_timestep: float = 0,
        timestep_scale: float = 0.001,
        seed: int = -1,
        use_kerras_sigma: bool = True,
        pad_mode: str = 'repeat',
        output_type: Optional[str] = 'pt',
    ) -> Any:
        """Stream visual generation outputs chunk by chunk."""
        video = self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            images=images,
            image_path=image_path,
            input_path=input_path,
            video=video,
            video_path=video_path,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            fps=fps,
            num_frames=num_frames,
            height=height,
            width=width,
            action_latents=action_latents,
            control_video=control_video,
            control_scale=control_scale,
            cond_timestep=cond_timestep,
            timestep_scale=timestep_scale,
            seed=seed,
            use_kerras_sigma=use_kerras_sigma,
            pad_mode=pad_mode,
            output_type=output_type,
            _normalize_output=False,
        )

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"[CosmosPredict2p5Pipeline.stream] Expected torch.Tensor from predict, got {type(video)}"
            )

        video = video.squeeze(0)
        self.memory_module.record(video)
        print(
            f"[CosmosPredict2p5Pipeline.stream] Recorded segment. "
            f"Total frames in memory: {len(getattr(self.memory_module, 'all_frames', []))}"
        )

        return video
