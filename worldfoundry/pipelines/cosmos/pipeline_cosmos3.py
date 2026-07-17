"""Cosmos3 visual generation pipeline module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..pipeline_utils import PipelineABC
from ...base_models.diffusion_model.video.cosmos3.artifacts import (
    resolve_cosmos3_model_source,
    resolve_cosmos3_variant_id,
    strip_cosmos3_loader_metadata,
)
from ...synthesis.visual_generation.cosmos.cosmos3_synthesis import Cosmos3Synthesis


def _synthesis_load_options(options: Mapping[str, Any], *, variant_id: str) -> dict[str, Any]:
    """Forward runtime selectors while keeping framework metadata out of Diffusers."""

    runtime_options = strip_cosmos3_loader_metadata(options)
    runtime_options["variant_id"] = variant_id
    revision = str(options.get("revision") or "").strip()
    if revision:
        runtime_options["revision"] = revision
    return runtime_options


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
    elif arr.shape[1] in {1, 3, 4}:
        arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.shape[0] in {1, 3, 4}:
        arr = np.transpose(arr, (1, 2, 3, 0))
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


def _single_image_to_hwc_uint8(image: Any) -> np.ndarray:
    """Normalize a one-frame PIL/NumPy/PyTorch result for image artifact writing."""

    if isinstance(image, (list, tuple)):
        if len(image) != 1:
            raise ValueError(f"Expected one Cosmos3 image frame, got {len(image)} frames.")
        image = image[0]
    if isinstance(image, torch.Tensor) and image.ndim == 3:
        image = image.unsqueeze(0)
    elif not isinstance(image, torch.Tensor):
        array = np.asarray(image)
        if array.ndim == 3:
            image = array[None, ...]
    frames = _video_to_thwc_uint8(image)
    if frames.shape[0] != 1:
        raise ValueError(f"Expected one Cosmos3 image frame, got shape {frames.shape}.")
    return frames[0]


def _write_image_artifact(image: Any, output_path: str | Path) -> str:
    """Write a single image and return the actual path, normalizing video suffixes."""

    from PIL import Image

    requested = Path(output_path)
    if requested.suffix.lower() not in {".bmp", ".jpeg", ".jpg", ".png", ".webp"}:
        requested = requested.with_suffix(".png") if requested.suffix else Path(f"{requested}.png")
    requested.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_single_image_to_hwc_uint8(image)).save(requested)
    return str(requested)


def _load_image_input(image: Any) -> Any:
    if isinstance(image, (str, Path)):
        from PIL import Image

        return Image.open(image).convert("RGB")
    return image


def _load_video_input(video: Any) -> Any:
    if isinstance(video, (str, Path)):
        from worldfoundry.core.io import load_video_frames

        return load_video_frames(video)
    return video


def _build_cosmos_action(
    action: Any,
    *,
    action_mode: str | None,
    action_chunk_size: int | None,
    domain_name: str | None,
    resolution_tier: int | None,
    raw_actions: Any,
    view_point: str | None,
    image: Any,
    video: Any,
) -> Any:
    """Build the official action condition from explicit structured inputs."""

    if action is None and action_mode is None:
        return None
    from ...base_models.diffusion_model.video.cosmos3.diffusers_cosmos3 import CosmosActionCondition

    if isinstance(action, CosmosActionCondition):
        overrides = (action_mode, action_chunk_size, domain_name, resolution_tier, raw_actions, view_point, image, video)
        if any(value is not None for value in overrides):
            raise ValueError(
                "A prebuilt CosmosActionCondition cannot be combined with separate action or conditioning fields."
            )
        return action
    if action is not None and not isinstance(action, Mapping):
        raise TypeError("Cosmos3 `action` must be a CosmosActionCondition or a structured mapping.")

    payload = dict(action or {})
    explicit_fields = {
        "mode": action_mode,
        "chunk_size": action_chunk_size,
        "domain_name": domain_name,
        "resolution_tier": resolution_tier,
        "raw_actions": raw_actions,
        "view_point": view_point,
    }
    for key, value in explicit_fields.items():
        if value is not None:
            payload[key] = value

    if payload.get("image") is not None and image is not None:
        raise ValueError("Cosmos3 action image was provided both inside `action` and as a top-level input.")
    if payload.get("video") is not None and video is not None:
        raise ValueError("Cosmos3 action video was provided both inside `action` and as a top-level input.")
    payload.setdefault("image", image)
    payload.setdefault("video", video)
    payload["image"] = _load_image_input(payload.get("image"))
    payload["video"] = _load_video_input(payload.get("video"))

    if payload.get("raw_actions") is not None and not isinstance(payload["raw_actions"], torch.Tensor):
        payload["raw_actions"] = torch.as_tensor(payload["raw_actions"], dtype=torch.float32)
    payload.setdefault("resolution_tier", 480)
    payload.setdefault("view_point", "ego_view")
    missing = [key for key in ("mode", "chunk_size", "domain_name") if payload.get(key) is None]
    if missing:
        raise ValueError("Cosmos3 structured action is missing required fields: " + ", ".join(missing))
    return CosmosActionCondition(**payload)


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
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        source_request: str | Mapping[str, Any] | None = model_path if isinstance(model_path, str) else options
        resolved_source = resolve_cosmos3_model_source(source_request, model_id=model_id)
        resolved_model_id = resolve_cosmos3_variant_id(
            options,
            model_id=model_id,
            model_source=resolved_source,
        )

        synthesis_model = Cosmos3Synthesis.from_pretrained(
            model_path=resolved_source,
            device=device,
            **_synthesis_load_options(options, variant_id=resolved_model_id),
        )
        return cls(
            synthesis_model=synthesis_model,
            device=device,
            model_id=resolved_model_id,
        )

    @classmethod
    def plan(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Plan for Cosmos3Pipeline."""
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        source_request: str | Mapping[str, Any] | None = model_path if isinstance(model_path, str) else options
        resolved_source = resolve_cosmos3_model_source(source_request)
        resolved_model_id = resolve_cosmos3_variant_id(options, model_source=resolved_source)
        return Cosmos3Synthesis.plan(
            model_path=resolved_source,
            **_synthesis_load_options(options, variant_id=resolved_model_id),
        )

    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        image: Any = None,
        images: Any = None,
        image_path: str | None = None,
        input_path: str | None = None,
        video: Any = None,
        videos: Any = None,
        video_path: str | None = None,
        interactions: Any = None,
        action: Any = None,
        action_mode: str | None = None,
        action_chunk_size: int | None = None,
        domain_name: str | None = None,
        resolution_tier: int | None = None,
        raw_actions: Any = None,
        view_point: str | None = None,
        enable_sound: bool = False,
        ref_image_path: str | None = None,
        output_path: str | None = None,
        output_dir: str | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        flow_shift: float | None = None,
        use_karras_sigmas: bool | None = None,
        seed: int | None = None,
        output_type: str = "video",
        return_dict: bool = False,
        return_omni_output: bool | None = None,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the complete pipeline generation flow."""
        del output_dir, operator_kwargs
        if self.synthesis_model is None:
            raise RuntimeError("Cosmos3Pipeline is not loaded. Use from_pretrained() first.")
        has_interactions = interactions is not None and not (
            isinstance(interactions, (list, tuple, dict, set)) and len(interactions) == 0
        )
        if has_interactions:
            raise ValueError(
                "Cosmos3Pipeline exposes visual T2I/T2V/I2V/V2V inference; generic interaction/action controls "
                "are not accepted by this binding."
            )
        input_image = image if image is not None else images
        generic_video_path = None
        if input_path is not None and Path(input_path).suffix.lower() in {".mp4", ".mov", ".avi", ".webm", ".mkv"}:
            generic_video_path = input_path
        if input_image is None:
            input_image = image_path or ref_image_path or (None if generic_video_path else input_path)
        if isinstance(input_image, (list, tuple)) and len(input_image) == 1:
            input_image = input_image[0]
        input_image = _load_image_input(input_image)
        input_video = video if video is not None else videos
        if input_video is None:
            input_video = video_path or generic_video_path
        input_video = _load_video_input(input_video)
        action_condition = _build_cosmos_action(
            action,
            action_mode=action_mode,
            action_chunk_size=action_chunk_size,
            domain_name=domain_name,
            resolution_tier=resolution_tier,
            raw_actions=raw_actions,
            view_point=view_point,
            image=input_image,
            video=input_video,
        )
        runtime_image = None if action_condition is not None else input_image
        runtime_video = None if action_condition is not None else input_video
        wants_omni_output = bool(return_omni_output) or enable_sound or action_condition is not None
        generation_kwargs = dict(kwargs)
        generation_kwargs.pop("task", None)
        generation_kwargs.pop("task_type", None)
        for key, value in {
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "fps": fps,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "use_karras_sigmas": use_karras_sigmas,
            "seed": seed,
        }.items():
            if value is not None:
                generation_kwargs[key] = value
        result = self.synthesis_model.predict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=runtime_image,
            video=runtime_video,
            enable_sound=enable_sound,
            action=action_condition,
            return_omni_output=wants_omni_output,
            output_type=output_type,
            **generation_kwargs,
        )
        if isinstance(result, Mapping):
            video_result = result.get("video")
            sound_result = result.get("sound")
            action_result = result.get("action")
            audio_sample_rate = result.get("audio_sample_rate")
        elif wants_omni_output and hasattr(result, "video"):
            video_result = result.video
            sound_result = getattr(result, "sound", None)
            action_result = getattr(result, "action", None)
            audio_sample_rate = getattr(result, "audio_sample_rate", None)
        else:
            video_result = result
            sound_result = None
            action_result = None
            audio_sample_rate = None
        artifact_path = None
        artifact_kind = "generated_world"
        audio_path = None
        action_path = None
        if output_type == "video":
            video_result = _video_to_thwc_uint8(video_result)
            if output_path is not None:
                from worldfoundry.core.io import write_video

                write_video(video_result, output_path, fps=int(round(float(fps or 24.0))))
                artifact_path = str(output_path)
        elif output_path is not None:
            artifact_path = _write_image_artifact(video_result, output_path)
            artifact_kind = "generated_image"
        elif num_frames == 1 or (
            output_type == "pil" and isinstance(video_result, (list, tuple)) and len(video_result) == 1
        ):
            artifact_kind = "generated_image"

        if sound_result is not None and output_path is not None:
            if audio_sample_rate is None:
                raise RuntimeError("Cosmos3 returned sound without an audio sample rate.")
            from worldfoundry.core.io import mux_audio_video, write_audio

            artifact_base = Path(artifact_path or output_path)
            audio_path = write_audio(
                sound_result,
                artifact_base.with_name(f"{artifact_base.stem}.audio.wav"),
                sample_rate=int(audio_sample_rate),
            )
            if artifact_path is not None and Path(artifact_path).suffix.lower() in {
                ".avi",
                ".mkv",
                ".mov",
                ".mp4",
                ".webm",
            }:
                mux_audio_video(artifact_path, audio_path)

        if action_result is not None and output_path is not None:
            from worldfoundry.core.io import write_json

            artifact_base = Path(artifact_path or output_path)
            action_payload = {
                "mode": getattr(action_condition, "mode", None),
                "domain_name": getattr(action_condition, "domain_name", None),
                "chunk_size": getattr(action_condition, "chunk_size", None),
                "view_point": getattr(action_condition, "view_point", None),
                "actions": action_result,
            }
            action_path = str(
                write_json(
                    artifact_base.with_name(f"{artifact_base.stem}.actions.json"),
                    action_payload,
                )
            )

        artifact_paths = [path for path in (artifact_path, audio_path, action_path) if path is not None]
        if return_dict:
            return {
                "status": "succeeded",
                "artifact_path": artifact_path,
                "artifact_paths": artifact_paths,
                "artifact_kind": artifact_kind,
                "video": video_result,
                "sound": sound_result,
                "action": action_result,
                "audio_sample_rate": audio_sample_rate,
                "audio_path": audio_path,
                "action_path": action_path,
                "model_id": self.model_id,
            }
        if wants_omni_output:
            return {
                "video": video_result,
                "sound": sound_result,
                "action": action_result,
                "audio_sample_rate": audio_sample_rate,
            }
        return video_result


__all__ = ["Cosmos3Pipeline"]
