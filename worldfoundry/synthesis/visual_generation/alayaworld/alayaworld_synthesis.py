"""WorldFoundry synthesis adapter for AlayaWorld."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.distributed as dist

from worldfoundry.synthesis.visual_generation.runtime_video_synthesis import RuntimeVideoSynthesis

from .runtime import AlayaWorldRuntime


class AlayaWorldSynthesis(RuntimeVideoSynthesis):
    """Expose the in-tree AlayaWorld rollout through the standard I2V interface."""

    MODEL_NAME = "alayaworld"
    GENERATION_TYPE = "i2v"
    RUNTIME_CLS = AlayaWorldRuntime
    PRIMARY_PATH_KEY = "checkpoint_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/alayaworld/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME

    def predict(
        self,
        prompt: str,
        images: Any = None,
        output_path: str | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ):
        # Every CP rank must execute the rollout, but only rank 0 should race to
        # write the common artifact path.  The runtime broadcasts decoded frames,
        # so workers still return the same standard result payload.
        if dist.is_initialized() and dist.get_world_size() > 1 and dist.get_rank() != 0:
            output_path = None
            kwargs.pop("save_path", None)
        return super().predict(
            prompt=prompt,
            images=images,
            output_path=output_path,
            fps=fps,
            return_dict=return_dict,
            **kwargs,
        )

    def _apply_prediction_runtime_overrides(self, overrides: Mapping[str, Any]) -> None:
        changed = any(self.runtime_kwargs.get(key) != value for key, value in overrides.items())
        if changed and self.generator is not None:
            close = getattr(self.generator, "close", None)
            if callable(close):
                close()
        super()._apply_prediction_runtime_overrides(overrides)

    def _prediction_runtime_overrides(
        self,
        kwargs: Mapping[str, Any],
        *,
        fps: int | None,
    ) -> dict[str, Any]:
        aliases = {
            "frames": "num_frames",
            "frame_num": "num_frames",
            "max_frames": "num_frames",
            "video_length": "num_frames",
            "steps": "sampling_steps",
            "num_steps": "sampling_steps",
            "num_inference_steps": "sampling_steps",
            "infer_steps": "sampling_steps",
            "trajectory": "camera_trajectory",
            "camera": "camera_trajectory",
        }
        direct = {
            "num_frames",
            "rounds",
            "height",
            "width",
            "sampling_steps",
            "seed",
            "camera_path",
            "camera_trajectory",
            "camera_translation_step",
            "camera_rotation_step_degrees",
            "intrinsic",
            "action_scale",
            "action_freq_scale",
            "action_history_memory",
            "spatial_enabled",
            "depth_backend",
            "spatial_num_context_frames",
            "spatial_retrieval_views",
            "spatial_downsample",
            "spatial_maximum_coverage",
            "spatial_retrieval_depth_threshold",
            "spatial_constant_depth",
            "spatial_include_sink",
            "spatial_require_full_context",
            "da3_process_res",
            "context_parallel",
            "decode_rank0_only",
            "compile_mode",
            "compile_backend",
            "compile_fullgraph",
            "compile_dynamic",
        }
        overrides: dict[str, Any] = {}
        if fps is not None:
            overrides["fps"] = fps
        for key, value in kwargs.items():
            if value is None:
                continue
            canonical = aliases.get(key, key)
            if canonical in direct:
                overrides[canonical] = value
        return overrides

    def close(self) -> None:
        """Release the lazily-created Alaya runtime and its CUDA resources."""

        generator = self.generator
        self.generator = None
        if generator is not None:
            close = getattr(generator, "close", None)
            if callable(close):
                close()


__all__ = ["AlayaWorldSynthesis"]
