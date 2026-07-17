"""Resident autoregressive inference for HY-WorldPlay.

The released pipeline already denoises four latent frames at a time.  This
adapter keeps that native block size and retains the expensive state that the
offline entry point normally recreates: prompt/text KV, the random stream,
generated latent history, camera/action history, geometry-selected vision KV,
and the causal VAE decoder feature maps.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import distributed as dist

from worldfoundry.core.realtime import RealtimeSpec

from .commons import auto_offload_model
from .commons.infer_state import get_infer_state
from .commons.pose_utils import pose_to_input
from .generate_custom_trajectory import generate_camera_trajectory_local
from .pipelines.pipeline_utils import retrieve_timesteps
from .utils.retrieval_context import generate_points_in_sphere

_KEY_TO_CONTROL = {
    "w": "forward",
    "s": "backward",
    "a": "left",
    "d": "right",
    "i": "camera_up",
    "k": "camera_down",
    "j": "camera_l",
    "l": "camera_r",
}
_CONTROL_ALIASES = {
    "back": "backward",
    "camera_left": "camera_l",
    "camera_right": "camera_r",
}


class HunyuanWorldPlayRealtimeSession:
    """One quality-preserving resident HY-WorldPlay rollout."""

    fps = 24
    latent_frames_per_chunk = 4
    first_output_frames = 13
    steady_output_frames = 16

    def __init__(self, runtime: Any, operator: Any) -> None:
        self.runtime = runtime
        self.core = runtime.model
        self.operator = operator
        self.device = torch.device(self.core.execution_device)
        self.dtype = self.core.target_dtype
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = (
            dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        )
        self.configured = False
        self.configure_metrics: dict[str, float] = {}
        self._generator: torch.Generator | None = None
        self._timesteps: torch.Tensor | None = None
        self._latents: torch.Tensor | None = None
        self._cond_latents: torch.Tensor | None = None
        self._image_condition: torch.Tensor | None = None
        self._motions: list[dict[str, float]] = []
        self._height = 0
        self._width = 0
        self._latent_height = 0
        self._latent_width = 0
        self._num_inference_steps = 0

    def realtime_spec(self) -> RealtimeSpec:
        return RealtimeSpec(
            fps=self.fps,
            first_chunk_frames=self.first_output_frames,
            steady_chunk_frames=self.steady_output_frames,
            controls=(
                "forward",
                "backward",
                "left",
                "right",
                "camera_up",
                "camera_down",
                "camera_l",
                "camera_r",
            ),
        )

    def next_output_frames(self) -> int:
        return self.first_output_frames if self.generated_latent_frames == 0 else self.steady_output_frames

    @property
    def generated_latent_frames(self) -> int:
        return 0 if self._latents is None else int(self._latents.shape[2])

    @staticmethod
    def _normalize_controls(controls: Sequence[str]) -> list[str]:
        result = []
        for control in controls:
            value = str(control).strip().lower()
            value = _CONTROL_ALIASES.get(value, value)
            if value:
                result.append(value)
        return result

    def _controls_for_latents(
        self,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None,
        *,
        count: int,
    ) -> list[list[str]]:
        fallback = self._normalize_controls(interactions)
        segments: list[tuple[float, list[str]]] = []
        for segment in control_segments or ():
            duration = max(float(segment.get("duration", 0.0) or 0.0), 0.0)
            controls = [
                _KEY_TO_CONTROL[key]
                for key in (str(item).lower() for item in segment.get("keys", ()))
                if key in _KEY_TO_CONTROL
            ]
            if duration > 0.0:
                segments.append((duration, controls))
        if not segments:
            return [fallback] * count
        total = sum(duration for duration, _ in segments)
        if total <= 0.0:
            return [fallback] * count
        result: list[list[str]] = []
        segment_index = 0
        segment_end = segments[0][0]
        for index in range(count):
            sample_time = (index + 0.5) * total / count
            while segment_index + 1 < len(segments) and sample_time > segment_end:
                segment_index += 1
                segment_end += segments[segment_index][0]
            result.append(segments[segment_index][1])
        return result

    def _motion(self, controls: Sequence[str]) -> dict[str, float]:
        controls = self._normalize_controls(controls)
        forward = float("forward" in controls) - float("backward" in controls)
        right = float("right" in controls) - float("left" in controls)
        yaw = float("camera_r" in controls) - float("camera_l" in controls)
        pitch = float("camera_up" in controls) - float("camera_down" in controls)
        motion: dict[str, float] = {}
        if forward:
            motion["forward"] = forward * float(self.operator.forward_speed)
        if right:
            motion["right"] = right * float(self.operator.forward_speed)
        if yaw:
            motion["yaw"] = yaw * np.deg2rad(float(self.operator.yaw_speed_deg))
        if pitch:
            motion["pitch"] = pitch * np.deg2rad(float(self.operator.pitch_speed_deg))
        return motion

    def _camera_conditions(
        self,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The first chunk includes the identity pose, so it needs only three new
        # motions. Every continuation contributes four new latent poses.
        new_motion_count = (
            self.latent_frames_per_chunk - 1
            if self.generated_latent_frames == 0
            else self.latent_frames_per_chunk
        )
        controls = self._controls_for_latents(
            interactions,
            control_segments,
            count=new_motion_count,
        )
        self._motions.extend(self._motion(items) for items in controls)
        poses = generate_camera_trajectory_local(self._motions)
        intrinsic = [
            [969.6969696969696, 0.0, 960.0],
            [0.0, 969.6969696969696, 540.0],
            [0.0, 0.0, 1.0],
        ]
        pose_data = {
            str(index): {"extrinsic": pose.tolist(), "K": intrinsic}
            for index, pose in enumerate(poses)
        }
        viewmats, intrinsics, action = pose_to_input(pose_data, len(poses))
        return viewmats.unsqueeze(0), intrinsics.unsqueeze(0), action.unsqueeze(0)

    def _condition_block(self) -> torch.Tensor:
        assert self._image_condition is not None
        channels = int(self.core.transformer.config.in_channels)
        condition = torch.zeros(
            (1, channels, self.latent_frames_per_chunk, self._latent_height, self._latent_width),
            device=self.device,
            # Match the released offline path: the fp16 image latent is
            # concatenated with an fp32 task mask, producing fp32 conditioning.
            dtype=torch.float32,
        )
        mask = torch.zeros(
            (1, 1, self.latent_frames_per_chunk, self._latent_height, self._latent_width),
            device=self.device,
            dtype=torch.float32,
        )
        if self.generated_latent_frames == 0:
            condition[:, :, :1] = self._image_condition.to(device=self.device, dtype=self.dtype)
            mask[:, :, :1] = 1
        return torch.cat((condition, mask), dim=1)

    @torch.inference_mode()
    def configure(
        self,
        image: Image.Image,
        *,
        prompt: str,
        seed: int = 1,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        flow_shift: float | None = None,
        guidance_scale: float | None = None,
        few_step: bool = True,
        user_height: int | None = None,
        user_width: int | None = None,
    ) -> dict[str, Any] | None:
        """Encode invariant conditions and initialize resident model caches."""

        if not isinstance(image, Image.Image):
            raise TypeError("HY-WorldPlay realtime configuration requires a PIL image.")
        infer_state = get_infer_state()
        if infer_state is not None and infer_state.use_vae_parallel:
            raise RuntimeError(
                "HY-WorldPlay realtime does not support tile-parallel VAE decode: "
                "persistent causal feat_map state is tile-local. Disable use_vae_parallel."
            )
        self.reset()
        started = time.perf_counter()
        core = self.core
        image = image.convert("RGB")
        target_resolution = core.ideal_resolution
        height, width = core.get_closest_resolution_given_reference_image(
            image, target_resolution
        )
        self._height = int(user_height or height)
        self._width = int(user_width or width)
        _, self._latent_height, self._latent_width = core.get_latent_size(
            self.first_output_frames,
            self._height,
            self._width,
        )
        self._num_inference_steps = int(num_inference_steps)
        core.scheduler = core._create_scheduler(
            core.config.flow_shift if flow_shift is None else flow_shift
        )
        resolved_guidance = core.config.guidance_scale if guidance_scale is None else guidance_scale
        core._guidance_scale = 1.0 if few_step else resolved_guidance
        core._guidance_rescale = 0.0
        core._clip_skip = None
        self._generator = torch.Generator(device=self.device).manual_seed(int(seed))

        prompt_started = time.perf_counter()
        with auto_offload_model(
            core.text_encoder, core.execution_device, enabled=core.enable_offloading
        ):
            prompt_embeds, negative_embeds, prompt_mask, negative_mask = core.encode_prompt(
                prompt,
                self.device,
                1,
                core.do_classifier_free_guidance,
                negative_prompt,
                clip_skip=None,
                data_type="video",
            )
        extra_kwargs = {}
        if core.config.glyph_byT5_v2:
            with auto_offload_model(
                core.byt5_model, core.execution_device, enabled=core.enable_offloading
            ):
                extra_kwargs = core._prepare_byt5_embeddings(prompt, self.device)
        if core.do_classifier_free_guidance:
            prompt_embeds = torch.cat((negative_embeds, prompt_embeds))
            if prompt_mask is not None:
                prompt_mask = torch.cat((negative_mask, prompt_mask))
        prompt_ms = (time.perf_counter() - prompt_started) * 1000.0

        latent_template = torch.zeros(
            (
                1,
                int(core.transformer.config.in_channels),
                self.latent_frames_per_chunk,
                self._latent_height,
                self._latent_width,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        n_tokens = self.latent_frames_per_chunk * self._latent_height * self._latent_width
        timestep_kwargs = core.prepare_extra_func_kwargs(
            core.scheduler.set_timesteps,
            {"n_tokens": n_tokens},
        )
        self._timesteps, self._num_inference_steps = retrieve_timesteps(
            core.scheduler,
            self._num_inference_steps,
            self.device,
            **timestep_kwargs,
        )
        core.num_warmup_steps = (
            len(self._timesteps) - self._num_inference_steps * core.scheduler.order
        )
        core._num_timesteps = len(self._timesteps)
        core.num_inference_steps = self._num_inference_steps
        core.chunk_latent_frames = self.latent_frames_per_chunk
        core.points_local = generate_points_in_sphere(50000, 8.0).to(self.device)
        if self.world_size > 1:
            # Geometry retrieval must choose identical context indices on every
            # sequence-parallel rank.
            dist.broadcast(core.points_local, src=0)

        visual_started = time.perf_counter()
        with auto_offload_model(core.vae, core.execution_device, enabled=core.enable_offloading):
            self._image_condition = core.get_image_condition_latents(
                "i2v", image, self._height, self._width
            )
        with auto_offload_model(
            core.vision_encoder, core.execution_device, enabled=core.enable_offloading
        ):
            vision_states = core._prepare_vision_states(
                np.asarray(image), target_resolution, latent_template, self.device
            )
        visual_ms = (time.perf_counter() - visual_started) * 1000.0

        cache_started = time.perf_counter()
        core.prepare_ar_text_cache(
            latents=latent_template,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            vision_states=vision_states,
            task_type="i2v",
            extra_kwargs=extra_kwargs,
            device=self.device,
        )
        cache_ms = (time.perf_counter() - cache_started) * 1000.0
        self._latents = latent_template[:, :, :0].clone()
        self._cond_latents = torch.zeros(
            (1, latent_template.shape[1] + 1, 0, self._latent_height, self._latent_width),
            device=self.device,
            dtype=torch.float32,
        )
        if self.rank == 0:
            core.vae.begin_stream_decode()
        self.configured = True
        self.configure_metrics = {
            "prompt_ms": prompt_ms,
            "visual_condition_ms": visual_ms,
            "text_cache_ms": cache_ms,
            "total_ms": (time.perf_counter() - started) * 1000.0,
            "world_size": float(self.world_size),
        }
        if self.rank != 0:
            return None
        return {
            "status": "configured",
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": dict(self.configure_metrics),
        }

    @torch.inference_mode()
    def generate(
        self,
        *,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Advance exactly one four-latent block on the persistent rollout."""

        if not self.configured or self._generator is None or self._timesteps is None:
            raise RuntimeError("HY-WorldPlay realtime session is not configured.")
        assert self._latents is not None and self._cond_latents is not None
        started = time.perf_counter()
        condition_started = started
        start_idx = self.generated_latent_frames
        viewmats, intrinsics, action = self._camera_conditions(
            interactions,
            control_segments,
        )
        condition_block = self._condition_block()
        noise = self.core.prepare_latents(
            1,
            int(self.core.transformer.config.in_channels),
            self._latent_height,
            self._latent_width,
            self.latent_frames_per_chunk,
            self.dtype,
            self.device,
            self._generator,
        )
        self._latents = torch.cat((self._latents, noise), dim=2)
        self._cond_latents = torch.cat((self._cond_latents, condition_block), dim=2)
        selected: list[int] = []
        if start_idx > 0:
            selected = self.core.select_ar_context_indices(
                viewmats=viewmats,
                current_frame_idx=start_idx,
                chunk_latent_frames=self.latent_frames_per_chunk,
                device=self.device,
            )
            self.core.rebuild_ar_vision_cache(
                latents=self._latents,
                cond_latents=self._cond_latents,
                viewmats=viewmats,
                Ks=intrinsics,
                action=action,
                selected_frame_indices=selected,
                timesteps=self._timesteps,
                task_type="i2v",
                device=self.device,
            )
        condition_ms = (time.perf_counter() - condition_started) * 1000.0

        # ``retrieve_timesteps`` configured the first block with its n_tokens
        # scheduler argument. Offline AR resets only continuation blocks.
        if start_idx > 0:
            self.core.scheduler.set_timesteps(self._num_inference_steps, device=self.device)
        model_started = time.perf_counter()
        block = self.core.denoise_ar_chunk(
            latents=self._latents,
            cond_latents=self._cond_latents,
            timesteps=self._timesteps,
            task_type="i2v",
            viewmats=viewmats,
            Ks=intrinsics,
            action=action,
            start_idx=start_idx,
            chunk_latent_frames=self.latent_frames_per_chunk,
            selected_frame_indices=selected,
            device=self.device,
            show_progress=False,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        model_ms = (time.perf_counter() - model_started) * 1000.0

        if self.rank != 0:
            return None
        decode_started = time.perf_counter()
        if getattr(self.core.vae.config, "shift_factor", None):
            decode_latents = (
                block / self.core.vae.config.scaling_factor
                + self.core.vae.config.shift_factor
            )
        else:
            decode_latents = block / self.core.vae.config.scaling_factor
        with (
            torch.autocast(
                device_type="cuda",
                dtype=self.core.vae_dtype,
                enabled=self.core.vae_autocast_enabled,
            ),
            auto_offload_model(
                self.core.vae,
                self.core.execution_device,
                enabled=self.core.enable_offloading,
            ),
        ):
            video = self.core.vae.decode_stream(decode_latents, return_dict=False)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        frames = (
            (video / 2 + 0.5)
            .clamp_(0, 1)
            .mul_(255)
            .to(device="cpu", dtype=torch.uint8)
            .permute(0, 2, 3, 4, 1)
            .numpy()[0]
        )
        return {
            "video": np.ascontiguousarray(frames),
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": {
                "condition_ms": condition_ms,
                "model_ms": model_ms,
                "decode_ms": decode_ms,
                "total_ms": (time.perf_counter() - started) * 1000.0,
                "context_latent_frames": len(selected),
                "generated_latent_frames": self.generated_latent_frames,
                "world_size": self.world_size,
            },
        }

    def reset(self) -> None:
        """Drop rollout state while keeping all weights resident."""

        vae = getattr(self.core, "vae", None)
        if self.rank == 0 and vae is not None and hasattr(vae, "reset_stream_decode"):
            vae.reset_stream_decode()
        self.configured = False
        self.configure_metrics = {}
        self._generator = None
        self._timesteps = None
        self._latents = None
        self._cond_latents = None
        self._image_condition = None
        self._motions = []
        self._height = self._width = 0
        self._latent_height = self._latent_width = 0
        self._num_inference_steps = 0
        for name in ("_kv_cache", "_kv_cache_neg"):
            if hasattr(self.core, name):
                setattr(self.core, name, None)


__all__ = ["HunyuanWorldPlayRealtimeSession"]
