"""Resident, quality-first interactive runtime for SANA-WM.

SANA-WM is a bidirectional long-context model rather than a causal token
streamer. A realtime session therefore advances in native 8k+1 temporal
windows and retains the last refined frame as the next anchor. Heavy
components stay resident and may be placed on separate visible CUDA devices;
this is component partitioning, not tensor-parallel execution.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from worldfoundry.core.io.paths import checkpoint_root_path
from worldfoundry.core.realtime import RealtimeSpec
from worldfoundry.runtime.local_checkpoint_cache import stage_checkpoint_for_realtime

_TOKEN_TO_KEY = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "camera_up": "i",
    "camera_down": "k",
    "camera_l": "j",
    "camera_r": "l",
}

_REQUIRED_CHECKPOINT_PATHS = (
    "config.yaml",
    "dit/sana_wm_1600m_720p.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "refiner/transformer/config.json",
    "refiner/transformer/diffusion_pytorch_model.safetensors",
    "refiner/connectors/config.json",
    "refiner/connectors/diffusion_pytorch_model.safetensors",
    "refiner/text_encoder/config.json",
    "refiner/text_encoder/model.safetensors.index.json",
)

DEFAULT_REALTIME_WINDOW_FRAMES = 81
DEFAULT_SAMPLING_STEPS = 60
DEFAULT_CFG_SCALE = 5.0


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(int(os.getenv(name, str(default)) or default), minimum)
    except ValueError:
        return max(int(default), minimum)


def _runtime_import_paths() -> tuple[Path, ...]:
    source_root = Path(__file__).resolve().parents[4]
    return (
        source_root / "worldfoundry/base_models/diffusion_model/image/sana",
        source_root / "worldfoundry/base_models/diffusion_model/video/wan",
        source_root / "worldfoundry/base_models/three_dimensions/point_clouds/pi3",
        source_root,
    )


def ensure_sana_import_paths() -> None:
    """Expose the in-tree runtime's historical top-level package names."""

    for path in reversed(_runtime_import_paths()):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _resolve_checkpoint(source: str | Path | None) -> Path:
    candidates = []
    if source:
        candidates.append(Path(source).expanduser())
    candidates.extend(
        (
            checkpoint_root_path("SANA-WM_bidirectional"),
            checkpoint_root_path("hfd", "Efficient-Large-Model--SANA-WM_bidirectional"),
        )
    )
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / item).exists() for item in _REQUIRED_CHECKPOINT_PATHS):
            return candidate.resolve()
    searched = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"SANA-WM checkpoint is missing or incomplete. Searched: {searched}")


def _snap_num_frames(value: int) -> int:
    """Use the nearest quality-preserving LTX temporal layout (8k+1)."""

    value = max(int(value), 9)
    lower = value - ((value - 1) % 8)
    upper = lower + 8
    return lower if value - lower < upper - value else upper


def _device_topology() -> dict[str, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("SANA-WM realtime requires CUDA.")
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("No visible CUDA device is available for SANA-WM.")

    defaults = {
        "stage1": 0,
        "refiner": 1 if count >= 2 else 0,
        "refiner_text": 2 if count >= 3 else 0,
        "vae": 3 if count >= 4 else 0,
        "stage1_text": 3 if count >= 4 else 0,
        "intrinsics": 3 if count >= 4 else 0,
    }
    env_names = {
        "stage1": "WORLDFOUNDRY_SANA_WM_STAGE1_DEVICE",
        "refiner": "WORLDFOUNDRY_SANA_WM_REFINER_DEVICE",
        "refiner_text": "WORLDFOUNDRY_SANA_WM_REFINER_TEXT_DEVICE",
        "vae": "WORLDFOUNDRY_SANA_WM_VAE_DEVICE",
        "stage1_text": "WORLDFOUNDRY_SANA_WM_STAGE1_TEXT_DEVICE",
        "intrinsics": "WORLDFOUNDRY_SANA_WM_INTRINSICS_DEVICE",
    }
    topology: dict[str, torch.device] = {}
    for name, default_index in defaults.items():
        raw = os.getenv(env_names[name], str(default_index)).strip().lower()
        if raw.startswith("cuda:"):
            raw = raw.split(":", 1)[1]
        try:
            index = int(raw)
        except ValueError as exc:
            raise ValueError(f"{env_names[name]} must be a CUDA index, got {raw!r}.") from exc
        if not 0 <= index < count:
            raise ValueError(f"{env_names[name]}={index} but only {count} CUDA devices are visible.")
        topology[name] = torch.device(f"cuda:{index}")
    return topology


def _frames_for_segments(
    segments: Sequence[Mapping[str, Any]] | None,
    interactions: Sequence[str],
    *,
    frame_count: int,
    fps: int,
) -> list[frozenset[str]]:
    fallback = frozenset(
        _TOKEN_TO_KEY[token]
        for item in interactions
        if (token := str(item).strip().lower()) in _TOKEN_TO_KEY
    )
    if not segments:
        return [fallback] * frame_count

    rows: list[tuple[float, frozenset[str]]] = []
    for segment in segments:
        duration = max(float(segment.get("duration", 0.0) or 0.0), 0.0)
        keys = frozenset(
            str(item).lower()
            for item in (segment.get("keys") or ())
            if str(item).lower() in "wasdijkl"
        )
        if duration:
            rows.append((duration, keys))
    if not rows:
        return [fallback] * frame_count

    total = sum(duration for duration, _ in rows)
    scale = (frame_count / float(fps)) / total if total > 0.0 else 1.0
    boundaries: list[tuple[float, frozenset[str]]] = []
    elapsed = 0.0
    for duration, keys in rows:
        elapsed += duration * scale
        boundaries.append((elapsed, keys))
    result: list[frozenset[str]] = []
    row_index = 0
    for frame_index in range(frame_count):
        timestamp = (frame_index + 1) / float(fps)
        while row_index + 1 < len(boundaries) and timestamp > boundaries[row_index][0] + 1e-8:
            row_index += 1
        result.append(boundaries[row_index][1])
    return result


def _compress_action_frames(frames: Sequence[frozenset[str]]) -> str:
    if not frames:
        return "none-1"
    segments: list[str] = []
    previous = frames[0]
    length = 1
    for keys in frames[1:]:
        if keys == previous:
            length += 1
            continue
        segments.append(f"{''.join(sorted(previous)) or 'none'}-{length}")
        previous = keys
        length = 1
    segments.append(f"{''.join(sorted(previous)) or 'none'}-{length}")
    return ",".join(segments)


def _validate_output_video(value: Any, *, expected_frames: int) -> np.ndarray:
    video = np.ascontiguousarray(value, dtype=np.uint8)
    if video.ndim != 4 or video.shape[-1] != 3 or len(video) == 0:
        raise RuntimeError(f"SANA-WM returned an invalid video array: {video.shape}")
    if len(video) != expected_frames:
        raise RuntimeError(
            f"SANA-WM returned {len(video)} frames; expected {expected_frames} "
            "for the configured native window."
        )
    return video


class SanaWMRealtimeSession:
    """One persistent high-quality SANA-WM exploration session."""

    fps = 16

    def __init__(self, checkpoint_source: str | Path | None = None) -> None:
        ensure_sana_import_paths()
        import pyrallis

        from worldfoundry.base_models.diffusion_model.image.sana.inference_video_scripts.inference_sana_wm import (
            GenerationParams,
            InferenceConfig,
            RefinerSettings,
            SanaWMPipeline,
        )

        source = _resolve_checkpoint(checkpoint_source)
        self.checkpoint = (
            stage_checkpoint_for_realtime(
                source,
                required_paths=_REQUIRED_CHECKPOINT_PATHS,
            )
            if _env_flag("WORLDFOUNDRY_SANA_WM_STAGE_CHECKPOINT", False)
            else source
        )
        self.devices = _device_topology()
        torch.cuda.set_device(self.devices["stage1"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Five seconds is the interactive default. It is still a native 8k+1
        # bidirectional window; callers can request the full 161-frame window.
        self.num_frames = _snap_num_frames(
            _env_int(
                "WORLDFOUNDRY_SANA_WM_NUM_FRAMES",
                DEFAULT_REALTIME_WINDOW_FRAMES,
                minimum=9,
            )
        )
        self.output_frames = self.num_frames - 1
        self.logger = logging.getLogger("worldfoundry.sana_wm.realtime")
        config: InferenceConfig = pyrallis.parse(
            config_class=InferenceConfig,
            config_path=str(self.checkpoint / "config.yaml"),
            args=[],
        )
        config.vae.vae_pretrained = str(self.checkpoint)
        refiner = RefinerSettings(
            root=self.checkpoint / "refiner",
            gemma_root=self.checkpoint / "refiner/text_encoder",
            sink_size=1,
            seed=_env_int("WORLDFOUNDRY_SANA_WM_REFINER_SEED", 42, minimum=0),
        )
        self.pipeline = SanaWMPipeline(
            config=config,
            model_path=self.checkpoint / "dit/sana_wm_1600m_720p.safetensors",
            device=self.devices["stage1"],
            vae_device=self.devices["vae"],
            text_device=self.devices["stage1_text"],
            refiner_device=self.devices["refiner"],
            refiner_text_device=self.devices["refiner_text"],
            refiner=refiner,
            offload_vae=False,
            offload_refiner=False,
            keep_refiner_text_encoder_resident=True,
            logger=self.logger,
        )
        self._intrinsics_estimator: Any | None = None
        if _env_flag("WORLDFOUNDRY_SANA_WM_USE_PI3X", True):
            from worldfoundry.base_models.diffusion_model.image.sana.inference_video_scripts.inference_sana_wm import (
                Pi3XIntrinsicsEstimator,
            )

            try:
                self._intrinsics_estimator = Pi3XIntrinsicsEstimator(
                    device=self.devices["intrinsics"],
                    logger=self.logger,
                )
            except Exception as exc:
                self.logger.warning(
                    "Pi3X weights could not be made resident; using calibrated intrinsics fallback: %s",
                    exc,
                )
        self.params_cls = GenerationParams
        self._image: Image.Image | None = None
        self._prompt = ""
        self._intrinsics: np.ndarray | None = None
        self._world_pose = np.eye(4, dtype=np.float32)
        self._configured = False
        self._chunk_index = 0
        self._seed = 42
        self._steps = DEFAULT_SAMPLING_STEPS
        self._cfg_scale = DEFAULT_CFG_SCALE
        self.last_metrics: dict[str, float] = {}
        self._state_lock = threading.RLock()

    def realtime_spec(self) -> RealtimeSpec:
        return RealtimeSpec(
            fps=self.fps,
            first_chunk_frames=self.output_frames,
            steady_chunk_frames=self.output_frames,
        )

    def runtime_info(self) -> dict[str, Any]:
        """Describe actual resident component placement without claiming parallelism."""

        return {
            "component_devices": {name: str(device) for name, device in self.devices.items()},
            "execution": "resident-component-partitioned",
            "tensor_parallel": False,
            "num_frames": self.num_frames,
            "sampling_steps": self._steps,
            "refiner_enabled": True,
            "prompt_updates": "chunk-boundary",
            "text_conditioning": ("stage1", "refiner"),
        }

    def _prepare_prompt_conditions(self, prompt: str) -> None:
        """Encode both text paths transactionally without touching world state."""

        stage1_key = self.pipeline._stage1_prompt_cache_key
        stage1_cache = self.pipeline._stage1_prompt_cache
        refiner = getattr(self.pipeline, "refiner", None)
        if refiner is None:
            raise RuntimeError(
                "SANA-WM prompt updates require the resident LTX-2 refiner."
            )
        refiner_key = refiner._cached_prompt
        refiner_cache = refiner._cached_prompt_tensors
        try:
            self.pipeline._encode_prompts(prompt, "")
            refiner._encode_prompt(prompt)
        except BaseException:
            # A failed second-stage encode must not leave half of the pipeline
            # on the new prompt. The old tensors remain live in these local
            # snapshots until both encoders have completed successfully.
            self.pipeline._stage1_prompt_cache_key = stage1_key
            self.pipeline._stage1_prompt_cache = stage1_cache
            refiner._cached_prompt = refiner_key
            refiner._cached_prompt_tensors = refiner_cache
            raise

    def update_prompt(self, prompt: str) -> bool:
        """Apply new text conditioning to the next complete native chunk.

        Only the stage-1 and refiner prompt caches are replaced. The current
        anchor frame, camera pose, intrinsics, seed policy, and chunk index are
        deliberately retained so the generated world remains continuous.
        """

        normalized = str(prompt or "").strip()
        if not normalized:
            raise ValueError("SANA-WM prompt updates require a non-empty prompt.")
        with self._state_lock:
            if not self._configured:
                raise RuntimeError(
                    "Configure the SANA-WM realtime session before updating its prompt."
                )
            if normalized == self._prompt:
                return False
            started = time.perf_counter()
            self._prepare_prompt_conditions(normalized)
            self._prompt = normalized
            self.last_metrics = {
                "condition_ms": (time.perf_counter() - started) * 1000.0,
            }
            return True

    def configure(
        self,
        *,
        image: Image.Image,
        prompt: str,
        seed: int = 42,
        fps: int = 16,
        num_frames: int | None = None,
        step: int = DEFAULT_SAMPLING_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
    ) -> dict[str, Any]:
        if int(fps) != self.fps:
            raise ValueError(f"SANA-WM realtime runs at the checkpoint-native {self.fps} FPS.")
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("SANA-WM realtime requires a user-provided text prompt.")
        if num_frames is not None:
            self.num_frames = _snap_num_frames(max(int(num_frames), 9))
            self.output_frames = self.num_frames - 1
        self._steps = max(int(step), 1)
        self._cfg_scale = float(cfg_scale)
        self._seed = int(seed)
        from worldfoundry.base_models.diffusion_model.image.sana.inference_video_scripts.inference_sana_wm import (
            TARGET_HEIGHT,
            TARGET_WIDTH,
            resize_and_center_crop,
            transform_intrinsics_for_crop,
        )

        original = image.convert("RGB")
        cropped, source_size, resized_size, crop_offset = resize_and_center_crop(
            original,
            target_h=TARGET_HEIGHT,
            target_w=TARGET_WIDTH,
        )
        intrinsics_started = time.perf_counter()
        intrinsics_source: np.ndarray
        if self._intrinsics_estimator is not None:
            try:
                intrinsics_source = self._intrinsics_estimator(original)
            except Exception as exc:
                self.logger.warning("Pi3X intrinsics failed; using a calibrated 55-degree fallback: %s", exc)
                width, height = original.size
                focal = 0.5 * width / math.tan(math.radians(55.0) / 2.0)
                intrinsics_source = np.asarray((focal, focal, width / 2.0, height / 2.0), dtype=np.float32)
        else:
            width, height = original.size
            focal = 0.5 * width / math.tan(math.radians(55.0) / 2.0)
            intrinsics_source = np.asarray((focal, focal, width / 2.0, height / 2.0), dtype=np.float32)
        intrinsics = transform_intrinsics_for_crop(
            intrinsics_source,
            source_size,
            resized_size,
            crop_offset,
        )

        self._image = cropped
        self._prompt = prompt
        self._intrinsics = np.broadcast_to(intrinsics, (self.num_frames, 4)).copy()
        self._world_pose = np.eye(4, dtype=np.float32)
        self._chunk_index = 0

        prompt_started = time.perf_counter()
        self._prepare_prompt_conditions(self._prompt)
        self._configured = True
        self.last_metrics = {
            "intrinsics_ms": (prompt_started - intrinsics_started) * 1000.0,
            "condition_ms": (time.perf_counter() - prompt_started) * 1000.0,
        }
        return {
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": dict(self.last_metrics),
            "runtime_info": self.runtime_info(),
        }

    def generate(
        self,
        *,
        interactions: Sequence[str] | None = None,
        control_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Advance one chunk, applying an optional prompt at its boundary."""

        with self._state_lock:
            condition_ms: float | None = None
            if prompt is not None and self.update_prompt(prompt):
                condition_ms = self.last_metrics.get("condition_ms")
            result = self._generate_current_prompt(
                interactions=interactions,
                control_segments=control_segments,
                seed=seed,
            )
            if condition_ms is not None:
                self.last_metrics["condition_ms"] = condition_ms
                result["realtime_metrics"]["condition_ms"] = condition_ms
            return result

    def _generate_current_prompt(
        self,
        *,
        interactions: Sequence[str] | None = None,
        control_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if not self._configured or self._image is None or self._intrinsics is None:
            raise RuntimeError("SANA-WM realtime session is not configured.")
        from worldfoundry.base_models.diffusion_model.image.sana.inference_video_scripts.inference_sana_wm import (
            action_string_to_c2w,
        )

        frame_keys = _frames_for_segments(
            control_segments,
            list(interactions or ()),
            frame_count=self.output_frames,
            fps=self.fps,
        )
        relative = action_string_to_c2w(
            _compress_action_frames(frame_keys),
            translation_speed=float(os.getenv("WORLDFOUNDRY_SANA_WM_TRANSLATION_PER_FRAME", "0.05")),
            rotation_speed_deg=float(os.getenv("WORLDFOUNDRY_SANA_WM_ROTATION_DEG_PER_FRAME", "1.2")),
        )
        c2w = np.einsum("ij,fjk->fik", self._world_pose, relative).astype(np.float32)
        params = self.params_cls(
            num_frames=self.num_frames,
            fps=self.fps,
            step=self._steps,
            cfg_scale=self._cfg_scale,
            seed=self._seed if seed is None else int(seed),
            negative_prompt="",
            sampling_algo="flow_euler_ltx",
        )
        started = time.perf_counter()
        result = self.pipeline.generate(
            self._image,
            self._prompt,
            c2w,
            self._intrinsics,
            params,
        )
        video = _validate_output_video(
            result["video"],
            expected_frames=self.output_frames,
        )
        # Copy only the anchor so Pillow cannot retain the full output chunk's
        # backing allocation across the next generation.
        self._image = Image.fromarray(video[-1].copy(), mode="RGB")
        self._world_pose = c2w[-1].copy()
        self._chunk_index += 1
        self.last_metrics = {
            key: float(value)
            for key, value in dict(result.get("realtime_metrics") or {}).items()
        }
        self.last_metrics["generation_ms"] = (time.perf_counter() - started) * 1000.0
        return {
            "frames": video,
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": dict(self.last_metrics),
            "runtime_info": self.runtime_info(),
        }

    def reset(self) -> None:
        with self._state_lock:
            self._image = None
            self._intrinsics = None
            self._prompt = ""
            self._world_pose = np.eye(4, dtype=np.float32)
            self._configured = False
            self._chunk_index = 0
            self.last_metrics = {}


__all__ = ["SanaWMRealtimeSession", "ensure_sana_import_paths"]
