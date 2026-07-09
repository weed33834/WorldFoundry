from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .runtime_env import (
    ensure_longvie_runtime,
    resolve_control_weight_path,
    resolve_dit_weight_path,
    resolve_wan21_i2v_dir,
    resolve_wan21_tokenizer_dir,
)


LONGVIE_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
TARGET_SIZE = (640, 352)


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def pop_first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.pop(key, None)
        if value is not None:
            return value
    return None


def pick(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def to_rgb_image(value: Any, *, target_size: tuple[int, int] = TARGET_SIZE) -> Image.Image:
    if value is None:
        raise ValueError("LongVie requires an input image.")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 1:
            raise ValueError(f"LongVie image input expects one image, got {len(value)}.")
        value = value[0]
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    elif isinstance(value, (str, Path)):
        image = Image.open(value).convert("RGB")
    else:
        array = np.asarray(value)
        if array.ndim == 4:
            array = array[0]
        if array.ndim != 3:
            raise ValueError(f"Unsupported LongVie image shape: {array.shape}")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255)
            array = array.astype(np.uint8)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        image = Image.fromarray(array[..., :3]).convert("RGB")
    return image.resize(target_size)


def video_to_frames(value: Any, *, target_size: tuple[int, int] = TARGET_SIZE) -> list[Image.Image]:
    if value is None:
        raise ValueError("LongVie requires dense and sparse control videos.")
    if isinstance(value, (str, Path)):
        ensure_longvie_runtime()
        import decord

        video = decord.VideoReader(str(value))
        return [Image.fromarray(frame).convert("RGB").resize(target_size) for frame in video[:].asnumpy()]
    if isinstance(value, np.ndarray):
        frames = value
        if frames.ndim == 3:
            frames = frames[None, ...]
        return [to_rgb_image(frame, target_size=target_size) for frame in frames]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_rgb_image(frame, target_size=target_size) for frame in value]
    raise TypeError(f"Unsupported LongVie video input type: {type(value).__name__}")


class LongVieOfficialRuntime:
    """Lazy bridge to the vendored official LongVie DiffSynth pipeline."""

    def __init__(
        self,
        *,
        control_weight_path: str | Path | None,
        dit_weight_path: str | Path | None,
        weight_dir: str | Path | None,
        wan_base_dir: str | Path | None,
        tokenizer_dir: str | Path | None,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        use_usp: bool = False,
        ring_degree: int = 1,
        ulysses_degree: int = 1,
        enable_vram_management: bool = True,
        control_layers: int = 12,
        variant: str = "longvie-1",
    ) -> None:
        self.control_weight_path = control_weight_path
        self.dit_weight_path = dit_weight_path
        self.weight_dir = weight_dir
        self.wan_base_dir = wan_base_dir
        self.tokenizer_dir = tokenizer_dir
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_usp = bool(use_usp)
        self.ring_degree = int(ring_degree)
        self.ulysses_degree = int(ulysses_degree)
        self.enable_vram_management = bool(enable_vram_management)
        self.control_layers = int(control_layers)
        self.variant = variant
        self.pipe: Any = None

    @property
    def loaded(self) -> bool:
        return self.pipe is not None

    def _torch_dtype(self):
        import torch

        if self.torch_dtype in {"bf16", "bfloat16", "torch.bfloat16"}:
            return torch.bfloat16
        if self.torch_dtype in {"fp16", "float16", "torch.float16"}:
            return torch.float16
        if self.torch_dtype in {"fp32", "float32", "torch.float32"}:
            return torch.float32
        return torch.bfloat16

    def _model_configs(self):
        from worldfoundry.base_models.diffusion_model.diffsynth.pipelines.wan_video_new_longvie import (
            ModelConfig,
        )

        base_dir = resolve_wan21_i2v_dir(self.wan_base_dir)
        diffusion_paths = sorted(str(path) for path in base_dir.glob("diffusion_pytorch_model*.safetensors"))
        return [
            ModelConfig(path=diffusion_paths, offload_device="cpu"),
            ModelConfig(path=str(base_dir / "models_t5_umt5-xxl-enc-bf16.pth"), offload_device="cpu"),
            ModelConfig(path=str(base_dir / "Wan2.1_VAE.pth"), offload_device="cpu"),
            ModelConfig(
                path=str(base_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                offload_device="cpu",
            ),
        ]

    def load(self) -> Any:
        if self.pipe is not None:
            return self.pipe
        ensure_longvie_runtime()
        try:
            from worldfoundry.base_models.diffusion_model.diffsynth.pipelines.wan_video_new_longvie import (
                LongViePipeline,
                ModelConfig,
            )
        except ModuleNotFoundError as exc:
            missing = exc.name or str(exc)
            raise RuntimeError(
                "LongVie execute=True requires the official video-generation runtime dependencies. "
                "Use the worldfoundry-longvie-official-cu118 runtime env or the runtime profile pip package list. "
                f"Missing module: {missing}."
            ) from exc

        dtype = self._torch_dtype()
        tokenizer_path = resolve_wan21_tokenizer_dir(self.tokenizer_dir)
        control_path = resolve_control_weight_path(self.control_weight_path, weight_dir=self.weight_dir)
        dit_path = resolve_dit_weight_path(
            self.dit_weight_path,
            weight_dir=self.weight_dir,
            required=self.variant == "longvie-2",
        )
        self.pipe = LongViePipeline.from_pretrained(
            torch_dtype=dtype,
            device=self.device,
            use_usp=self.use_usp,
            model_configs=self._model_configs(),
            tokenizer_config=ModelConfig(path=str(tokenizer_path)),
            redirect_common_files=False,
            control_weight_path=str(control_path),
            dit_weight_path=str(dit_path or ""),
            ring_degree=self.ring_degree,
            ulysses_degree=self.ulysses_degree,
            control_layers=self.control_layers,
        )
        if self.enable_vram_management:
            self.pipe.enable_vram_management()
        return self.pipe

    def generate_segment(self, **kwargs: Any) -> tuple[Any, Any]:
        pipe = self.load()
        signature = inspect.signature(pipe.__call__)
        has_var_kwargs = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        if not has_var_kwargs:
            allowed = set(signature.parameters)
            kwargs = {key: value for key, value in kwargs.items() if key in allowed}
        result = pipe(**kwargs)
        if isinstance(result, tuple):
            return result
        return result, None

    @staticmethod
    def save_video(video: Any, output_path: str | Path, *, fps: int = 16, quality: int = 10) -> Path:
        ensure_longvie_runtime()
        from worldfoundry.core.io import save_video

        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        save_video(video, str(target), fps=fps, quality=quality)
        return target


__all__ = [
    "LONGVIE_NEGATIVE_PROMPT",
    "TARGET_SIZE",
    "LongVieOfficialRuntime",
    "first_present",
    "pick",
    "pop_first",
    "to_rgb_image",
    "video_to_frames",
]
