"""Module for base_models -> diffusion_model -> video -> cosmos3 -> worldfoundry_runtime.py functionality."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import DEFAULT_COSMOS3_REPO_ID, find_local_artifact_path, resolve_local_artifact_path


_DIFFUSERS_MARKERS = ("model_index.json", "model.safetensors.index.json")
_SUPPORTED_TASKS = (
    "text-to-image",
    "text-to-video",
    "image-to-image",
    "image-to-video",
    "video-to-video",
    "world-generation",
)


@dataclass(frozen=True)
class Cosmos3RuntimePlan:
    """Cosmos runtime plan implementation."""
    model_path: str
    resolved_model_path: str | None
    backend: str
    supported_tasks: tuple[str, ...]
    blocked: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """To dict.

        Returns:
            The return value.
        """
        return asdict(self)


def _has_diffusers_layout(path: Path | None) -> bool:
    """Helper function to has diffusers layout.

    Args:
        path: The path.

    Returns:
        The return value.
    """
    if path is None or not path.is_dir():
        return False
    return (path / "model_index.json").is_file()


def _torch_dtype(dtype: Any) -> Any:
    """Helper function to torch dtype.

    Args:
        dtype: The dtype.

    Returns:
        The return value.
    """
    if dtype is None or not isinstance(dtype, str):
        return dtype

    import torch

    normalized = dtype.lower().replace("torch.", "")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported Cosmos3 torch dtype: {dtype!r}")
    return aliases[normalized]


class Cosmos3Runtime:
    """Lazy in-tree Cosmos3 runtime backed by the vendored diffusers pipeline."""

    def __init__(
        self,
        pipeline: Any,
        model_path: str,
        device: str = "cuda",
        backend: str = "diffusers",
    ) -> None:
        """Init.

        Args:
            pipeline: The pipeline.
            model_path: The model path.
            device: The device.
            backend: The backend.

        Returns:
            The return value.
        """
        self.pipeline = pipeline
        self.model_path = model_path
        self.device = device
        self.backend = backend

    @classmethod
    def plan(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        backend: str = "diffusers",
        **_: Any,
    ) -> dict[str, Any]:
        """Plan.

        Args:
            model_path: The model path.
            backend: The backend.

        Returns:
            The return value.
        """
        if isinstance(model_path, Mapping):
            raw_model_path = str(model_path.get("pretrained_model_path") or model_path.get("model_path") or DEFAULT_COSMOS3_REPO_ID)
        else:
            raw_model_path = str(model_path or DEFAULT_COSMOS3_REPO_ID)

        resolved = find_local_artifact_path(raw_model_path)
        blockers: list[str] = []
        if resolved is None:
            blockers.append("Missing Cosmos3 checkpoint under local path or WORLDFOUNDRY_HFD_ROOT.")
        elif backend == "diffusers" and not _has_diffusers_layout(resolved):
            blockers.append("Cosmos3 checkpoint is present but is not a converted diffusers pipeline directory.")
        elif backend != "diffusers":
            blockers.append(f"Unsupported Cosmos3 backend: {backend!r}.")

        return Cosmos3RuntimePlan(
            model_path=raw_model_path,
            resolved_model_path=None if resolved is None else str(resolved),
            backend=backend,
            supported_tasks=_SUPPORTED_TASKS,
            blocked=bool(blockers),
            blockers=tuple(blockers),
        ).to_dict()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        *,
        device: str = "cuda",
        backend: str = "diffusers",
        torch_dtype: Any = "bfloat16",
        device_map: Any = None,
        **kwargs: Any,
    ) -> "Cosmos3Runtime":
        """From pretrained.

        Args:
            model_path: The model path.

        Returns:
            The return value.
        """
        if backend != "diffusers":
            raise ValueError(f"Unsupported Cosmos3 backend: {backend!r}")

        if isinstance(model_path, Mapping):
            raw_model_path = str(model_path.get("pretrained_model_path") or model_path.get("model_path") or DEFAULT_COSMOS3_REPO_ID)
            kwargs = {**{k: v for k, v in model_path.items() if k not in {"pretrained_model_path", "model_path"}}, **kwargs}
        else:
            raw_model_path = str(model_path or DEFAULT_COSMOS3_REPO_ID)

        resolved_model_path = resolve_local_artifact_path(raw_model_path)
        if not _has_diffusers_layout(resolved_model_path):
            raise FileNotFoundError(
                "Cosmos3 diffusers pipeline layout was not found. Convert or stage a pipeline directory "
                f"with one of {_DIFFUSERS_MARKERS!r}: {resolved_model_path}"
            )

        from .diffusers_cosmos3 import Cosmos3OmniDiffusersPipeline

        load_kwargs = dict(kwargs)
        load_kwargs["torch_dtype"] = _torch_dtype(torch_dtype)
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        pipeline = Cosmos3OmniDiffusersPipeline.from_pretrained(str(resolved_model_path), **load_kwargs)
        if device_map is None and hasattr(pipeline, "to"):
            pipeline = pipeline.to(device)
        return cls(pipeline=pipeline, model_path=str(resolved_model_path), device=device, backend=backend)

    def api_init(self, *_: Any, **__: Any) -> None:
        """Api init.

        Returns:
            The return value.
        """
        raise NotImplementedError("Cosmos3Runtime is a local in-tree runtime and does not use an API backend.")

    def predict(
        self,
        prompt: str | list[str],
        *,
        image: Any = None,
        negative_prompt: str | list[str] | None = None,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: float = 24.0,
        seed: int | None = None,
        output_type: str = "video",
        **kwargs: Any,
    ) -> Any:
        """Predict.

        Args:
            prompt: The prompt.

        Returns:
            The return value.
        """
        generator = kwargs.pop("generator", None)
        if seed is not None and generator is None:
            import torch

            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        return self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            generator=generator,
            device=self.device,
            output_type=output_type,
            **kwargs,
        )

    def decode_latents(self, latents: list[Any]) -> list[Any]:
        """Decode latents.

        Args:
            latents: The latents.

        Returns:
            The return value.
        """
        return self.pipeline.decode_latents(latents)


__all__ = ["Cosmos3Runtime", "Cosmos3RuntimePlan"]
