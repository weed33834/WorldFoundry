"""WoW synthesis adapters for the official in-tree inference demos."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image

from ...base_synthesis import BaseSynthesis


WAN_DEFAULT_NEGATIVE_PROMPT = "low quality, distorted, ugly, bad anatomy"
DIT_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
    "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)


def _as_local_dir(path: str | os.PathLike[str], label: str) -> Path:
    model_root = Path(path).expanduser()
    if model_root.is_dir():
        return model_root

    path_text = str(path)
    is_hf_repo_id = (
        "/" in path_text
        and not path_text.startswith(("/", ".", "~"))
        and "://" not in path_text
    )
    if is_hf_repo_id:
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:  # pragma: no cover - depends on optional runtime env
            raise FileNotFoundError(
                f"{label} points to Hugging Face repo {path_text!r}, but huggingface_hub is not available."
            ) from exc
        return Path(snapshot_download(repo_id=path_text))

    raise FileNotFoundError(
        f"{label} must be a local checkpoint directory or Hugging Face repo id for WoW in-tree inference: {model_root}"
    )


def _find_wan_dit_paths(model_root: Path) -> list[str]:
    sharded = [model_root / f"diffusion_pytorch_model-0000{i}-of-00007.safetensors" for i in range(1, 8)]
    if all(path.is_file() for path in sharded):
        return [str(path) for path in sharded]

    single = model_root / "diffusion_pytorch_model.safetensors"
    if single.is_file():
        return [str(single)]

    discovered = sorted(model_root.glob("diffusion_pytorch_model*.safetensors"))
    if discovered:
        return [str(path) for path in discovered]

    raise FileNotFoundError(
        "WoW Wan checkpoint directory is missing diffusion_pytorch_model*.safetensors files: "
        f"{model_root}"
    )


def _custom_checkpoint_path(model_root: Path, custom_checkpoint_name: str | os.PathLike[str] | None) -> Path | None:
    if not custom_checkpoint_name:
        return None
    candidate = Path(custom_checkpoint_name).expanduser()
    return candidate if candidate.is_absolute() else model_root / candidate


def _wow_dit_runtime_dir() -> Path:
    return Path(__file__).resolve().parent / "wow_runtime" / "dit_models" / "wow-dit-2b"


def _load_wow_dit_video2world_module() -> Any:
    runtime_dir = _wow_dit_runtime_dir()
    module_path = runtime_dir / "video2world.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"WoW DiT video2world runtime not found: {module_path}")

    # The official script imports the sibling imaginaire package as a top-level module.
    runtime_dir_text = str(runtime_dir)
    if runtime_dir_text not in sys.path:
        sys.path.insert(0, runtime_dir_text)

    module_name = "worldfoundry_wow_dit2b_video2world"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load WoW DiT video2world module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_dit_path(
    pretrained_model_path: str | os.PathLike[str] | None,
    *,
    model_size: str,
    resolution: str,
    fps: int,
    explicit_dit_path: str | os.PathLike[str] | None,
) -> str:
    if explicit_dit_path:
        candidate = Path(explicit_dit_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"WoW DiT checkpoint not found: {candidate}")
        return str(candidate)

    if pretrained_model_path:
        candidate = Path(pretrained_model_path).expanduser()
        if candidate.is_file():
            return str(candidate)
        if candidate.is_dir():
            names = (
                "wow_dit_2b.pt",
                f"model-{resolution}p-{fps}fps.pt",
                "model.pt",
            )
            for name in names:
                nested = candidate / name
                if nested.is_file():
                    return str(nested)

    default_path = _wow_dit_runtime_dir() / "checkpoints" / "wow_dit_2b.pt"
    if default_path.is_file():
        return str(default_path)

    raise FileNotFoundError(
        "WoW DiT backend requires a local DiT checkpoint file. Pass dit_path or model_path; "
        f"looked for {default_path} for model_size={model_size}, resolution={resolution}, fps={fps}."
    )


def _materialize_image_input(input_image: Image.Image, output_path: str | os.PathLike[str] | None) -> str:
    if output_path:
        target = Path(output_path).expanduser().with_suffix(".input.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        input_image.save(target)
        return str(target)
    handle = tempfile.NamedTemporaryFile(prefix="worldfoundry_wow_input_", suffix=".png", delete=False)
    handle.close()
    input_image.save(handle.name)
    return handle.name


def _option(options: Mapping[str, Any], args: Any, name: str, default: Any) -> Any:
    value = options.get(name)
    if value is not None:
        return value
    if args is not None and hasattr(args, name):
        value = getattr(args, name)
        if value is not None:
            return value
    return default


class WoWSynthesis(BaseSynthesis):
    """WoW synthesis wrapper matching the official Wan and DiT demo parameters."""

    def __init__(
        self,
        pipeline: Any,
        *,
        backend: str,
        device: str = "cuda",
        runtime_module: Any | None = None,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.backend = backend
        self.device = device
        self.runtime_module = runtime_module

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | Mapping[str, Any] | None = "WoW-world-model/WoW-1-Wan-14B-600k",
        synthesis_args: Any = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> "WoWSynthesis":
        """Load the official WoW Wan demo backend by default.

        The official repository marks the Wan demo as recommended. The DiT 2B
        backend remains available by passing ``runtime_backend="dit2b"`` and a
        local ``dit_path`` or checkpoint file as ``model_path``.
        """

        options: dict[str, Any] = dict(kwargs)
        if isinstance(pretrained_model_path, Mapping):
            options.update(pretrained_model_path)
            pretrained_model_path = (
                options.pop("model_path", None)
                or options.pop("pretrained_model_path", None)
                or options.pop("checkpoint_folder", None)
                or options.pop("repo_root", None)
            )

        backend = str(
            options.pop("runtime_backend", None)
            or getattr(synthesis_args, "runtime_backend", "wan")
            or "wan"
        ).lower()
        if backend in {"dit", "dit-2b", "dit2b", "cosmos", "cosmos2"}:
            return cls._from_dit2b(
                pretrained_model_path=pretrained_model_path,
                synthesis_args=synthesis_args,
                device=device,
                **options,
            )
        return cls._from_wan(
            pretrained_model_path=pretrained_model_path,
            synthesis_args=synthesis_args,
            device=device,
            **options,
        )

    @classmethod
    def _from_wan(
        cls,
        *,
        pretrained_model_path: str | os.PathLike[str] | None,
        synthesis_args: Any,
        device: str,
        **kwargs: Any,
    ) -> "WoWSynthesis":
        if pretrained_model_path is None:
            raise FileNotFoundError("WoW Wan backend requires a local checkpoint folder.")
        model_root = _as_local_dir(pretrained_model_path, "WoW Wan checkpoint")
        custom_checkpoint_name = kwargs.pop(
            "custom_checkpoint",
            kwargs.pop("custom_checkpoint_name", getattr(synthesis_args, "custom_checkpoint", "WoW_video_dit.pt")),
        )

        clip_model_path = model_root / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        t5_model_path = model_root / "models_t5_umt5-xxl-enc-bf16.pth"
        vae_model_path = model_root / "Wan2.1_VAE.pth"
        for label, path in (
            ("CLIP image encoder", clip_model_path),
            ("T5 text encoder", t5_model_path),
            ("Wan VAE", vae_model_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"WoW Wan {label} checkpoint not found: {path}")

        from ....base_models.diffusion_model.diffsynth import ModelManager, Wan22VideoPusaPipeline

        mm = ModelManager(device="cpu")
        mm.load_models([str(clip_model_path)], torch_dtype=torch.float32)
        mm.load_models(
            [_find_wan_dit_paths(model_root), str(t5_model_path), str(vae_model_path)],
            torch_dtype=torch.bfloat16,
        )

        custom_checkpoint = _custom_checkpoint_path(model_root, custom_checkpoint_name)
        if custom_checkpoint is not None:
            if custom_checkpoint.is_file():
                state_dict = torch.load(str(custom_checkpoint), map_location="cpu")
                dit_model = mm.fetch_model("wan_video_dit")
                if dit_model is not None:
                    dit_model.load_state_dict(state_dict, strict=False)
                    print(f"[WoWSynthesis] Loaded custom DiT checkpoint: {custom_checkpoint}")
            else:
                print(f"[WoWSynthesis] Custom checkpoint not found, using base DiT: {custom_checkpoint}")

        pipeline = Wan22VideoPusaPipeline.from_model_manager(mm, torch_dtype=torch.bfloat16, device=device)

        enable_vram = (
            getattr(synthesis_args, "enable_vram_management", True)
            and not getattr(synthesis_args, "no_vram_management", False)
        )
        if enable_vram:
            persistent_param_gb = getattr(synthesis_args, "persistent_param_gb", 70)
            pipeline.enable_vram_management(num_persistent_param_in_dit=int(persistent_param_gb * 10**9))

        return cls(pipeline=pipeline, backend="wan", device=device)

    @classmethod
    def _from_dit2b(
        cls,
        *,
        pretrained_model_path: str | os.PathLike[str] | None,
        synthesis_args: Any,
        device: str,
        **kwargs: Any,
    ) -> "WoWSynthesis":
        module = _load_wow_dit_video2world_module()
        model_size = str(_option(kwargs, synthesis_args, "model_size", "2B"))
        resolution = str(_option(kwargs, synthesis_args, "resolution", "720"))
        fps = int(_option(kwargs, synthesis_args, "fps", 16))
        setup_args = Namespace(
            model_size=model_size,
            resolution=resolution,
            fps=fps,
            dit_path=_resolve_dit_path(
                pretrained_model_path,
                model_size=model_size,
                resolution=resolution,
                fps=fps,
                explicit_dit_path=_option(kwargs, synthesis_args, "dit_path", None),
            ),
            seed=int(_option(kwargs, synthesis_args, "seed", 42)),
            num_gpus=int(_option(kwargs, synthesis_args, "num_gpus", 1)),
            disable_guardrail=bool(_option(kwargs, synthesis_args, "disable_guardrail", True)),
            offload_guardrail=bool(_option(kwargs, synthesis_args, "offload_guardrail", False)),
            disable_prompt_refiner=bool(_option(kwargs, synthesis_args, "disable_prompt_refiner", True)),
            offload_prompt_refiner=bool(_option(kwargs, synthesis_args, "offload_prompt_refiner", False)),
        )
        pipeline = module.setup_pipeline(setup_args)
        return cls(pipeline=pipeline, backend="dit2b", device=device, runtime_module=module)

    @torch.no_grad()
    def predict(
        self,
        input_image: Image.Image | None = None,
        text_prompt: str = "",
        synthesis_args: Any = None,
        input_path: str | os.PathLike[str] | None = None,
        output_path: str | os.PathLike[str] | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Generate a WoW video and optionally save/return a runner mapping."""

        if self.backend == "dit2b":
            result = self._predict_dit2b(
                input_image=input_image,
                input_path=input_path,
                text_prompt=text_prompt,
                synthesis_args=synthesis_args,
                output_path=output_path,
                return_dict=return_dict,
                **kwargs,
            )
        else:
            result = self._predict_wan(
                input_image=input_image,
                text_prompt=text_prompt,
                synthesis_args=synthesis_args,
                output_path=output_path,
                fps=fps,
                return_dict=return_dict,
                **kwargs,
            )
        return result

    def _predict_wan(
        self,
        *,
        input_image: Image.Image | None,
        text_prompt: str,
        synthesis_args: Any,
        output_path: str | os.PathLike[str] | None,
        fps: int | None,
        return_dict: bool,
        **kwargs: Any,
    ) -> Any:
        if input_image is None:
            raise ValueError("WoW Wan backend requires an input image or first frame.")
        steps = int(_option(kwargs, synthesis_args, "steps", 50))
        seed = int(_option(kwargs, synthesis_args, "seed", 42))
        tiled = bool(_option(kwargs, synthesis_args, "tiled", not bool(getattr(synthesis_args, "no_tiled", False))))
        num_frames = int(_option(kwargs, synthesis_args, "num_frames", 41))
        negative_prompt = str(_option(kwargs, synthesis_args, "negative_prompt", WAN_DEFAULT_NEGATIVE_PROMPT))
        output_fps = int(fps if fps is not None else _option(kwargs, synthesis_args, "output_fps", 15))

        output_video = self.pipeline(
            prompt=text_prompt,
            negative_prompt=negative_prompt,
            input_image=input_image,
            num_inference_steps=steps,
            seed=seed,
            tiled=tiled,
            num_frames=num_frames,
        )

        artifact_path = ""
        if output_path is not None:
            from worldfoundry.core.io.video import write_video

            artifact_path = str(output_path)
            write_video(output_video, artifact_path, fps=output_fps, quality=5)

        if return_dict:
            return {
                "status": "ok",
                "runtime": "wow-wan-official-demo",
                "backend_quality": "official_in_tree_runtime",
                "artifact_kind": "generated_video",
                "artifact_path": artifact_path,
                "video": output_video,
                "metadata": {
                    "steps": steps,
                    "seed": seed,
                    "tiled": tiled,
                    "num_frames": num_frames,
                    "fps": output_fps,
                    "negative_prompt": negative_prompt,
                },
            }
        return output_video

    def _predict_dit2b(
        self,
        *,
        input_image: Image.Image | None,
        input_path: str | os.PathLike[str] | None,
        text_prompt: str,
        synthesis_args: Any,
        output_path: str | os.PathLike[str] | None,
        return_dict: bool,
        **kwargs: Any,
    ) -> Any:
        if input_path is None:
            if input_image is None:
                raise ValueError("WoW DiT backend requires input_path or input_image.")
            input_path = _materialize_image_input(input_image, output_path)

        negative_prompt = str(_option(kwargs, synthesis_args, "negative_prompt", DIT_DEFAULT_NEGATIVE_PROMPT))
        num_conditional_frames = int(_option(kwargs, synthesis_args, "num_conditional_frames", 1))
        guidance = float(_option(kwargs, synthesis_args, "guidance", 7.0))
        seed = int(_option(kwargs, synthesis_args, "seed", 42))
        num_sampling_step = int(_option(kwargs, synthesis_args, "num_sampling_step", 35))

        output_video = self.pipeline(
            prompt=text_prompt,
            negative_prompt=negative_prompt,
            input_path=str(input_path),
            num_conditional_frames=num_conditional_frames,
            guidance=guidance,
            seed=seed,
            num_sampling_step=num_sampling_step,
        )

        artifact_path = ""
        save_fps = 10 if getattr(getattr(self.pipeline, "config", None), "state_t", None) == 16 else 16
        if output_path is not None and output_video is not None:
            artifact_path = str(output_path)
            self.runtime_module.save_image_or_video(output_video, artifact_path, fps=save_fps)

        if return_dict:
            return {
                "status": "ok" if output_video is not None else "blocked",
                "runtime": "wow-dit2b-official-demo",
                "backend_quality": "official_in_tree_runtime",
                "artifact_kind": "generated_video",
                "artifact_path": artifact_path,
                "video": output_video,
                "metadata": {
                    "input_path": str(input_path),
                    "num_conditional_frames": num_conditional_frames,
                    "guidance": guidance,
                    "seed": seed,
                    "num_sampling_step": num_sampling_step,
                    "fps": save_fps,
                    "negative_prompt": negative_prompt,
                },
            }
        return output_video
