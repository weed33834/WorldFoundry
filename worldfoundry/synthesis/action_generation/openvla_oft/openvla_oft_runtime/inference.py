from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from .vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType, PROPRIO_DIM


OPENVLA_IMAGE_SIZE = 224
OFFICIAL_COMPONENT_FILES = {
    "moojink/openvla-7b-oft-finetuned-libero-spatial": {
        "action_head": "action_head--150000_checkpoint.pt",
        "proprio_projector": "proprio_projector--150000_checkpoint.pt",
    },
    "moojink/openvla-7b-oft-finetuned-libero-object": {
        "action_head": "action_head--150000_checkpoint.pt",
        "proprio_projector": "proprio_projector--150000_checkpoint.pt",
    },
    "moojink/openvla-7b-oft-finetuned-libero-goal": {
        "action_head": "action_head--50000_checkpoint.pt",
        "proprio_projector": "proprio_projector--50000_checkpoint.pt",
    },
    "moojink/openvla-7b-oft-finetuned-libero-10": {
        "action_head": "action_head--150000_checkpoint.pt",
        "proprio_projector": "proprio_projector--150000_checkpoint.pt",
    },
    "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": {
        "action_head": "action_head--300000_checkpoint.pt",
        "proprio_projector": "proprio_projector--300000_checkpoint.pt",
    },
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        try:
            return _jsonable(value.detach().cpu())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_device(torch: Any, requested: str) -> str:
    device = requested or "cuda"
    if device == "cuda":
        device = "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("OpenVLA-OFT requested CUDA, but torch.cuda.is_available() is false.")
    return device


def _resolve_dtype(torch: Any, requested: str | None) -> tuple[Any, str]:
    dtype_name = str(requested or "bfloat16").lower()
    if dtype_name in {"auto", "bf16", "bfloat16"}:
        return torch.bfloat16, "bfloat16"
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16, "float16"
    if dtype_name in {"fp32", "float32"}:
        return torch.float32, "float32"
    raise ValueError(f"Unsupported OpenVLA-OFT torch_dtype value: {requested}")


def _is_hf_repo_id(value: str) -> bool:
    return value.count("/") == 1 and not value.startswith((".", "/", "~", "${"))


def _load_component_state_dict(torch: Any, checkpoint_path: Path) -> dict[str, Any]:
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _find_component_file(checkpoint_dir: Path, repo_id: str | None, kind: str) -> Path:
    filename = OFFICIAL_COMPONENT_FILES.get(repo_id or "", {}).get(kind)
    if filename:
        candidate = checkpoint_dir / filename
        if candidate.is_file():
            return candidate
    matches = sorted(path for path in checkpoint_dir.glob(f"*{kind}*checkpoint*.pt") if path.is_file())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"OpenVLA-OFT {kind} checkpoint was not found in {checkpoint_dir}.")
    raise FileExistsError(f"OpenVLA-OFT found multiple {kind} checkpoints in {checkpoint_dir}: {matches}")


def _snapshot_or_local_dir(checkpoint_location: str, *, cache_dir: str | None = None, local_files_only: bool = False) -> tuple[Path, str | None]:
    expanded = Path(checkpoint_location).expanduser()
    if expanded.exists():
        if not expanded.is_dir():
            raise NotADirectoryError(f"OpenVLA-OFT checkpoint must be a directory: {expanded}")
        return expanded.resolve(), None
    if _is_hf_repo_id(checkpoint_location):
        from huggingface_hub import snapshot_download

        try:
            snapshot = snapshot_download(
                repo_id=checkpoint_location,
                cache_dir=cache_dir or None,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            if local_files_only:
                raise
            try:
                snapshot = snapshot_download(
                    repo_id=checkpoint_location,
                    cache_dir=cache_dir or None,
                    local_files_only=True,
                )
            except Exception:
                raise exc
        return Path(snapshot).resolve(), checkpoint_location
    raise FileNotFoundError(f"OpenVLA-OFT checkpoint directory does not exist: {checkpoint_location}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_uint8_array(image: Any) -> np.ndarray:
    from PIL import Image

    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    elif isinstance(image, (str, Path)):
        image_path = Path(image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"OpenVLA-OFT image path does not exist: {image_path}")
        array = np.asarray(Image.open(image_path).convert("RGB"))
    else:
        try:
            import torch

            if isinstance(image, torch.Tensor):
                tensor = image.detach().cpu()
                if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
                    tensor = tensor.permute(1, 2, 0)
                array = tensor.numpy()
            else:
                array = np.asarray(image)
        except ImportError:
            array = np.asarray(image)

    if array.ndim != 3:
        raise ValueError(f"OpenVLA-OFT image must be HxWxC or CxHxW, got shape {array.shape}.")
    if array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"OpenVLA-OFT image must have 3 channels, got shape {array.shape}.")
    if array.dtype.kind == "f":
        max_value = 1.0 if float(np.nanmax(array)) <= 1.0 else 255.0
        array = np.clip(array, 0.0, max_value) / max_value * 255.0
    return np.asarray(array, dtype=np.uint8)


def _prepare_image(image: Any, *, center_crop: bool) -> Any:
    from PIL import Image

    pil = Image.fromarray(_to_uint8_array(image)).convert("RGB")
    if pil.size != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE):
        pil = pil.resize((OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE), Image.Resampling.LANCZOS)
    if center_crop:
        crop_side = int(round(OPENVLA_IMAGE_SIZE * math.sqrt(0.9)))
        left = (OPENVLA_IMAGE_SIZE - crop_side) // 2
        top = (OPENVLA_IMAGE_SIZE - crop_side) // 2
        pil = pil.crop((left, top, left + crop_side, top + crop_side))
        pil = pil.resize((OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE), Image.Resampling.LANCZOS)
    return pil


def _present_image(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value:
        return False
    return True


def _first_present_image(*values: Any) -> Any:
    for value in values:
        if _present_image(value):
            return value
    return None


def _select_images(image: Any, observation: Mapping[str, Any], *, num_images: int) -> list[Any]:
    primary = _first_present_image(
        observation.get("full_image"),
        observation.get("image"),
        observation.get("rgb"),
        observation.get("ref_image_path"),
        image,
    )
    if primary is None:
        raise ValueError("OpenVLA-OFT requires observation.full_image or an image input.")

    images = [primary]
    if num_images <= 1:
        return images

    wrist_keys = ["wrist_image", "full_image_wrist", "image_wrist", "wrist"]
    seen = set()
    for key in wrist_keys + sorted(key for key in observation if "wrist" in str(key)):
        if key in seen:
            continue
        seen.add(key)
        value = observation.get(key)
        if _present_image(value):
            images.append(value)
        if len(images) >= num_images:
            break
    if len(images) < num_images:
        raise ValueError(
            f"OpenVLA-OFT checkpoint expects {num_images} images, but only {len(images)} were provided."
        )
    return images[:num_images]


def _resolve_unnorm_key(norm_stats: Mapping[str, Any], requested: str | None, task_suite_name: str | None) -> str:
    candidates = [requested, task_suite_name, f"{task_suite_name}_no_noops" if task_suite_name else None]
    for candidate in candidates:
        if candidate and candidate in norm_stats:
            return str(candidate)
    for key in norm_stats:
        if task_suite_name and str(key).startswith(task_suite_name):
            return str(key)
    if len(norm_stats) == 1:
        return str(next(iter(norm_stats)))
    raise KeyError(f"OpenVLA-OFT unnorm_key was not found. requested={requested!r}; available={list(norm_stats)}")


def _normalize_proprio(proprio: Any, norm_stats: Mapping[str, Any]) -> np.ndarray:
    proprio_array = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if proprio_array.shape[0] != PROPRIO_DIM:
        raise ValueError(f"OpenVLA-OFT LIBERO proprio state must have {PROPRIO_DIM} values, got {proprio_array.shape}.")
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = np.asarray(norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool)), dtype=bool)
        high = np.asarray(norm_stats["max"], dtype=np.float32)
        low = np.asarray(norm_stats["min"], dtype=np.float32)
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = np.asarray(norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool)), dtype=bool)
        high = np.asarray(norm_stats["q99"], dtype=np.float32)
        low = np.asarray(norm_stats["q01"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported OpenVLA-OFT proprio normalization: {ACTION_PROPRIO_NORMALIZATION_TYPE}")
    return np.clip(
        np.where(mask, 2 * (proprio_array - low) / (high - low + 1e-8) - 1, proprio_array),
        a_min=-1.0,
        a_max=1.0,
    )


@dataclass(frozen=True)
class OpenVLAOFTRuntimeConfig:
    checkpoint_location: str
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    cache_dir: str | None = None
    local_files_only: bool = False
    attn_implementation: str = "eager"
    unnorm_key: str = "libero_spatial_no_noops"
    task_suite_name: str = "libero_spatial"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_proprio: bool = True
    num_images_in_input: int = 2
    center_crop: bool = True
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50


class OpenVLAOFTRuntime:
    """Checkpoint-backed OpenVLA-OFT inference with in-tree model code and HF asset resolution."""

    def __init__(self, config: OpenVLAOFTRuntimeConfig) -> None:
        self.config = config
        self.checkpoint_dir: Path | None = None
        self.repo_id: str | None = None
        self.device = ""
        self.dtype: Any | None = None
        self.dtype_name = ""
        self.processor: Any | None = None
        self.model: Any | None = None
        self.action_head: Any | None = None
        self.proprio_projector: Any | None = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        import torch
        from transformers import LlamaTokenizerFast

        from .action_heads import DiffusionActionHead, L1RegressionActionHead
        from .configuration_prismatic import OpenVLAConfig
        from .modeling_prismatic import OpenVLAForActionPrediction
        from .processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from .projectors import ProprioProjector

        checkpoint_dir, repo_id = _snapshot_or_local_dir(
            self.config.checkpoint_location,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only,
        )
        self.checkpoint_dir = checkpoint_dir
        self.repo_id = repo_id or self.config.checkpoint_location if _is_hf_repo_id(self.config.checkpoint_location) else repo_id
        self.device = _resolve_device(torch, self.config.device)
        self.dtype, self.dtype_name = _resolve_dtype(torch, self.config.torch_dtype)

        image_processor = PrismaticImageProcessor.from_pretrained(str(checkpoint_dir), local_files_only=True)
        tokenizer = LlamaTokenizerFast.from_pretrained(str(checkpoint_dir), local_files_only=True)
        self.processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)

        model_config = OpenVLAConfig.from_pretrained(str(checkpoint_dir), local_files_only=True)
        model_kwargs: dict[str, Any] = {
            "config": model_config,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation
        try:
            model = OpenVLAForActionPrediction.from_pretrained(str(checkpoint_dir), **model_kwargs)
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            model = OpenVLAForActionPrediction.from_pretrained(str(checkpoint_dir), **model_kwargs)

        stats_path = checkpoint_dir / "dataset_statistics.json"
        if stats_path.is_file():
            model.norm_stats = _load_json(stats_path)
        elif getattr(model.config, "norm_stats", None):
            model.norm_stats = model.config.norm_stats
        else:
            raise FileNotFoundError(f"OpenVLA-OFT dataset_statistics.json was not found in {checkpoint_dir}.")

        model.vision_backbone.set_num_images_in_input(self.config.num_images_in_input)
        model = model.to(device=self.device, dtype=self.dtype).eval()

        action_head = None
        if self.config.use_l1_regression:
            action_head = L1RegressionActionHead(input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=ACTION_DIM)
        elif self.config.use_diffusion:
            action_head = DiffusionActionHead(
                input_dim=model.llm_dim,
                hidden_dim=model.llm_dim,
                action_dim=ACTION_DIM,
                num_diffusion_steps_train=self.config.num_diffusion_steps_train,
            )
            action_head.noise_scheduler.set_timesteps(self.config.num_diffusion_steps_inference)
        if action_head is not None:
            action_head_path = _find_component_file(checkpoint_dir, self.repo_id, "action_head")
            action_head.load_state_dict(_load_component_state_dict(torch, action_head_path))
            action_head = action_head.to(device=self.device, dtype=self.dtype).eval()

        proprio_projector = None
        if self.config.use_proprio:
            proprio_projector = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=PROPRIO_DIM)
            proprio_path = _find_component_file(checkpoint_dir, self.repo_id, "proprio_projector")
            proprio_projector.load_state_dict(_load_component_state_dict(torch, proprio_path))
            proprio_projector = proprio_projector.to(device=self.device, dtype=self.dtype).eval()

        self.model = model
        self.action_head = action_head
        self.proprio_projector = proprio_projector

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not instruction:
            raise ValueError("OpenVLA-OFT requires a non-empty instruction prompt.")
        self.load()
        assert self.model is not None
        assert self.processor is not None

        import torch

        started = time.monotonic()
        norm_stats = self.model.norm_stats
        unnorm_key = _resolve_unnorm_key(norm_stats, self.config.unnorm_key, self.config.task_suite_name)
        images = [
            _prepare_image(item, center_crop=self.config.center_crop)
            for item in _select_images(image, observation, num_images=self.config.num_images_in_input)
        ]
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, images[0]).to(self.device, dtype=self.dtype)
        if len(images) > 1:
            wrist_inputs = [self.processor(prompt, item).to(self.device, dtype=self.dtype) for item in images[1:]]
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"]] + [item["pixel_values"] for item in wrist_inputs],
                dim=1,
            )

        proprio = None
        if self.config.use_proprio:
            state = _first_present_image(observation.get("state"), observation.get("proprio"))
            if state is None:
                raise ValueError("OpenVLA-OFT requires observation.state for LIBERO proprio conditioning.")
            proprio = _normalize_proprio(state, norm_stats[unnorm_key]["proprio"])

        with torch.inference_mode():
            actions, _ = self.model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=self.proprio_projector,
                action_head=self.action_head,
                use_film=False,
            )

        return {
            "status": "completed",
            "actions": _jsonable(actions),
            "model_id": "openvla-oft",
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_ref": self.repo_id or "",
            "runtime": "worldfoundry.openvla_oft.in_tree_runtime",
            "backend_quality": "checkpoint_backed",
            "official_prompt": prompt,
            "unnorm_key": unnorm_key,
            "num_images_in_input": self.config.num_images_in_input,
            "use_proprio": self.config.use_proprio,
            "torch_dtype": self.dtype_name,
            "device": self.device,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
