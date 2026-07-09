"""
This module provides utilities and a runtime class for configuring and interacting with the
StarVLA (Vision-Language-Action) model. It supports both a plan mode for generating runtime
configuration plans and checkpoint-backed inference using the in-tree StarVLA source subset.

It handles checkpoint selection, base VLM resolution, and marshaling runtime
parameters into a structured plan or executing real model predictions.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core import jsonable
from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path


RUNTIME_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StarVLARuntimeConfig:
    """Configuration settings for the StarVLA runtime.

    Attributes:
        checkpoint_dir: Path to the StarVLA policy checkpoint directory.
        base_vlm: Identifier or path for the base Vision-Language Model (VLM) (e.g., Qwen-VL).
        action_model_type: Type of the action model (e.g., 'transformer', 'diffusion').
        action_dim: Dimension of the action space.
        action_horizon: Number of future actions predicted.
        device: Computational device to use (e.g., 'cuda:0', 'cpu').
        track: Capability track for the model (e.g., 'WM4A', 'VLA').
        source_repo_dir: Optional path to the in-tree StarVLA source directory.
        attn_implementation: Attention implementation to use (e.g., 'flash_attention_2').
        enable_official_runtime: Flag to enable loading the StarVLA source package for inference.
    """

    checkpoint_dir: Path
    base_vlm: str
    action_model_type: str
    action_dim: int
    action_horizon: int
    device: str
    track: str
    source_repo_dir: Path | None
    attn_implementation: str
    enable_official_runtime: bool


def select_starvla_checkpoint(
    *,
    checkpoint_dir: str | Path | None,
    checkpoints: tuple[Mapping[str, Any], ...],
    variant_id: str | None = None,
    track: str | None = None,
) -> Path:
    """Select the StarVLA checkpoint directory.

    Args:
        checkpoint_dir: Explicit checkpoint override supplied by the caller.
        checkpoints: Runtime-profile checkpoint records used when no override exists.
        variant_id: Optional model-zoo variant id, such as a WM4A variant.
        track: Optional capability track used to infer the default checkpoint.
    Returns:
        The resolved Path to the selected StarVLA checkpoint directory.
    Raises:
        ValueError: If no suitable checkpoint can be determined.
    """

    candidate = checkpoint_dir
    # Normalize selector string for case-insensitive matching
    selector = f"{variant_id or ''} {track or ''}".lower()
    if candidate is None and checkpoints:
        # Prioritize WM4A/WAN2D2/World Action Model related checkpoints
        if "wm4a" in selector or "wan2d2" in selector or "world_action" in selector or "wam" in selector:
            candidate = _select_checkpoint_by_needles(checkpoints, ("wm4a", "wan2d2", "world_action"))
        # Fallback to Qwen/VLA related policy checkpoints
        elif "qwen" in selector or "vla" in selector:
            candidate = _select_checkpoint_by_needles(checkpoints, ("qwen", "policy"))
        # If no specific match, default to the local directory of the first available checkpoint
        candidate = candidate or checkpoints[0].get("local_dir")
    if candidate is None:
        raise ValueError("StarVLA requires a checkpoint_dir or profile checkpoint.")
    return _expand_path(candidate).resolve()


def _select_checkpoint_by_needles(checkpoints: tuple[Mapping[str, Any], ...], needles: tuple[str, ...]) -> str | None:
    """Helper to select a checkpoint from a list based on keywords (needles).

    Args:
        checkpoints: A tuple of checkpoint records, where each record is a dictionary.
        needles: A tuple of strings (keywords) to search for in checkpoint metadata.
    Returns:
        The 'local_dir' of the first matching checkpoint, or None if no match is found.
    """
    for item in checkpoints:
        # Concatenate relevant string values from the checkpoint item into a single blob for searching
        blob = " ".join(
            str(item.get(key) or "")
            for key in ("repo_id", "local_dir", "role", "variant_id", "id")
        ).lower()
        # Check if any of the specified needles are present in the blob
        if any(needle in blob for needle in needles):
            local_dir = item.get("local_dir")
            return str(local_dir) if local_dir else None
    return None


def select_starvla_base_vlm(
    *,
    base_vlm: str | Path | None,
    checkpoints: tuple[Mapping[str, Any], ...],
) -> str:
    """Select the Qwen base VLM path or repo id used by the StarVLA policy checkpoint.

    Args:
        base_vlm: Explicit base VLM override supplied by the caller.
        checkpoints: Runtime-profile checkpoint records to infer the base VLM from.
    Returns:
        The resolved base VLM identifier or path.
    Raises:
        ValueError: If no base VLM can be determined.
    """
    # Attempt to select a base VLM from the checkpoints, looking for "base" or "vlm" roles.
    candidate = base_vlm or _select_checkpoint_by_needles(checkpoints, ("base", "vlm"))
    # If no specific base VLM found, try to find a Qwen-VL related checkpoint.
    candidate = candidate or _select_checkpoint_by_needles(checkpoints, ("qwen3", "vl"))
    if candidate is None:
        raise ValueError("StarVLA requires base_vlm in runtime YAML or a base VLM checkpoint record.")
    return _path_or_model_id(candidate)


def _checkpoint_weight_file(checkpoint_dir: Path) -> Path:
    """Determines the primary weight file within a StarVLA checkpoint directory.

    Args:
        checkpoint_dir: The root directory of the StarVLA checkpoint.
    Returns:
        The path to the primary weight file (usually the last `.pt` file in `checkpoints/`).
    """
    checkpoint_root = checkpoint_dir / "checkpoints"
    # Find all .pt files and sort them to get the latest/final one
    candidates = sorted(checkpoint_root.glob("*.pt"))
    # Return the last candidate if found, otherwise default to a known filename
    return candidates[-1] if candidates else checkpoint_root / "steps_50000_pytorch_model.pt"


def _repository_root() -> Path:
    """Returns the root directory of the current project."""
    return project_root()


def _default_source_repo_dir() -> Path:
    """Returns the default path for the in-tree StarVLA source runtime.

    This path can be overridden by the WORLDFOUNDRY_STARVLA_REPO_ROOT environment variable.
    """
    env_override = os.getenv("WORLDFOUNDRY_STARVLA_REPO_ROOT")
    if env_override:
        return _expand_path(env_override)
    in_tree = RUNTIME_ROOT
    return in_tree


def _expand_path(value: str | Path) -> Path:
    """Resolves a given path, handling WorldFoundry specific path resolution and relative paths.

    Args:
        value: The path string or Path object to resolve.
    Returns:
        The expanded and resolved Path object.
    """
    path = resolve_worldfoundry_path(value)
    # If the path is not absolute, treat it as relative to the project repository root
    return path if path.is_absolute() else (_repository_root() / path)


def _path_or_model_id(value: str | Path) -> str:
    """Determines if a string represents a file path or a model identifier.

    Args:
        value: The string or Path object to evaluate.
    Returns:
        The resolved absolute path if it's a path, otherwise the original string.
    """
    text = str(value)
    # Check if the text resembles a file path (contains '$', starts with '/', '~', or '.')
    if "$" in text or text.startswith(("/", "~", ".")):
        return str(_expand_path(text).resolve())
    return text


def _load_checkpoint_config(checkpoint_dir: Path) -> dict[str, Any]:
    """Loads the `config.yaml` file from a given checkpoint directory.

    Args:
        checkpoint_dir: The path to the checkpoint directory.
    Returns:
        A dictionary containing the loaded configuration, or an empty dictionary if not found.
    """
    config_path = checkpoint_dir / "config.yaml"
    if not config_path.is_file():
        return {}

    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_starvla_plan_payload(
    *,
    config: StarVLARuntimeConfig,
    context: Mapping[str, Any],
    profile: Mapping[str, Any],
    runtime_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic StarVLA in-tree runtime plan.

    This function constructs a comprehensive JSON-serializable dictionary that
    describes how the StarVLA model should be loaded and executed, consolidating
    information from the runtime configuration, WorldFoundry context, profile,
    and user-supplied options.

    Args:
        config: Resolved checkpoint, model, and device settings.
        context: WorldFoundry runtime-profile context generated for this run.
        profile: Serialized runtime profile.
        runtime_options: User-supplied runtime and inference options.
    Returns:
        A dictionary representing the StarVLA runtime plan.
    """

    checkpoint_config = _load_checkpoint_config(config.checkpoint_dir)
    # Safely extract framework, action_model, and qwenvl configurations, ensuring they are dictionaries
    framework_config = checkpoint_config.get("framework") if isinstance(checkpoint_config.get("framework"), Mapping) else {}
    action_model = framework_config.get("action_model") if isinstance(framework_config.get("action_model"), Mapping) else {}
    qwenvl = framework_config.get("qwenvl") if isinstance(framework_config.get("qwenvl"), Mapping) else {}

    return {
        "schema_version": "worldfoundry-runtime-profile-starvla-plan",
        "profile": profile,
        "context": dict(context),
        "runtime": {
            "backend": "worldfoundry.starvla.in_tree_runtime.plan",
            "backend_quality": "in_tree_plan",
            "runtime_package": "worldfoundry.synthesis.action_generation.starvla.runtime",
            "runtime_root": str(RUNTIME_ROOT),
            "checkpoint_dir": str(config.checkpoint_dir),
            "checkpoint_file": str(_checkpoint_weight_file(config.checkpoint_dir)),
            "dataset_statistics": str(config.checkpoint_dir / "dataset_statistics.json"),
            "config_yaml": str(config.checkpoint_dir / "config.yaml"),
            # Prioritize base VLM from config, fallback to checkpoint config, then empty string
            "base_vlm": config.base_vlm or str(qwenvl.get("base_vlm") or ""),
            "source_repo_dir": "" if config.source_repo_dir is None else str(config.source_repo_dir),
            "attn_implementation": config.attn_implementation,
            "official_runtime_enabled": config.enable_official_runtime,
            # Prioritize framework name from checkpoint config, fallback to default
            "framework_name": str(framework_config.get("name") or "QwenOFT"),
            # Prioritize action model type from checkpoint config, fallback to runtime config
            "action_model_type": str(action_model.get("action_model_type") or config.action_model_type),
            # Prioritize action dimension from checkpoint config, fallback to runtime config
            "action_dim": int(action_model.get("action_dim") or config.action_dim),
            # Prioritize future action window size from checkpoint config, fallback to derived from action_horizon
            "future_action_window_size": int(action_model.get("future_action_window_size") or config.action_horizon - 1),
            # Prioritize past action window size from checkpoint config, fallback to default
            "past_action_window_size": int(action_model.get("past_action_window_size") or 0),
            "device": config.device,
            "track": config.track,
        },
        "inference": {
            # Prioritize instruction from runtime_options, then context, then empty string
            "instruction": str(runtime_options.get("instruction") or context.get("prompt") or ""),
            # Extract and sort observation keys if 'starvla_observation' is a mapping
            "observation_keys": sorted(runtime_options.get("starvla_observation", {}).keys())
            if isinstance(runtime_options.get("starvla_observation"), Mapping)
            else [],
            "action_space": jsonable(runtime_options.get("action_space")),
            "policy_controls": jsonable(runtime_options.get("policy_controls")),
        },
        "limitations": [
            "StarVLA architecture code is loaded from the in-tree source directory when enable_official_runtime is true.",
            "This runtime verifies checkpoint-backed predict_action only; LIBERO simulator scoring is a separate benchmark run.",
            "WM4A world-action checkpoints still require a separate WM4A-specific runtime.",
        ],
    }


class StarVLAPlanRuntime:
    """Manages the StarVLA runtime, supporting both plan generation and official model inference.

    This class provides an interface to configure StarVLA, generate a runtime plan
    (a JSON document describing the model setup), and optionally execute the
    `predict_action` method using a locally checked-out official StarVLA codebase.
    """

    def __init__(self, config: StarVLARuntimeConfig) -> None:
        """Create a StarVLA plan-only in-tree runtime.

        Args:
            config: Checkpoint, architecture, track, and device settings for planning.
        """

        self.config = config
        self._model: Any | None = None

    def write_plan(
        self,
        *,
        context: Mapping[str, Any],
        profile: Mapping[str, Any],
        runtime_options: Mapping[str, Any],
        plan_path: str | Path,
    ) -> dict[str, Any]:
        """Write the StarVLA runtime plan JSON.

        This method generates the runtime plan payload and saves it to the specified
        `plan_path` as a pretty-printed JSON file.

        Args:
            context: WorldFoundry runtime-profile context generated for this run.
            profile: Serialized runtime profile.
            runtime_options: User-supplied runtime and inference options.
            plan_path: Destination path for the runtime plan.
        Returns:
            The generated plan payload dictionary.
        """

        payload = build_starvla_plan_payload(
            config=self.config,
            context=context,
            profile=profile,
            runtime_options=runtime_options,
        )
        target = Path(plan_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write the payload as pretty-printed, non-ASCII-escaped, sorted JSON
        target.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    def predict_action(
        self,
        *,
        prompt: str,
        images: Any,
        output_path: str | Path,
        state: Any = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run checkpoint-backed StarVLA ``predict_action`` through the official source bridge.

        This method loads the official StarVLA model (if `enable_official_runtime` is True),
        prepares the inputs, performs inference to predict actions, and saves the
        action trace to a JSON file.

        Args:
            prompt: Natural-language task instruction.
            images: Input RGB observation image or image list. Can be PIL Image, NumPy array, or path.
            output_path: Destination action-trace JSON.
            state: Optional proprioceptive state passed to the official example.
            extra_metadata: Additional trace metadata.
        Returns:
            A dictionary summarizing the prediction result and artifact details.
        Raises:
            RuntimeError: If `enable_official_runtime` is False, preventing actual inference.
        """

        if not self.config.enable_official_runtime:
            raise RuntimeError(
                "StarVLA real inference requires enable_official_runtime=True and a local official "
                "StarVLA source checkout. Use plan_only=True to inspect the runtime plan."
            )
        model, torch = self._load_official_model()
        import numpy as np

        started = time.monotonic()
        # Coerce input images into the format expected by the official model (list of images)
        # and construct the example dictionary with prompt and optional state
        example: dict[str, Any] = {
            "image": _coerce_images(images),
            "lang": prompt,
        }
        if state is not None:
            example["state"] = state

        with torch.inference_mode():
            prediction = model.predict_action(examples=[example])

        # Convert raw predictions to a numpy array for consistent handling
        actions = np.asarray(prediction["normalized_actions"], dtype=np.float32)
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": "worldfoundry-starvla-action-trace",
            "status": "success",
            "model_id": "starvla",
            "backend": "official.starVLA.baseframework.predict_action",
            "backend_quality": "official_source_checkpoint_backed_gpu_validation",
            "artifact_kind": "action_trace",
            "instruction": prompt,
            "action_shape": list(actions.shape),
            "actions": actions.tolist(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "metadata": jsonable(
                {
                    "checkpoint_dir": self.config.checkpoint_dir,
                    "checkpoint_file": _checkpoint_weight_file(self.config.checkpoint_dir),
                    "base_vlm": self.config.base_vlm,
                    "source_repo_dir": self.config.source_repo_dir,
                    "track": self.config.track,
                    # Merge any additional metadata provided by the caller
                    **dict(extra_metadata or {}),
                }
            ),
        }
        # Save the action trace payload to the specified output path
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "status": "success",
            "model_id": "starvla",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "artifact_sha256": sha256(target.read_bytes()).hexdigest(),
            "backend": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "duration_seconds": payload["duration_seconds"],
        }

    def _load_official_model(self) -> tuple[Any, Any]:
        """Loads the official StarVLA model from the designated source repository.

        This method handles adding the source repository to `sys.path`, patching
        the `read_mode_config` function to inject runtime configuration (base VLM, attn_implementation),
        loading the model, and then restoring the original function.

        Returns:
            A tuple containing the loaded model instance and the torch module.
        Raises:
            FileNotFoundError: If the StarVLA source package is missing.
        """
        if self._model is not None:
            return self._model

        source_repo_dir = self.config.source_repo_dir or _default_source_repo_dir()
        source_repo_dir = source_repo_dir.expanduser().resolve()
        if not source_repo_dir.is_dir():
            raise FileNotFoundError(f"StarVLA in-tree source package is missing: {source_repo_dir}")
        source_text = str(source_repo_dir)
        # Add the StarVLA source directory to sys.path to allow importing its modules.
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

        import torch
        # Dynamically import the base_framework module from the added source path
        import starVLA.model.framework.base_framework as base_framework

        original_read_mode_config = base_framework.read_mode_config

        # Define a patched version of `read_mode_config` to inject runtime config
        def patched_read_mode_config(path: Any) -> tuple[dict[str, Any], Mapping[str, Any]]:
            """Patched version of base_framework.read_mode_config.

            Injects `base_vlm` and `attn_implementation` from the runtime configuration
            into the model's framework config before it's used for loading.
            """
            model_config, norm_stats = original_read_mode_config(path)
            framework = model_config.setdefault("framework", {})
            qwenvl = framework.setdefault("qwenvl", {})
            # Override base VLM and attention implementation with values from StarVLARuntimeConfig
            qwenvl["base_vlm"] = self.config.base_vlm
            qwenvl["attn_implementation"] = self.config.attn_implementation
            return model_config, norm_stats

        # Temporarily replace the original function with the patched version
        base_framework.read_mode_config = patched_read_mode_config
        try:
            # Load the model using the potentially patched configuration
            model = base_framework.baseframework.from_pretrained(str(_checkpoint_weight_file(self.config.checkpoint_dir)))
        finally:
            # Ensure the original function is restored after model loading
            base_framework.read_mode_config = original_read_mode_config

        model = model.to(self.config.device).eval()
        self._model = (model, torch)
        return self._model


def _coerce_images(images: Any) -> list[Any]:
    """Coerces various image inputs into a list of usable image objects.

    Args:
        images: An image, a list/tuple of images, or a dictionary mapping keys to images.
                Each image can be a PIL Image, NumPy array, or a path (string/Path).
    Returns:
        A list of PIL RGB Image objects.
    Raises:
        ValueError: If `images` is None.
    """
    if images is None:
        raise ValueError("StarVLA predict_action requires at least one RGB image.")
    if isinstance(images, Mapping):
        # If a dictionary, sort by keys to ensure deterministic order
        candidates = [images[key] for key in sorted(images)]
    elif isinstance(images, (list, tuple)):
        candidates = list(images)
    else:
        # If a single image, wrap it in a list
        candidates = [images]
    return [_coerce_image(item) for item in candidates]


def _coerce_image(image: Any) -> Any:
    """Coerces a single image input into a PIL RGB Image object.

    Handles PIL Images, file paths, and NumPy arrays, performing necessary
    conversions (e.g., channel reordering, type casting, color space conversion).

    Args:
        image: The image input, can be a PIL Image, NumPy array, or a path (string/Path).
    Returns:
        A PIL Image object in RGB mode.
    Raises:
        ValueError: For unsupported image formats or shapes.
    """
    from PIL import Image
    import numpy as np

    if isinstance(image, Image.Image):
        # Convert existing PIL Image to RGB mode
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        text = str(image)
        if text.startswith("memory://"):
            raise ValueError(f"StarVLA real runtime cannot load in-memory placeholder image: {text}")
        # Open image from path and convert to RGB
        return Image.open(Path(text).expanduser()).convert("RGB")

    # Assume numpy array if not PIL Image or path
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"StarVLA image arrays must be HWC or CHW rank-3, got shape {array.shape}")

    # Heuristic to detect CHW (Channel-Height-Width) format and convert to HWC (Height-Width-Channel)
    # If the first dimension is 1, 3, or 4 (potential channel count) and the last isn't, assume CHW.
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)

    # Convert to uint8 if not already
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    # Handle grayscale (1 channel) by repeating to 3 channels
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    # Handle RGBA (4 channels) by discarding alpha
    if array.shape[-1] == 4:
        array = array[..., :3]

    if array.shape[-1] != 3:
        raise ValueError(f"StarVLA image arrays must have 3 color channels, got shape {array.shape}")
    return Image.fromarray(array, mode="RGB")
