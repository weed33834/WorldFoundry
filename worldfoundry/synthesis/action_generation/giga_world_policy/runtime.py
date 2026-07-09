"""Utilities and runtime for in-tree GigaWorld-Policy inference.

This module provides functions for resolving paths to GigaWorld-Policy assets and
a class for loading and running the GigaWorld-Policy model to predict actions.
It integrates with the worldfoundry framework for path management and output formatting.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path


def _jsonable(value: Any) -> Any:
    """Recursively converts a value into a JSON-serializable format.

    Handles common data structures, Path objects, and objects with `tolist()` or `item()` methods.

    Args:
        value: The input value to convert.

    Returns:
        The JSON-serializable representation of the input value.
    """
    if isinstance(value, Mapping):
        # Recursively process items in mappings (dictionaries)
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # Recursively process items in sequences (lists, tuples)
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        # Convert Path objects to string representations
        return str(value)
    if hasattr(value, "tolist"):
        # Handle objects like numpy arrays that can be converted to lists
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        # Handle objects like numpy scalars that can be converted to single items
        return _jsonable(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        # Base cases for primitive JSON types
        return value
    # Default conversion for unhandled types: cast to string
    return str(value)


def _worldfoundry_repository_root() -> Path:
    """Returns the root path of the worldfoundry repository."""
    return project_root()


def _expand_path_value(value: Any) -> Path | None:
    """Expands a given path-like value into an absolute Path object.

    Resolves paths relative to the worldfoundry repository root if they are not absolute.

    Args:
        value: The path value, which can be None, an empty string, or a path string/object.

    Returns:
        An absolute and resolved Path object, or None if the input value is None or empty.
    """
    if value in (None, ""):
        return None
    repo_root = _worldfoundry_repository_root()
    path = resolve_worldfoundry_path(value)
    if not path.is_absolute():
        # If the path is relative, make it absolute by joining with the repository root
        path = repo_root / path
    return path.resolve()


def _first_existing_path(*values: Any, require_existing: bool = True) -> Path:
    """Finds the first existing path among a list of candidates.

    Args:
        *values: Variable arguments, each representing a potential path.
        require_existing: If True, raises FileNotFoundError if no path exists.
                          If False and no path exists, returns a temporary non-existing path.

    Returns:
        The first `Path` object that exists (if `require_existing` is True)
        or the first expanded path if `require_existing` is False.

    Raises:
        FileNotFoundError: If `require_existing` is True and no provided path exists.
    """
    candidates: list[Path] = []
    for value in values:
        path = _expand_path_value(value)
        if path is None:
            continue
        candidates.append(path)
        if not require_existing or path.exists():
            return path
    if candidates:
        joined = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No existing GigaWorld-Policy path was provided. Checked: {joined}")
    if not require_existing:
        # If no path is required to exist, return a dummy path for non-existing assets
        return (_worldfoundry_repository_root() / "tmp" / "giga_world_policy_missing_asset").resolve()
    raise FileNotFoundError("No GigaWorld-Policy path was provided.")


def _role_path(roles: Mapping[str, str], *needles: str) -> str | None:
    """Searches a dictionary of role-to-path mappings for a role containing specific substrings.

    Args:
        roles: A mapping where keys are role strings and values are path strings.
        *needles: Variable arguments, each a substring that must be present in the role name.

    Returns:
        The path string associated with the first matching role, or None if no such role is found.
    """
    for role, path in roles.items():
        # Check if all specified 'needles' are present in the current role string
        if all(needle in role for needle in needles):
            return path
    return None


def _parse_bool_list(value: str | Sequence[bool]) -> list[bool]:
    """Parses a string or sequence into a list of boolean values.

    Handles comma-separated string representations like "1,true,0,no".

    Args:
        value: The input string or sequence of booleans.

    Returns:
        A list of boolean values.
    """
    if isinstance(value, str):
        # Split the string by comma, strip whitespace, convert to lowercase
        items = [item.strip().lower() for item in value.split(",") if item.strip()]
        # Determine boolean value based on common true-like strings
        return [item in {"1", "true", "yes"} for item in items]
    # For sequences, convert each item directly to a boolean
    return [bool(item) for item in value]


def _as_tensor_image(value: Any) -> Any:
    """Converts a value to a PyTorch tensor, assuming it represents an image.

    Args:
        value: The input value, potentially an image.

    Returns:
        A PyTorch tensor representation of the image.
    """
    import torch

    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _select_image(images: Any, key: str) -> Any:
    """Selects an image from a mapping using a specified key.

    Args:
        images: A mapping (e.g., dictionary) containing image data.
        key: The key to look up in the images mapping.

    Returns:
        The image data associated with the given key.

    Raises:
        ValueError: If `images` is not a mapping or the key is not found.
    """
    if isinstance(images, Mapping) and key in images:
        return images[key]
    raise ValueError(f"GigaWorld-Policy requires image key {key!r}.")


def select_giga_world_policy_paths(
    *,
    model_id_path: Any = None,
    transformer_path: Any = None,
    stats_path: Any = None,
    t5_embedding_pkl: Any = None,
    checkpoints: Sequence[Mapping[str, Any]] = (),
    require_existing: bool = True,
) -> dict[str, Path]:
    """Resolves required GigaWorld-Policy runtime paths from inputs and profile checkpoints.

    This function attempts to find the correct `Path` objects for various model components
    by checking direct arguments first, then looking into a sequence of checkpoint records.

    Args:
        model_id_path: WAN model directory or local Hugging Face snapshot path.
        transformer_path: Trained world-action transformer directory path.
        stats_path: Normalization statistics JSON path.
        t5_embedding_pkl: Precomputed T5 embedding tensor path.
        checkpoints: A sequence of checkpoint records, where each record is a mapping
                     that can contain 'role', 'name', 'local_dir', or 'path' to identify
                     and locate model components.
        require_existing: Whether every resolved path must already exist on the filesystem.

    Returns:
        A dictionary mapping component names (e.g., "model_id_path") to their resolved
        absolute `Path` objects.
    """
    roles: dict[str, str] = {}
    # Extract roles and local directories from checkpoint records
    for item in checkpoints:
        role = str(item.get("role") or item.get("name") or "").lower()
        local_dir = str(item.get("local_dir") or item.get("path") or "")
        if role and local_dir:
            roles[role] = local_dir
    return {
        "model_id_path": _first_existing_path(
            model_id_path,
            _role_path(roles, "wan"),
            _role_path(roles, "model"),
            require_existing=require_existing,
        ),
        "transformer_path": _first_existing_path(
            transformer_path,
            _role_path(roles, "transformer"),
            require_existing=require_existing,
        ),
        "stats_path": _first_existing_path(
            stats_path,
            _role_path(roles, "norm", "stats"),
            _role_path(roles, "stats"),
            require_existing=require_existing,
        ),
        "t5_embedding_pkl": _first_existing_path(
            t5_embedding_pkl,
            _role_path(roles, "t5", "embedding"),
            _role_path(roles, "t5"),
            require_existing=require_existing,
        ),
    }


@dataclass(frozen=True)
class GigaWorldPolicyRuntimeConfig:
    """Runtime settings for in-tree GigaWorld-Policy inference.

    This dataclass encapsulates all configuration parameters required to initialize
    and run the GigaWorld-Policy model, including paths to assets, hardware settings,
    and various model hyper-parameters.
    """

    model_id_path: Path
    transformer_path: Path
    stats_path: Path
    t5_embedding_pkl: Path
    device: str
    dtype: str
    t5_len: int
    dst_width: int
    dst_height: int
    action_chunk: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    norm_mode: str
    crop_mode: str
    state_dim: int
    action_dim: int
    delta_mask: str
    image_keys: tuple[str, ...]


class GigaWorldPolicyRuntime:
    """Lazy in-tree GigaWorld-Policy runtime backed by vendored official code.

    This class manages the loading and execution of the GigaWorld-Policy model.
    It loads PyTorch models and pipelines only when `load()` is explicitly called
    or implicitly by `predict_action()`, optimizing for scenarios where the policy
    might not always be used immediately after instantiation.
    """

    def __init__(self, config: GigaWorldPolicyRuntimeConfig) -> None:
        """Initializes the GigaWorldPolicyRuntime with the given configuration.

        Args:
            config: An instance of `GigaWorldPolicyRuntimeConfig` containing
                    all necessary parameters for model loading and inference.
        """
        self.config = config
        self.policy: Any | None = None

    def load(self) -> None:
        """Loads the Wan VAE, world-action transformer, and WAPipeline.

        This method imports the necessary PyTorch and custom model components,
        initializes the VAE, transformer, and the full WAPipeline based on
        the provided configuration, and prepares them for inference.
        It is idempotent: calling it multiple times will only load the models once.
        """
        if self.policy is not None:
            return

        # Install necessary aliases for the vendored `giga_world_policy_runtime` code
        from worldfoundry.synthesis.action_generation.giga_world_policy.giga_world_policy_runtime import install_aliases

        install_aliases()

        # Import PyTorch and GigaWorld-Policy specific components
        import torch
        from diffusers.models import AutoencoderKLWan

        from worldfoundry.synthesis.action_generation.giga_world_policy.giga_world_policy_runtime.world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
        from worldfoundry.synthesis.action_generation.giga_world_policy.giga_world_policy_runtime.world_action_model.pipeline.utils import (
            add_state_to_action,
            build_ref_image,
            denormalize_action,
            extract_normalization_tensors,
            load_stats,
            load_t5_embedding_from_pkl,
            normalize_state,
        )
        from worldfoundry.synthesis.action_generation.giga_world_policy.giga_world_policy_runtime.world_action_model.pipeline.wa_pipeline import WAPipeline

        # Setup device and data type for PyTorch operations
        device = torch.device(self.config.device)
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map[self.config.dtype]

        # Load T5 embedding, normalization statistics, and delta mask
        t5_embedding = load_t5_embedding_from_pkl(str(self.config.t5_embedding_pkl), target_len=self.config.t5_len).to(
            device=device,
            dtype=torch.float32,
        )
        stats = load_stats(str(self.config.stats_path))
        norm = extract_normalization_tensors(stats, device=device, state_dim=self.config.state_dim, action_dim=self.config.action_dim)
        delta_mask = torch.tensor(_parse_bool_list(self.config.delta_mask), device=device, dtype=torch.bool)
        if delta_mask.numel() != self.config.action_dim:
            raise ValueError(f"delta_mask length {delta_mask.numel()} does not match action_dim {self.config.action_dim}.")

        # Load the VAE and transformer models, then initialize the WAPipeline
        vae = AutoencoderKLWan.from_pretrained(str(self.config.model_id_path), subfolder="vae", torch_dtype=torch.bfloat16)
        transformer = CasualWorldActionTransformer.from_pretrained(str(self.config.transformer_path)).to(dtype)
        pipe = WAPipeline.from_pretrained(str(self.config.model_id_path), vae=vae, transformer=transformer, torch_dtype=dtype)
        pipe.to(device)

        # Store all loaded components and utility functions in a dictionary for easy access
        self.policy = {
            "pipe": pipe,
            "torch": torch,
            "device": device,
            "dtype": dtype,
            "t5_embedding": t5_embedding,
            "norm": norm,
            "delta_mask": delta_mask,
            "build_ref_image": build_ref_image,
            "normalize_state": normalize_state,
            "denormalize_action": denormalize_action,
            "add_state_to_action": add_state_to_action,
        }

    def predict_action(
        self,
        *,
        prompt: str,
        images: Any,
        state: Any,
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Runs one GigaWorld-Policy action prediction and writes an action trace to JSON.

        This method performs the full inference pipeline:
        1. Ensures models are loaded (`self.load()` is called implicitly if needed).
        2. Preprocesses input images and state.
        3. Invokes the `WAPipeline` for action generation.
        4. Post-processes the generated action (denormalization, add state).
        5. Saves the action trace along with metadata to a specified JSON file.

        Args:
            prompt: Task instruction string, included in the output metadata.
            images: A mapping of camera names to image data (e.g., PyTorch tensors, numpy arrays).
            state: Robot state vector (e.g., list, numpy array).
            output_path: Destination path for the output JSON action trace file.
            extra_metadata: Optional additional metadata to include in the output JSON.

        Returns:
            A dictionary summarizing the prediction result, including artifact path and SHA256.
        """
        self.load()  # Ensure the policy is loaded before prediction
        assert self.policy is not None

        torch = self.policy["torch"]
        started = time.monotonic()
        device = self.policy["device"]
        dtype = self.policy["dtype"]

        # Prepare state tensor, ensuring correct shape, device, and dtype
        state_tensor = torch.as_tensor(state)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        state_tensor = state_tensor.to(device=device, dtype=torch.float32)

        # Process input images using configured keys
        image_tensors = {
            key: _as_tensor_image(_select_image(images, key))
            for key in self.config.image_keys
        }
        # Build the reference image for the pipeline from the processed images
        ref_image = self.policy["build_ref_image"](
            images=image_tensors,
            dst_size=(self.config.dst_width, self.config.dst_height),
            crop_mode=self.config.crop_mode,
            image_keys=self.config.image_keys,
        )

        # Normalize the state tensor before passing it to the pipeline
        norm_state = self.policy["normalize_state"](state_tensor, self.policy["norm"], mode=self.config.norm_mode).to(device=device, dtype=dtype)

        # Run the GigaWorld-Policy pipeline to predict actions
        _, action = self.policy["pipe"](
            height=self.config.dst_height,
            width=self.config.dst_width,
            action_chunk=self.config.action_chunk,
            state=norm_state,
            num_frames=self.config.num_frames,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            image=ref_image,
            action_only=True,  # Request only the action output
            return_dict=False,
            prompt_embeds=self.policy["t5_embedding"].unsqueeze(0).to(device=device, dtype=torch.float32),
        )

        # Post-process the predicted action: denormalize and add state delta
        action = self.policy["denormalize_action"](action[0].float(), self.policy["norm"], mode=self.config.norm_mode)
        action = self.policy["add_state_to_action"](action, state_tensor[0].float(), action_chunk=self.config.action_chunk, mask=self.policy["delta_mask"])

        # Prepare output path and directory
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        # Construct the JSON payload for the action trace
        payload = {
            "schema_version": "worldfoundry-giga-world-policy-action-trace",
            "status": "success",
            "model_id": "giga-world-policy",
            "backend": "worldfoundry.giga_world_policy.in_tree_runtime",
            "backend_quality": "official_in_tree",
            "artifact_kind": "action_trace",
            "instruction": prompt,
            "action_shape": list(action.shape),
            "actions": _jsonable(action.detach().cpu()),  # Convert action tensor to JSON-serializable list
            "duration_seconds": round(time.monotonic() - started, 3),
            "metadata": _jsonable(dict(extra_metadata or {})),
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Compute SHA256 hash of the written artifact
        artifact_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()

        # Return a summary dictionary of the prediction result
        return {
            "status": "success",
            "model_id": "giga-world-policy",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "artifact_sha256": artifact_sha256,
            "backend": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "duration_seconds": payload["duration_seconds"],
        }