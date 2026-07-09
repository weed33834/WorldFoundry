"""
Utility module for managing and interacting with the OpenPI (Open-Vocabulary Policy Integration) runtime.

This module provides functionalities for selecting OpenPI checkpoints, loading image data
for observations, resolving observations for policy inference, and running a vendored
OpenPI policy to predict actions. It integrates with `worldfoundry` path utilities and
handles necessary environment setups like installing OpenPI aliases.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path
from worldfoundry.synthesis.action_generation.openpi.openpi_runtime import install_aliases


def _jsonable(value: Any) -> Any:
    """
    Recursively converts a value into a JSON-serializable format.

    Handles common Python types, Path objects, numpy arrays (via `tolist` or `item`),
    and falls back to string conversion for other types.

    Args:
        value: The value to convert.

    Returns:
        A JSON-serializable representation of the value.
    """
    if isinstance(value, Mapping):
        # Recursively process dictionary keys and values.
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # Recursively process list/tuple items.
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        # Convert Path objects to string representation.
        return str(value)
    if hasattr(value, "tolist"):
        # Handle numpy arrays or similar objects by converting to a Python list.
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        # Handle numpy scalars or similar objects by converting to a Python scalar.
        return _jsonable(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        # Base cases: already JSON-serializable types.
        return value
    # Fallback for any other type, converting to string.
    return str(value)


def select_openpi_checkpoint(
    *,
    checkpoint_dir: str | Path | None,
    checkpoints: Sequence[Mapping[str, Any]],
    config_name: str,
    require_exists: bool = True,
) -> Path:
    """
    Selects the most suitable local OpenPI checkpoint path based on provided candidates and configuration.

    The selection process prioritizes explicitly provided directories, then candidates
    matching the `config_name` in their metadata, and finally any available checkpoint.

    Args:
        checkpoint_dir: An optional explicit directory path for the checkpoint.
        checkpoints: A sequence of checkpoint candidate mappings, each potentially containing
            'local_dir', 'role', or 'path' keys.
        config_name: The name of the policy configuration, used for filtering preferred candidates.
        require_exists: If True, only returns a path to an existing directory. If False,
            may return a non-existent path if no existing ones are found.

    Returns:
        The resolved Path to the selected OpenPI checkpoint directory.

    Raises:
        FileNotFoundError: If `require_exists` is True and no existing checkpoint is found.
    """
    if checkpoint_dir:
        explicit_path = _checkpoint_path({"local_dir": str(checkpoint_dir)})
        if explicit_path is not None and (explicit_path.exists() or not require_exists):
            return explicit_path

    candidates: list[Mapping[str, Any]] = []
    # Extend with other provided checkpoint candidates.
    candidates.extend(dict(item) for item in checkpoints)

    # First, try to find checkpoints whose metadata contains tokens from the config_name.
    preferred = _preferred_checkpoint_candidates(candidates, config_name)
    for item in preferred:
        path = _checkpoint_path(item)
        if path is not None and path.exists():
            return path

    # If no preferred existing checkpoint, iterate through all candidates.
    for item in candidates:
        path = _checkpoint_path(item)
        if path is None:
            continue
        if path.exists():
            return path
    # If no existing checkpoint is found and `require_exists` is False,
    # try to return the path of any candidate (preferred first).
    if not require_exists:
        for item in (*preferred, *candidates):
            path = _checkpoint_path(item)
            if path is not None:
                return path
    raise FileNotFoundError("No local OpenPI checkpoint was found.")


def _checkpoint_path(item: Mapping[str, Any]) -> Path | None:
    """
    Resolves a checkpoint item mapping into an absolute Path object.

    Looks for 'local_dir' within the item to determine the checkpoint path.
    Paths are resolved relative to the worldfoundry project root if not absolute.

    Args:
        item: A dictionary representing a checkpoint candidate, expected to have a 'local_dir' key.

    Returns:
        An absolute Path object if 'local_dir' is present and valid, otherwise None.
    """
    local_dir = str(item.get("local_dir") or "")
    if not local_dir:
        return None
    # Resolve the path using worldfoundry specific path resolution logic.
    path = resolve_worldfoundry_path(local_dir)
    # If the path is not absolute, assume it's relative to the project root.
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def _preferred_checkpoint_candidates(
    candidates: Sequence[Mapping[str, Any]],
    config_name: str,
) -> tuple[Mapping[str, Any], ...]:
    """
    Filters a list of checkpoint candidates to identify those preferred based on the configuration name.

    A candidate is "preferred" if keywords from the `config_name` are found in its
    'role', 'local_dir', or 'path' metadata fields (case-insensitive, underscore-normalized).

    Args:
        candidates: A sequence of checkpoint candidate mappings.
        config_name: The name of the policy configuration (e.g., "libero-100-test").

    Returns:
        A tuple of preferred checkpoint candidate mappings. If no candidates fully match,
        it returns candidates that partially match any token.
    """
    # Normalize config_name into a tuple of significant tokens.
    tokens = tuple(part for part in config_name.lower().replace("-", "_").split("_") if part)
    if not tokens:
        return ()

    preferred: list[Mapping[str, Any]] = []
    for item in candidates:
        # Create a haystack string from relevant metadata fields for matching.
        haystack = " ".join(str(item.get(key) or "").lower().replace("-", "_") for key in ("role", "local_dir", "path"))
        # A candidate is preferred if ALL tokens from the config_name are found in the haystack.
        if all(token in haystack for token in tokens):
            preferred.append(item)

    if preferred:
        return tuple(preferred)

    # If no candidates fully match, return candidates where ANY token from the config_name is found.
    return tuple(
        item
        for item in candidates
        if any(token in " ".join(str(item.get(key) or "").lower().replace("-", "_") for key in ("role", "local_dir", "path")) for token in tokens)
    )


def _load_rgb_array(image: Any) -> Any:
    """
    Loads and preprocesses an image into a 224x224 RGB NumPy array.

    Supports various input types: None, sequences of images (takes the first),
    PIL Image objects, file paths (str/Path), and NumPy arrays.
    It converts images to RGB, resizes them to 224x224, and ensures a uint8 dtype.

    Args:
        image: The input image, which can be None, a sequence of images, a PIL Image,
               a file path (str or Path), or a NumPy array.

    Returns:
        A NumPy array of shape (224, 224, 3) with dtype uint8 representing the
        processed RGB image, or None if the input was None or an empty sequence.

    Raises:
        FileNotFoundError: If `image` is a path but the file does not exist.
        ValueError: If `image` is a NumPy array with an unsupported number of dimensions.
    """
    if image is None:
        return None
    # If a sequence (e.g., list of images), take the first item.
    if isinstance(image, Sequence) and not isinstance(image, (bytes, bytearray, str)):
        if not image:
            return None
        return _load_rgb_array(image[0])

    # Defer import of heavy libraries until they are needed.
    import numpy as np
    from PIL import Image

    if isinstance(image, Image.Image):
        # Convert PIL Image to RGB, resize, and convert to NumPy array.
        rgb = image.convert("RGB").resize((224, 224))
        return np.asarray(rgb, dtype=np.uint8)
    if isinstance(image, (str, Path)):
        # Load image from file path, convert to RGB, resize, and convert to NumPy array.
        image_path = Path(image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"OpenPI image path does not exist: {image_path}")
        return np.asarray(Image.open(image_path).convert("RGB").resize((224, 224)), dtype=np.uint8)

    # Assume input is a NumPy array or similar array-like object.
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"OpenPI image array must be HxWxC or CxHxW, got shape {array.shape}.")
    # Transpose array if it's in CxHxW format (e.g., (3, 224, 224)) to HxWxC.
    if array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    # Normalize float arrays to 0-255 range.
    if array.dtype.kind == "f":
        array = np.clip(array, 0.0, 1.0) * 255.0
    # Convert processed array back to PIL Image, then to RGB, resize, and finally to uint8 NumPy array.
    return np.asarray(Image.fromarray(array.astype(np.uint8)).convert("RGB").resize((224, 224)), dtype=np.uint8)


def _resolve_observation(prompt: str, image: Any, openpi_observation: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    Constructs a full OpenPI observation dictionary from a prompt, image, and optional
    additional OpenPI observation data.

    This function initializes a standard Libero policy example observation,
    sets the prompt, integrates proprioceptive data if provided, and
    embeds processed image data.

    Args:
        prompt: The textual instruction for the policy.
        image: The raw image input (can be path, PIL Image, array, etc.).
        openpi_observation: An optional dictionary containing additional OpenPI
            specific observation data, e.g., 'proprio'.

    Returns:
        A dictionary representing the full observation ready for an OpenPI policy.
    """
    # Ensure OpenPI aliases are installed for module imports.
    install_aliases()

    # Defer import of numpy.
    import numpy as np

    # Dynamically import the OpenPI Libero policy module.
    libero_policy = importlib.import_module("openpi.policies.libero_policy")

    # Initialize a base observation structure from the Libero policy.
    observation = libero_policy.make_libero_example()
    # Set the textual prompt for the observation.
    observation["prompt"] = prompt or "do something"

    # Integrate proprioceptive data if provided in the openpi_observation.
    if openpi_observation:
        proprio = openpi_observation.get("proprio")
        if proprio is not None:
            observation["observation/state"] = np.asarray(proprio)

    # Process and integrate the image data.
    rgb_array = _load_rgb_array(image)
    if rgb_array is not None:
        observation["observation/image"] = rgb_array
        observation["observation/wrist_image"] = rgb_array  # Duplicate for wrist camera as per policy expectation
    return observation


@dataclass(frozen=True)
class OpenPIRuntimeConfig:
    """
    Configuration settings for initializing and running an OpenPI policy.

    Attributes:
        checkpoint_dir: The path to the directory containing the policy checkpoint files.
        config_name: The name of the OpenPI policy configuration (e.g., "libero-100-test").
        pytorch_device: The PyTorch device to use for inference (e.g., "cuda:0", "cpu").
        seed: The random seed to use for reproducible policy inference.
    """

    checkpoint_dir: Path
    config_name: str
    pytorch_device: str | None
    seed: int


class OpenPIRuntime:
    """
    A lazy-loading runtime for performing inference with an in-tree OpenPI policy.

    This class encapsulates the logic for loading a specific OpenPI policy from a
    checkpoint and then using it to predict actions based on observations.
    Policy loading is deferred until the first call to `load()` or `predict_action()`.
    """

    def __init__(self, config: OpenPIRuntimeConfig) -> None:
        """
        Initializes the OpenPIRuntime with the given configuration.

        The policy itself is not loaded during initialization but is loaded on demand.

        Args:
            config: An instance of OpenPIRuntimeConfig specifying policy details.
        """
        self.config = config
        self.policy: Any | None = None

    def load(self) -> None:
        """
        Loads the configured OpenPI policy from its local checkpoint directory.

        This method is idempotent; if the policy is already loaded, it does nothing.
        It imports necessary OpenPI modules dynamically and creates the policy instance.

        Raises:
            FileNotFoundError: If the specified checkpoint directory does not exist.
        """
        if self.policy is not None:
            return
        # Ensure OpenPI aliases are installed for module imports.
        install_aliases()

        # Dynamically import OpenPI configuration and runtime support modules.
        _policy_config = importlib.import_module("openpi.policies.policy_config")
        _config = importlib.import_module("openpi.runtime_support.config")

        # Resolve and validate the checkpoint directory path.
        checkpoint = self.config.checkpoint_dir.expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"OpenPI checkpoint directory does not exist: {checkpoint}")

        # Get the training configuration for the specified policy.
        train_config = _config.get_config(self.config.config_name)
        # Create the trained policy instance using the configuration and checkpoint.
        self.policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint,
            pytorch_device=self.config.pytorch_device,
        )

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        output_path: str | Path,
        openpi_observation: Mapping[str, Any] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Runs one OpenPI policy inference step and saves the action trace to a file.

        This method first ensures the policy is loaded, then constructs the observation,
        performs inference, extracts the predicted actions, and saves them along with
        metadata into a JSON file following a WorldFoundry action trace schema.

        Args:
            instruction: The textual instruction for the policy.
            image: The image input for the policy (e.g., path, PIL Image, NumPy array).
            output_path: The file path where the action trace JSON will be saved.
            openpi_observation: Optional additional OpenPI observation data (e.g., proprioception).
            extra_metadata: Optional dictionary of extra metadata to include in the trace file.

        Returns:
            A dictionary containing metadata about the generated action trace artifact.
        """
        # Load the policy if it hasn't been loaded yet.
        self.load()
        # Assert that the policy is now available for inference.
        assert self.policy is not None

        # Defer import of numpy.
        import numpy as np

        # Set the random seed for reproducibility of policy inference.
        np.random.seed(self.config.seed)
        started = time.monotonic()
        # Resolve the full observation dictionary for the policy.
        observation = _resolve_observation(instruction, image, openpi_observation)
        # Perform inference using the loaded policy.
        result = self.policy.infer(observation)
        # Extract actions from the inference result.
        actions = np.asarray(result["actions"])

        # Resolve the output path and ensure its parent directory exists.
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        # Construct the payload for the action trace JSON file.
        payload = {
            "schema_version": "worldfoundry-openpi-action-trace",
            "status": "success",
            "model_id": "openpi",
            "backend": "worldfoundry.openpi.in_tree_runtime.create_trained_policy_infer",
            "backend_quality": "official_demo",
            "artifact_kind": "action_trace",
            "checkpoint_dir": str(self.config.checkpoint_dir),
            "config_name": self.config.config_name,
            "instruction": instruction,
            "seed": self.config.seed,
            "action_shape": list(actions.shape),
            "action": _jsonable(actions[0]),  # Store the first action as a single item for convenience.
            "actions": _jsonable(actions),  # Store all actions.
            "result_keys": sorted(str(key) for key in result.keys()),
            "duration_seconds": round(time.monotonic() - started, 3),
            "metadata": _jsonable(dict(extra_metadata or {})),
        }
        # Write the JSON payload to the target file.
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Calculate SHA256 hash of the created artifact.
        artifact_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        # Return a summary dictionary of the created artifact.
        return {
            "status": "success",
            "model_id": "openpi",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "artifact_sha256": artifact_sha256,
            "backend": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "duration_seconds": payload["duration_seconds"],
        }
