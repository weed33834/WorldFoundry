"""This module provides an in-tree runtime for executing Diffusion Policy models.

It includes utilities for configuring the runtime, selecting checkpoints,
preprocessing observations, and performing action prediction, with the capability
to output WorldFoundry-compatible action traces.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core import jsonable
from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path


# Defines the root directory where the in-tree Diffusion Policy runtime package is located.
# This path is used to dynamically import the policy's modules.
RUNTIME_ROOT = (
    Path(__file__).resolve().parent
    / "diffusion_policy_runtime"
)


@dataclass(frozen=True)
class DiffusionPolicyRuntimeConfig:
    """Configuration class for the Diffusion Policy in-tree runtime.

    Attributes:
        checkpoint_path: Path to the Diffusion Policy model checkpoint file.
        device: The PyTorch device (e.g., "cuda" or "cpu") on which to run the policy.
        output_dir: Optional directory to store any runtime-generated files (e.g., model logs).
    """
    checkpoint_path: Path
    device: str = "cuda"
    output_dir: Path | None = None


def _expand_runtime_path(value: str | Path) -> Path:
    """Expands a given path, resolving WorldFoundry specific prefixes and ensuring it's an absolute path.

    If the path is not absolute after resolving WorldFoundry prefixes, it's treated as
    relative to the project root directory.

    Args:
        value: The path to expand, which can be a string or a Path object.

    Returns:
        The fully resolved, absolute path.
    """
    # Resolve WorldFoundry-specific path prefixes (e.g., "worldfoundry://").
    path = resolve_worldfoundry_path(value)
    # If the path is still not absolute, assume it's relative to the project's root directory.
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def select_diffusion_policy_checkpoint(
    *,
    checkpoint_path: str | Path | None,
    checkpoints: tuple[Mapping[str, Any], ...],
) -> Path:
    """Select the checkpoint path for the in-tree Diffusion Policy runtime.

    Prioritizes an explicit `checkpoint_path` provided by the caller. If not provided,
    it falls back to the first checkpoint's `local_dir` found in the `checkpoints`
    list (typically from a runtime profile).

    Args:
        checkpoint_path: Explicit checkpoint override supplied by the caller.
        checkpoints: Runtime-profile checkpoint records used when no override exists.

    Raises:
        ValueError: If no checkpoint path can be determined from the inputs.
        FileNotFoundError: If the determined checkpoint path does not exist as a file.

    Returns:
        The resolved and validated path to the Diffusion Policy checkpoint file.
    """
    candidate = checkpoint_path
    # If no explicit checkpoint path is provided, try to use the first checkpoint from the profile.
    if candidate is None and checkpoints:
        candidate = checkpoints[0].get("local_dir")
    if candidate is None:
        raise ValueError("Diffusion Policy requires a checkpoint_path or profile checkpoint.")
    
    # Expand the path to be absolute and resolve any project-relative references.
    path = _expand_runtime_path(candidate)
    if not path.is_file():
        raise FileNotFoundError(f"Diffusion Policy checkpoint not found: {path}")
    return path


def _install_runtime_alias() -> None:
    """Dynamically imports the in-tree Diffusion Policy runtime package.

    This ensures that `diffusion_policy` refers to the package located within the `RUNTIME_ROOT`
    directory, preventing conflicts if another `diffusion_policy` package is already installed
    or imported from elsewhere. It registers the in-tree package in `sys.modules`.

    Raises:
        RuntimeError: If `diffusion_policy` is already imported from an external location,
                      or if the in-tree package cannot be loaded.
    """
    # Check if 'diffusion_policy' is already in sys.modules.
    if "diffusion_policy" in sys.modules:
        module = sys.modules["diffusion_policy"]
        module_file = Path(str(getattr(module, "__file__", ""))).resolve()
        # If it's already imported and points to the in-tree runtime, do nothing.
        if RUNTIME_ROOT in module_file.parents or module_file == RUNTIME_ROOT / "__init__.py":
            return
        # If it's imported from an external location, raise an error to prevent conflicts.
        raise RuntimeError(f"diffusion_policy is already imported from outside the in-tree runtime: {module_file}")

    # Create a module specification for the in-tree 'diffusion_policy' package.
    spec = importlib.util.spec_from_file_location(
        "diffusion_policy",
        RUNTIME_ROOT / "__init__.py",
        submodule_search_locations=[str(RUNTIME_ROOT)],
    )
    # Ensure the specification and its loader are valid.
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Diffusion Policy runtime package from {RUNTIME_ROOT}")
    # Create the module object from the spec.
    module = importlib.util.module_from_spec(spec)
    # Register the new module in sys.modules, effectively making it importable.
    sys.modules["diffusion_policy"] = module
    # Execute the module's code to fully initialize it.
    spec.loader.exec_module(module)


def _coerce_observation(observation: Any, *, device: str) -> Mapping[str, Any]:
    """Normalize caller observations into the upstream lowdim policy input format.

    This function handles converting various observation formats (e.g., NumPy arrays,
    PyTorch tensors, or dictionaries containing an 'obs' key) into a standardized
    `{"obs": torch.Tensor}` dictionary suitable for the Diffusion Policy.
    It ensures the tensor is on the correct device and has the expected dimensions.

    Args:
        observation: Tensor, ndarray, or mapping containing an ``obs`` value.
                     Expected shape is `[B, T, D]` or `[T, D]`.
        device: Torch device used for inference tensors (e.g., "cuda" or "cpu").

    Raises:
        ValueError: If `observation` does not contain a valid 'obs' input or
                    if the final tensor shape is incorrect.

    Returns:
        A dictionary `{"obs": torch.Tensor}` with the observation coerced to
        the required format, device, and dtype.
    """
    import numpy as np
    import torch

    # Extract the observation value, supporting both direct input and dictionary with "obs" key.
    obs_value = observation.get("obs") if isinstance(observation, Mapping) else observation
    if obs_value is None:
        raise ValueError("Diffusion Policy predict_action requires observation/obs input.")
    
    # Convert NumPy array to PyTorch tensor.
    if isinstance(obs_value, np.ndarray):
        obs_tensor = torch.from_numpy(obs_value)
    else:
        obs_tensor = obs_value
    
    # Ensure the input is a PyTorch tensor, coercing if necessary.
    if not torch.is_tensor(obs_tensor):
        obs_tensor = torch.as_tensor(obs_tensor)
    
    # Move the tensor to the specified device and ensure float32 dtype.
    obs_tensor = obs_tensor.to(device=device, dtype=torch.float32)
    
    # Add a batch dimension if the observation is a single sequence ([T, D] -> [1, T, D]).
    if obs_tensor.ndim == 2:
        obs_tensor = obs_tensor.unsqueeze(0)
    
    # Validate the final shape: expected [B, T, D].
    if obs_tensor.ndim != 3:
        raise ValueError("Diffusion Policy obs must have shape [B, T, D] or [T, D].")
    return {"obs": obs_tensor}


class DiffusionPolicyRuntime:
    """A class to manage and execute an in-tree Diffusion Policy model for action prediction.

    This class provides a lazy loading mechanism for the policy model, ensuring that
    the model is only loaded into memory when inference is first requested. It supports
    configuring the checkpoint, device, and output directory, and generates
    WorldFoundry-compatible action traces.
    """
    def __init__(self, config: DiffusionPolicyRuntimeConfig) -> None:
        """Create a lazy in-tree Diffusion Policy checkpoint runtime.

        Initializes the runtime with the provided configuration but defers
        the actual loading of the policy model until `_load_policy` is called.

        Args:
            config: Checkpoint, device, and output-directory settings for inference.
        """
        self.config = config
        self._policy = None

    def _load_policy(self) -> Any:
        """Loads the Diffusion Policy model checkpoint, initializes the workspace, and prepares the policy for inference.

        This method is called lazily and ensures the policy model is loaded and ready
        for prediction. It handles dynamic import of the Diffusion Policy package,
        deserialization of the checkpoint, workspace setup, and model state loading.

        Returns:
            The loaded and configured Diffusion Policy model instance.

        Raises:
            RuntimeError: If the in-tree runtime cannot be installed or if the
                          model loading process encounters issues.
        """
        # If the policy is already loaded, return the cached instance.
        if self._policy is not None:
            return self._policy

        # Dynamically install the in-tree diffusion_policy package alias.
        _install_runtime_alias()
        import dill
        import hydra
        import torch

        from diffusion_policy.workspace.base_workspace import BaseWorkspace

        # Load the policy checkpoint, including its configuration and state dictionaries.
        # Use dill for deserialization as Diffusion Policy checkpoints often contain custom objects.
        payload = torch.load(self.config.checkpoint_path.open("rb"), pickle_module=dill, map_location="cpu")
        cfg = payload["cfg"]
        
        # Instantiate the workspace class defined in the checkpoint's configuration.
        workspace_cls = hydra.utils.get_class(cfg._target_)
        output_dir = str(self.config.output_dir) if self.config.output_dir is not None else None
        workspace = workspace_cls(cfg, output_dir=output_dir)
        
        state_dicts = payload.get("state_dicts", {})
        # Identify state dictionary keys that correspond to policy models (e.g., 'model', 'ema_model').
        policy_state_keys = {"model", "ema_model"}
        loadable_policy_keys = {
            key
            for key in state_dicts
            if key in policy_state_keys and hasattr(getattr(workspace, key, None), "load_state_dict")
        }
        # Exclude state dicts that are not policy-related or cannot be loaded into the workspace.
        exclude_keys = tuple(key for key in state_dicts if key not in loadable_policy_keys)
        # Load the payload into the workspace, specifically handling policy-related components.
        workspace.load_payload(payload, exclude_keys=exclude_keys, include_keys=())
        
        # Determine whether to use the Exponential Moving Average (EMA) model if enabled in training config.
        training_cfg = getattr(cfg, "training", None)
        use_ema = bool(getattr(training_cfg, "use_ema", False))
        policy = workspace.ema_model if use_ema and workspace.ema_model is not None else workspace.model
        
        # Move the selected policy model to the configured device and set it to evaluation mode.
        policy.to(torch.device(self.config.device))
        policy.eval()
        assert isinstance(workspace, BaseWorkspace) # Ensure type consistency for downstream use.
        self._policy = policy
        return policy

    def predict_action(
        self,
        *,
        observation: Any,
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run upstream ``policy.predict_action`` and write a WorldFoundry trace.

        This method performs inference using the loaded Diffusion Policy model and
        serializes the prediction results into a JSON-formatted WorldFoundry action trace file.

        Args:
            observation: Lowdim observation with shape ``[B, T, D]`` or ``[T, D]``.
                         Can be a NumPy array, PyTorch tensor, or a dictionary
                         containing an "obs" key.
            output_path: JSON artifact path for the action trace file.
            extra_metadata: Optional additional WorldFoundry metadata to include in the trace.

        Returns:
            A dictionary containing the prediction status, model ID, artifact details,
            runtime path, and the generated prediction.
        """
        import torch

        # Lazily load the policy model if it hasn't been loaded yet.
        policy = self._load_policy()
        
        # Coerce the input observation into the expected PyTorch tensor format for the policy.
        obs_dict = _coerce_observation(observation, device=self.config.device)
        
        # Perform inference without gradient computation.
        with torch.no_grad():
            raw_result = policy.predict_action(obs_dict)
        
        # Construct the WorldFoundry action trace dictionary.
        trace = {
            "model_id": "diffusion-policy",
            "artifact_kind": "action_trace",
            "runtime": "worldfoundry.diffusion_policy.in_tree_runtime.predict_action",
            "checkpoint_path": str(self.config.checkpoint_path),
            "device": self.config.device,
            # Convert prediction tensors to CPU NumPy arrays/lists and make them JSON-safe.
            "prediction": jsonable({key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in raw_result.items()}),
            "metadata": jsonable(dict(extra_metadata or {})),
        }
        
        # Resolve the output path, create parent directories, and write the trace to a JSON file.
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        
        # Return a summary of the prediction, including the path to the saved artifact.
        return {
            "status": "success",
            "model_id": "diffusion-policy",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "runtime": trace["runtime"],
            "prediction": trace["prediction"],
        }
