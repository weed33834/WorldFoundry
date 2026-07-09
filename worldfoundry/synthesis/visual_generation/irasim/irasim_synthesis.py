"""Module for the IRASim synthesis wrapper, providing an in-tree runtime for world model generation."""

from __future__ import annotations

import json
from pathlib import Path
from worldfoundry.core.io.paths import package_module_root as package_root
from typing import Any, Mapping, Sequence

from ...base_synthesis import BaseSynthesis


class IRASimSynthesis(BaseSynthesis):
    """In-tree IRASim runtime wrapper with vendored model and diffusion code."""

    MODEL_ID = "irasim"
    DISPLAY_NAME = "IRASim"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """
        Initializes the IRASimSynthesis instance.

        Args:
            model_id: Identifier for the model, defaults to `irasim`.
            device: The device to run the model on (e.g., "cuda", "cpu").
            checkpoint_path: Path to the model checkpoint. If None, it will be resolved at runtime.
            config_path: Path to the model configuration file. If None, it will be resolved at runtime.
        """
        super().__init__()
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "robotics_world_model"
        self.device = device
        # Resolve and store the absolute path for the checkpoint
        self.checkpoint_path = None if checkpoint_path is None else str(Path(checkpoint_path).expanduser())
        # Resolve and store the absolute path for the configuration
        self.config_path = None if config_path is None else str(Path(config_path).expanduser())

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "IRASimSynthesis":
        """
        Build a lazy IRASim runtime wrapper.

        Args:
            pretrained_model_path: Optional checkpoint path or mapping with runtime options.
            args: Unused compatibility argument for pipeline factories.
            device: Target torch device.
            model_id: Runtime profile id, defaults to ``irasim``.
            **kwargs: Optional ``checkpoint_path`` and ``config_path`` values.
        """
        # 'args' is a compatibility argument and is not used by IRASimSynthesis.
        del args
        # Initialize options from pretrained_model_path if it's a mapping, otherwise an empty dict.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If pretrained_model_path is a direct path (not a mapping), set it as checkpoint_path.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_path"] = str(pretrained_model_path)
        # Update options with any additional keyword arguments provided.
        options.update(kwargs)
        # Construct and return an IRASimSynthesis instance, resolving parameters from various sources.
        return cls(
            model_id=str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID),
            device=str(device or options.get("device") or "cuda"),
            checkpoint_path=options.get("checkpoint_path") or options.get("ckpt_path"),
            config_path=options.get("config_path") or options.get("config"),
        )

    @staticmethod
    def _runtime_root() -> Path:
        """
        Determines the root path for the IRASim runtime vendored within the package.

        Returns:
            A Path object pointing to the IRASim runtime directory.
        """
        return package_root("worldfoundry.synthesis.visual_generation.irasim.irasim_runtime")

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Prepares an IRASim in-tree runtime plan for world model generation.

        This method does not execute the generation directly but creates a JSON plan file
        that can be consumed by an external executor to perform the actual synthesis.

        Args:
            prompt: Optional prompt or instruction for the generation.
            images: Optional visual context metadata, typically a path or dict of image info.
            video: Optional video context metadata, typically a path or dict of video info.
            interactions: Optional sequence of actions or interaction trajectories.
            output_path: Target path for the output JSON plan file. If None, defaults to 'irasim_plan.json' in the current working directory.
            fps: Optional target frames per second for video generation.
            **kwargs: Additional runtime options to be included in the plan's 'extra' field.

        Returns:
            A dictionary confirming the plan preparation, including the status, model ID,
            artifact kind, and the path to the generated plan file.
        """
        import sys

        # Local import to prevent circular dependencies or unnecessary imports at module level.
        from worldfoundry.synthesis.visual_generation.irasim import irasim_runtime

        # Ensures legacy import paths are set up for the vendored irasim_runtime.
        irasim_runtime.ensure_legacy_import_paths()
        # Sets 'irasim_runtime' in sys.modules to allow direct import by other modules
        # that might expect it to be globally available.
        sys.modules.setdefault("irasim_runtime", irasim_runtime)

        # Determine the target path for the output JSON plan file.
        # If output_path is not provided, default to 'irasim_plan.json' in the current directory.
        target = Path(output_path) if output_path is not None else Path.cwd() / "irasim_plan.json"
        # Expand user path (e.g., ~) and resolve to an absolute path.
        target = target.expanduser().resolve()
        # Create parent directories if they don't exist, preventing FileNotFoundError.
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # Construct the payload dictionary containing all necessary information for the generation plan.
        payload = {
            "status": "prepared",
            "model_id": self.model_id,
            "artifact_kind": "generated_world",
            "backend": "worldfoundry.irasim.in_tree_runtime",
            "backend_quality": "in_tree_runtime_plan",
            "runtime_root": str(self._runtime_root()),
            "checkpoint_path": self.checkpoint_path,
            "config_path": self.config_path,
            "prompt": prompt,
            "has_images": images is not None,
            "has_video": video is not None,
            "interactions": list(interactions),
            "fps": fps,
            # Convert all extra keyword arguments to strings for serialization robustness.
            "extra": {key: str(value) for key, value in kwargs.items()},
            "note": (
                "IRASim source is vendored in-tree. Full generation remains gated on caller-provided "
                "external checkpoint archives and task-specific dataset/action fixtures."
            ),
        }
        # Write the payload to the target JSON file with UTF-8 encoding,
        # ensuring non-ASCII characters are preserved and formatted for readability.
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        
        # Return a summary of the prepared plan.
        return {
            "status": "prepared",
            "model_id": self.model_id,
            "artifact_kind": "generated_world",
            "artifact_path": str(target),
            "runtime": payload["backend"],
            "backend_quality": payload["backend_quality"],
        }