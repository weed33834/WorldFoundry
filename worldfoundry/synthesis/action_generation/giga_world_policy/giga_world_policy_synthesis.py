"""
Provides a synthesis wrapper for the GigaWorld-Policy action generation model.

This module defines the `GigaWorldPolicySynthesis` class, which extends `ActionModelSynthesis`
to handle the specific configuration, runtime, and prediction logic for
GigaWorld-Policy models. It manages model loading, runtime configuration resolution,
and orchestrates the prediction process, including generating action plans
and optionally performing in-tree inference.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from ..base_action_synthesis import ActionModelSynthesis
from worldfoundry.synthesis.action_generation.giga_world_policy.runtime import (
    GigaWorldPolicyRuntime,
    GigaWorldPolicyRuntimeConfig,
    select_giga_world_policy_paths,
)


def _string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    """Converts a value to a tuple of cleaned, non-empty strings.

    Args:
        value: The input value, which can be a single comma-separated string
               or a sequence of strings.
        name: The name of the value, used for error messages.

    Returns:
        A tuple of cleaned string items.

    Raises:
        TypeError: If the value is neither a string nor a sequence.
        ValueError: If the resulting tuple of strings is empty.
    """
    if isinstance(value, str):
        # Split a comma-separated string into items, strip whitespace, and filter out empty strings.
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Sequence):
        # Convert sequence items to strings, strip whitespace, and filter out empty strings.
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise TypeError(f"{name} must be a string or sequence of strings.")
    if not items:
        raise ValueError(f"{name} must not be empty.")
    return tuple(items)


class GigaWorldPolicySynthesis(ActionModelSynthesis):
    """A synthesis wrapper for the GigaWorld-Policy action generation model.

    This class provides methods to load, configure, and run the GigaWorld-Policy
    model for action synthesis, extending the base `ActionModelSynthesis` class.
    It handles runtime configuration, model instantiation, and prediction.
    """

    MODEL_ID = "giga-world-policy"

    def __init__(
        self,
        profile,
        *,
        device: str = "cuda",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Initializes the GigaWorldPolicySynthesis wrapper.

        Args:
            profile: The runtime profile for the model, providing paths and metadata.
            device: The device to run the model on (e.g., "cuda", "cpu").
            command_template: Optional template for external command execution.
            env: Optional environment variables for external command execution.
            runtime_options: Additional options specific to the GigaWorld-Policy runtime.
        """
        super().__init__(profile, device=device, command_template=command_template, env=env)
        # Store runtime options; defaults to an empty dictionary if None.
        self.runtime_options = dict(runtime_options or {})
        self._runtime: GigaWorldPolicyRuntime | None = None
        self._runtime_key: tuple[Any, ...] | None = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path=None,
        args=None,
        device=None,
        model_id: str | None = None,
        profile_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        acquisition_root: str | Path | None = None,
        hf_models_root: str | Path | None = None,
        command_template: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> "GigaWorldPolicySynthesis":
        """Create a lazy GigaWorld-Policy synthesis wrapper.

        This class method allows instantiating the synthesis wrapper by loading
        a pretrained model and resolving its runtime profile.

        Args:
            pretrained_model_path: Optional path to a pretrained model or a mapping of options.
            args: Unused compatibility argument, will be deleted.
            device: Runtime device string (e.g., "cuda", "cpu").
            model_id: The ID of the model to load, if not specified in options or profile.
            profile_path: Optional override path for the runtime profile file.
            manifest_path: Optional path to the acquisition manifest file.
            acquisition_root: Optional root directory for acquired assets.
            hf_models_root: Optional root directory for Hugging Face models cache.
            command_template: Optional command template override for execution.
            **kwargs: Additional options passed directly to the runtime configuration.

        Returns:
            An instance of `GigaWorldPolicySynthesis`.
        """
        del args  # This argument is deprecated and unused.

        # Initialize options from pretrained_model_path if it's a mapping, otherwise an empty dict.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If pretrained_model_path is provided and not a mapping, treat it as 'model_id_path'.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["model_id_path"] = str(pretrained_model_path)
        # Merge any additional keyword arguments into the options.
        options.update(kwargs)

        # Resolve the model ID from various possible sources, prioritizing explicit options.
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)

        # Load the runtime profile using the resolved model ID and optional path overrides.
        profile = load_runtime_profile(
            resolved_model_id,
            manifest_path=manifest_path or options.get("manifest_path"),
            profile_path=profile_path or options.get("profile_path"),
            acquisition_root=acquisition_root or options.get("acquisition_root"),
            hf_models_root=hf_models_root or options.get("hf_models_root"),
        )
        # Instantiate the class with the loaded profile and resolved options.
        return cls(
            profile=profile,
            device=str(device or options.get("device") or "cuda"),
            command_template=command_template or options.get("command_template"),
            env=options.get("env"),
            runtime_options=options,
        )

    def _runtime_config(self, options: Mapping[str, Any], *, require_existing: bool = True) -> GigaWorldPolicyRuntimeConfig:
        """Resolves runtime paths and scalar generation settings for the GigaWorld-Policy model.

        This method merges default runtime configurations with instance-level
        and call-level options to produce a complete `GigaWorldPolicyRuntimeConfig`.

        Args:
            options: Per-call runtime options that override instance or default settings.
            require_existing: If True, all model asset paths must exist locally; otherwise,
                              path validation may be more permissive (e.g., for plan-only mode).

        Returns:
            A `GigaWorldPolicyRuntimeConfig` object populated with resolved settings.
        """
        # Merge instance-level runtime options with call-level options,
        # with call-level options taking precedence.
        explicit_options = {**self.runtime_options, **dict(options)}

        # Load default runtime configuration specific to VLA/VA/WAM models.
        runtime_defaults = load_vla_va_wam_runtime_config(
            self.MODEL_ID,
            explicit_options.get("runtime_config_path"),
        )
        # Merge runtime defaults with explicit options.
        merged = {**runtime_defaults, **explicit_options}

        # Select and resolve specific model asset paths.
        paths = select_giga_world_policy_paths(
            model_id_path=merged.get("model_id_path") or merged.get("wan_model_path") or merged.get("checkpoint_dir"),
            transformer_path=merged.get("transformer_path"),
            stats_path=merged.get("stats_path") or merged.get("norm_stats_path"),
            t5_embedding_pkl=merged.get("t5_embedding_pkl") or merged.get("t5_embedding_path"),
            checkpoints=self.profile.checkpoints,
            require_existing=require_existing,
        )
        # Construct and return the GigaWorldPolicyRuntimeConfig object.
        return GigaWorldPolicyRuntimeConfig(
            model_id_path=paths["model_id_path"],
            transformer_path=paths["transformer_path"],
            stats_path=paths["stats_path"],
            t5_embedding_pkl=paths["t5_embedding_pkl"],
            device=str(merged.get("device") or self.device),
            dtype=str(merged["dtype"]),
            t5_len=int(merged["t5_len"]),
            # Resolve destination width/height from multiple possible keys.
            dst_width=int(explicit_options.get("dst_width") or explicit_options.get("width") or runtime_defaults["dst_width"]),
            dst_height=int(explicit_options.get("dst_height") or explicit_options.get("height") or runtime_defaults["dst_height"]),
            action_chunk=int(merged["action_chunk"]),
            num_frames=int(merged["num_frames"]),
            # Resolve number of inference steps from multiple possible keys.
            num_inference_steps=int(
                explicit_options.get("num_inference_steps")
                or explicit_options.get("infer_steps")
                or runtime_defaults["num_inference_steps"]
            ),
            # Resolve guidance scale from multiple possible keys.
            guidance_scale=float(
                explicit_options.get("guidance_scale")
                or explicit_options.get("cfg_scale")
                or runtime_defaults["guidance_scale"]
            ),
            norm_mode=str(merged["norm_mode"]),
            crop_mode=str(merged["crop_mode"]),
            state_dim=int(merged["state_dim"]),
            action_dim=int(merged["action_dim"]),
            delta_mask=str(merged["delta_mask"]),
            image_keys=_string_tuple(merged["image_keys"], name="image_keys"),
        )

    def _runtime_for(self, config: GigaWorldPolicyRuntimeConfig) -> GigaWorldPolicyRuntime:
        """Retrieves or creates a GigaWorldPolicyRuntime instance based on the given configuration.

        This method implements a caching mechanism: if the configuration matches the
        last used configuration, the existing runtime instance is returned. Otherwise,
        a new runtime instance is created and cached.

        Args:
            config: The `GigaWorldPolicyRuntimeConfig` to use for the runtime.

        Returns:
            A `GigaWorldPolicyRuntime` instance configured as specified.
        """
        # Create a unique key from the configuration values for caching.
        key = tuple(config.__dict__.values())
        # If no runtime exists or the configuration has changed, create a new runtime.
        if self._runtime is None or self._runtime_key != key:
            self._runtime = GigaWorldPolicyRuntime(config)
            self._runtime_key = key
        return self._runtime

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        timeout_seconds: int = 21600,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepares a GigaWorld-Policy plan and optionally runs in-tree inference.

        Args:
            prompt: The text prompt for the action generation.
            images: Input images for the model (e.g., NumPy array or tensor).
            video: Input video for the model.
            interactions: A sequence of interaction strings.
            output_path: Optional path to save the output artifact.
            fps: Frames per second for video output, if applicable.
            timeout_seconds: Maximum time allowed for prediction (currently unused).
            **kwargs: Additional runtime options or metadata.

        Returns:
            A dictionary containing the prediction result, run directory, plan path,
            and the resolved profile. If `plan_only` is true, it returns details
            about the prepared plan without running inference.
        """
        del timeout_seconds  # This argument is currently unused.

        # Determine if only a plan should be generated (without actual inference).
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))

        # Create or resolve a run directory for storing temporary files and outputs.
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="giga_world_policy_")).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Generate the context dictionary based on inputs.
        context = self._context(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            run_dir=run_dir,
            extra=kwargs,
        )

        # Resolve the runtime configuration, requiring existing paths only if not in plan_only mode.
        runtime_config = self._runtime_config(
            {**kwargs, "device": self.device},
            require_existing=not plan_only,
        )

        # Define the path for the runtime plan file.
        plan_path = run_dir / "runtime_profile_plan.json"
        # Construct the payload for the runtime plan, including schema, profile, context, and runtime details.
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-giga-world-policy-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": {
                "backend": "worldfoundry.giga_world_policy.in_tree_runtime",
                "model_id_path": str(runtime_config.model_id_path),
                "transformer_path": str(runtime_config.transformer_path),
                "stats_path": str(runtime_config.stats_path),
                "t5_embedding_pkl": str(runtime_config.t5_embedding_pkl),
                "device": runtime_config.device,
                "image_keys": list(runtime_config.image_keys),
            },
        }
        # Write the plan payload to a JSON file.
        plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # If only a plan is requested, return the plan details without running inference.
        if plan_only:
            return {
                "status": "prepared",
                "model_id": self.model_id,
                "artifact_kind": self.profile.artifact_kind,
                "artifact_path": str(context["output_path"]),
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "profile": self.profile.to_dict(),
            }

        # Resolve 'state' from kwargs or 'giga_world_policy_observation' if available.
        state = kwargs.get("state")
        observation = kwargs.get("giga_world_policy_observation")
        if state is None and isinstance(observation, Mapping):
            state = observation.get("state")

        # Get the runtime instance for the current configuration and perform prediction.
        result = self._runtime_for(runtime_config).predict_action(
            prompt=prompt,
            images=images,
            state=state,
            output_path=context["output_path"],
            extra_metadata={"run_dir": str(run_dir), "profile": self.profile.to_dict(), "interactions": list(interactions)},
        )
        # Return the prediction result augmented with run directory, plan path, and profile.
        return {**result, "run_dir": str(run_dir), "plan_path": str(plan_path), "profile": self.profile.to_dict()}