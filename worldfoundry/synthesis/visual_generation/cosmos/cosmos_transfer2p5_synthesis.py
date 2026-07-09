"""
Module for defining and synthesizing a plan for the Cosmos Transfer2.5 video generation model.

This module provides classes to represent the configuration and readiness of the
Cosmos Transfer2.5 model, including paths to its source pipeline and checkpoints.
It enables the creation of a runtime plan, typically output as a JSON file,
detailing the model's status and intended usage for video inference.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ...base_synthesis import BaseSynthesis
from ....base_models.diffusion_model.video import cosmos2p5 as cosmos2p5_base
from ....base_models.diffusion_model.video.cosmos2p5 import find_local_artifact_path


@dataclass(frozen=True)
class CosmosTransfer2p5Plan:
    """
    Represents a detailed plan for the Cosmos Transfer2.5 model, including
    its model ID, source pipeline path, checkpoint path, and any reasons
    why it might be blocked from execution.

    Attributes:
        model_id: Identifier for the model (e.g., Hugging Face repo ID).
        source_pipeline: Path to the Python file containing the model's pipeline definition.
        checkpoint_path: Optional path to the model's local checkpoint or weights.
        blocked_reason: Optional string explaining why the model cannot be fully
                        synthesized or run (e.g., missing files).
        notes: A tuple of strings providing additional information or instructions
               regarding the model plan.
    """

    model_id: str
    source_pipeline: Path
    checkpoint_path: Path | None
    blocked_reason: str | None
    notes: tuple[str, ...]


class CosmosTransfer2p5Synthesis(BaseSynthesis):
    """
    Manages the synthesis process for the Cosmos Transfer2.5 model, creating
    a plan and reporting its readiness for controlled video inference.

    This class extends BaseSynthesis to provide methods for initializing
    a plan and generating a runtime prediction (which in this case
    outputs a detailed plan instead of performing actual inference).
    """

    def __init__(self, plan: CosmosTransfer2p5Plan):
        """
        Initializes the CosmosTransfer2p5Synthesis with a specific plan.

        Args:
            plan: A CosmosTransfer2p5Plan instance detailing the model's
                  configuration and readiness.
        """
        super().__init__()
        self.plan = plan

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "nvidia/Cosmos-Transfer2.5-2B",
        controlnet_variant: str = "edge",
        **_: Any,
    ) -> "CosmosTransfer2p5Synthesis":
        """
        Constructs a CosmosTransfer2p5Synthesis instance by building a
        Transfer2.5 plan based on a pretrained model identifier and ControlNet variant.

        This method attempts to locate the necessary pipeline source and model
        checkpoints locally to determine the model's readiness without
        requiring an external checkout.

        Args:
            model_path: The Hugging Face repo ID or a local directory path
                        where the model checkpoint is expected.
            controlnet_variant: The specific ControlNet variant (e.g., "edge", "canny")
                                expected by the transfer model.
            **_: Arbitrary keyword arguments, ignored in this implementation.

        Returns:
            A CosmosTransfer2p5Synthesis instance initialized with the
            constructed plan.
        """
        # Construct the expected path for the Cosmos Transfer2.5 pipeline source file.
        source_pipeline = Path(cosmos2p5_base.__file__).with_name("pipeline").joinpath(
            "pipeline_cosmos2_5_transfer.py"
        )
        # Attempt to find the local ControlNet checkpoint artifact based on model_path and variant.
        checkpoint_path = find_local_artifact_path(
            model_path,
            (
                f"controlnet/general/{controlnet_variant}",
                "general",
                "transformer",
            ),
        )

        blocked_reason = None
        # Determine if the checkpoint is missing, which blocks inference.
        if checkpoint_path is None:
            blocked_reason = (
                "Cosmos Transfer2.5 checkpoint is not present under the configured WORLDFOUNDRY_HFD_ROOT. "
                "Set WORLDFOUNDRY_CKPT_DIR/WORLDFOUNDRY_HFD_ROOT or pass an explicit local model_path. "
                "The public repo is gated and cannot be fetched at runtime."
            )
        # Determine if the pipeline source file is missing, which also blocks inference.
        if not source_pipeline.exists():
            blocked_reason = "Cosmos Transfer2.5 pipeline source has not been vendored in-tree."

        # Create the plan object with all collected information.
        plan = CosmosTransfer2p5Plan(
            model_id=model_path,
            source_pipeline=source_pipeline,
            checkpoint_path=checkpoint_path,
            blocked_reason=blocked_reason,
            notes=(
                "Use diffusers Cosmos2_5_TransferPipeline semantics for controlled video inference.",
                "Load controlnet from diffusers/controlnet/general/<variant> when weights are available locally.",
            ),
        )
        return cls(plan=plan)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Generates and writes a Transfer2.5 runtime plan to a JSON file,
        reporting on checkpoint and runtime readiness.

        This method does not perform actual inference but rather outputs
        a structured plan detailing the model's configuration, status,
        and any reasons preventing its full execution.

        Args:
            args: Positional inference arguments (currently ignored, reserved for future runtime).
            kwargs: Keyword inference arguments (some are popped for the plan, others stored as 'extra').

        Returns:
            A dictionary summarizing the status of the plan creation,
            including model ID, artifact kind, backend, and the path to
            the generated plan file.
        """
        del args  # This method does not use positional arguments
        output_path = kwargs.pop("output_path", None)
        # Determine the full path where the plan JSON file will be written.
        plan_path = _plan_path(output_path)

        # Construct the payload dictionary for the JSON plan.
        # Note: The status is currently always "blocked" as full inference wiring is not yet enabled.
        payload = {
            "status": "blocked" if self.plan.blocked_reason else "blocked",
            "model_id": self.plan.model_id,
            "artifact_kind": "generated_world",  # Indicates the output is a synthetic world/plan
            "backend": "worldfoundry.cosmos_transfer2p5.in_tree_runtime_plan",
            "backend_quality": "in_tree_runtime_plan",
            "source_pipeline": str(self.plan.source_pipeline),
            "source_pipeline_exists": self.plan.source_pipeline.is_file(),
            "checkpoint_path": str(self.plan.checkpoint_path) if self.plan.checkpoint_path is not None else None,
            "prompt": kwargs.pop("prompt", ""),
            "has_images": kwargs.pop("images", None) is not None,
            "has_video": kwargs.pop("video", None) is not None,
            "interactions": _jsonable(kwargs.pop("interactions", [])),
            "fps": kwargs.pop("fps", None),
            "extra": _jsonable(kwargs),  # Any remaining kwargs are stored as extra information
            # Provide a detailed blocked reason, including a fallback if self.plan.blocked_reason is None
            "blocked_reason": self.plan.blocked_reason
            or "Cosmos Transfer2.5 checkpoint is local, but full inference wiring is not enabled in this wrapper yet.",
            "notes": list(self.plan.notes),
        }
        # Ensure the directory for the plan file exists.
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        # Write the plan payload to a JSON file.
        plan_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Return a summary dictionary. If there's a specific blocked reason from the plan, use it.
        if self.plan.blocked_reason is not None:
            return {
                "status": "blocked",
                "model_id": self.plan.model_id,
                "artifact_kind": "generated_world",
                "runtime": payload["backend"],
                "backend_quality": payload["backend_quality"],
                "plan_path": str(plan_path),
                "blocked_reason": self.plan.blocked_reason,
            }
        # Otherwise, use the more general blocked reason from the payload.
        return {
            "status": "blocked",
            "model_id": self.plan.model_id,
            "artifact_kind": "generated_world",
            "runtime": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "plan_path": str(plan_path),
            "blocked_reason": payload["blocked_reason"],
        }


def _plan_path(output_path: Any) -> Path:
    """
    Determines the final `Path` object for the plan JSON file based on the
    provided output path.

    If `output_path` is None, it defaults to `cosmos-transfer2.5-plan.json`
    in the current working directory. If `output_path` is a file path,
    it ensures the suffix is `.json`. If it's a directory, it appends
    `cosmos-transfer2.5-plan.json` to it.

    Args:
        output_path: The user-specified output path, which can be None,
                     a string representing a file, or a directory.

    Returns:
        A resolved `Path` object pointing to the final JSON plan file location.
    """
    if output_path is None:
        # If no output path is specified, default to a file in the current working directory.
        return (Path.cwd() / "cosmos-transfer2.5-plan.json").resolve()
    target = Path(output_path).expanduser()
    if target.suffix:
        # If the target path has a file extension, ensure it is .json.
        return target.with_suffix(".json").resolve()
    # If the target path is a directory (or has no suffix), append the default filename.
    return (target / "cosmos-transfer2.5-plan.json").resolve()


def _jsonable(value: Any) -> Any:
    """
    Recursively converts a given value into a JSON-serializable format.

    This utility function handles common Python types, converting them
    into types that `json.dumps` can process. It specifically converts
    dictionaries (keys to strings), lists/tuples/sets, and other objects
    to their string representation if they are not basic JSON types.

    Args:
        value: The value to convert. Can be a dictionary, list, tuple, set,
               string, int, float, bool, None, or any other object.

    Returns:
        A JSON-serializable representation of the input value.
    """
    if isinstance(value, dict):
        # Recursively convert dictionary keys to strings and values.
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        # Recursively convert elements of iterable collections.
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        # Basic JSON-serializable types and None are returned as is.
        return value
    # All other types are converted to their string representation.
    return str(value)