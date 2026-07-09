"""Module for the IRASim operator implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from .base_operator import BaseOperator


class IRASimOperator(BaseOperator):
    """Operator interface for IRASim (Interactive Robot Action Simulation) models.

    This operator normalizes complex robotic control data (such as continuous joint trajectories
    or discrete robot actions) and pairs them with initial multimodal states (prompts and frames)
    to generate simulated future video sequences.
    """

    MODEL_ID = "irasim"
    DEFAULT_INPUT_SCHEMA = {"prompt": True, "image": True, "video": True, "actions": ["robot_action"]}

    def __init__(self, input_schema: Dict[str, Any] | None = None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["textual_instruction", "visual_instruction", "action_instruction"])
        self.input_schema = {**self.DEFAULT_INPUT_SCHEMA, **dict(input_schema or {})}
        self.interaction_template = ["robot_action", "trajectory"]
        self.interaction_template_init()

    @staticmethod
    def _actions(interaction) -> list[str]:
        """Safely parses discrete robot actions or trajectory descriptors into normalized string lists."""
        if interaction is None:
            return []
        if isinstance(interaction, str):
            text = interaction.strip()
            return [text] if text else []
        if isinstance(interaction, Iterable):
            return [str(item).strip() for item in interaction if str(item).strip()]
        raise TypeError(f"IRASim interaction must be robot actions or trajectory paths, got {type(interaction).__name__}.")

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        self._actions(interaction)
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self) -> Dict[str, Any]:
        """Extracts the verified physical action trajectory and appends it to interaction history.

        Exposes `robot_action_sequence` specifically for downstream IRASim simulators which expect
        explicitly modeled robot kinematics rather than abstract user commands.
        """
        actions = self._actions(self.current_interaction[-1] if self.current_interaction else None)
        self.interaction_history.append(actions)
        result: Dict[str, Any] = {"actions": actions}
        if actions:
            result["robot_action_sequence"] = actions
        return result

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Resolves task descriptions into standard text prompts for multimodal conditioning."""
        text = prompt if prompt is not None else kwargs.get("task")
        return {"prompt": "" if text is None else str(text)}

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Passes through structural perceptions (images/videos) natively for downstream video generation models."""
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": dict(kwargs),
        }
