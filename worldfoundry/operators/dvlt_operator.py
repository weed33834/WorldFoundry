"""Module for the DVLT operator implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .base_operator import BaseOperator


class DVLTOperator(BaseOperator):
    """Operator for DVLT multi-view 3D reconstruction inputs."""

    MODEL_ID = "dvlt"
    DEFAULT_INPUT_SCHEMA = {
        "prompt": False,
        "image": True,
        "video": True,
        "actions": ["view_sequence", "reference_frame", "reconstruction_options"],
    }

    def __init__(self, input_schema: Dict[str, Any] | None = None):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["visual_instruction", "action_instruction"])
        self.input_schema = {**self.DEFAULT_INPUT_SCHEMA, **dict(input_schema or {})}
        self.interaction_template = ["view_sequence", "reference_frame", "reconstruction_options"]
        self.interaction_template_init()

    @staticmethod
    def _normalise_actions(interaction: Any) -> list[Any]:
        """Normalise actions implementation."""
        if interaction is None:
            return []
        if isinstance(interaction, Path):
            return [str(interaction)]
        if isinstance(interaction, str):
            text = interaction.strip()
            return [text] if text else []
        if isinstance(interaction, Mapping):
            return [{str(key): value for key, value in interaction.items()}]
        if isinstance(interaction, Iterable):
            return [str(item) if isinstance(item, Path) else item for item in interaction]
        raise TypeError(
            "DVLT interaction must be a sequence, mapping, path, or string; "
            f"got {type(interaction).__name__}."
        )

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        self._normalise_actions(interaction)
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self) -> Dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        actions = self._normalise_actions(self.current_interaction[-1] if self.current_interaction else None)
        self.interaction_history.append(actions)
        return {
            "actions": actions,
            "reconstruction_controls": actions,
            "operator_metadata": {
                "model_id": self.MODEL_ID,
                "task": "multi_view_3d_reconstruction",
                "outputs": ["camera_poses", "depths", "world_points", "glb_pointcloud"],
            },
        }

    def process_prompt(self, prompt: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Process the input prompt or caption to ensure compatibility."""
        del kwargs
        return {"prompt": "" if prompt is None else str(prompt)}

    def process_perception(
        self,
        images: Any = None,
        video: Any = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        return {
            "images": images,
            "video": video,
            "ref_image_path": str(ref_image_path) if ref_image_path is not None else None,
            "extra_inputs": dict(kwargs),
        }
