"""Module for the LumaRay2 operator implementation."""

from typing import Optional, Dict, Any

from .base_operator import BaseOperator
from ._media import require_public_url


def _validate_public_url(url: str, field_name: str) -> str:
    """Validate public url implementation."""
    return require_public_url(url, field_name, "Luma")


class LumaRay2Operator(BaseOperator):
    """
    Luma Ray2 数据处理 Operator

    Luma keyframe 输入需要公网 URL 或 generation id。
    """

    def __init__(
        self,
        operation_types: list = None,
    ):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["prompt_processing", "keyframe_processing"]
        super(LumaRay2Operator, self).__init__(operation_types)

        self.interaction_template = ["text_prompt", "image_prompt", "generation_prompt"]
        self.interaction_template_init()

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        if not isinstance(interaction, str):
            raise TypeError(f"Interaction must be a string, got {type(interaction)}")
        return True

    def process_interaction(self, **kwargs) -> Dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return {
            "processed_prompt": now_interaction
        }

    def process_perception(
        self,
        images: Optional[str] = None,
        last_frame: Optional[str] = None,
        start_generation_id: Optional[str] = None,
        end_generation_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        if images is not None and start_generation_id is not None:
            raise ValueError("Luma frame0 accepts either images or start_generation_id, not both.")
        if last_frame is not None and end_generation_id is not None:
            raise ValueError("Luma frame1 accepts either last_frame or end_generation_id, not both.")

        keyframes: Dict[str, Any] = {}

        if images is not None:
            keyframes["frame0"] = {
                "type": "image",
                "url": _validate_public_url(images, "images"),
            }
        elif start_generation_id is not None:
            if not isinstance(start_generation_id, str):
                raise TypeError(
                    f"start_generation_id must be a string, got {type(start_generation_id)}"
                )
            keyframes["frame0"] = {
                "type": "generation",
                "id": start_generation_id,
            }

        if last_frame is not None:
            keyframes["frame1"] = {
                "type": "image",
                "url": _validate_public_url(last_frame, "last_frame"),
            }
        elif end_generation_id is not None:
            if not isinstance(end_generation_id, str):
                raise TypeError(
                    f"end_generation_id must be a string, got {type(end_generation_id)}"
                )
            keyframes["frame1"] = {
                "type": "generation",
                "id": end_generation_id,
            }

        return {
            "keyframes": keyframes or None,
            "images": images,
            "last_frame": last_frame,
            "start_generation_id": start_generation_id,
            "end_generation_id": end_generation_id,
        }
