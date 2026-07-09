"""Module for the Hailuo2p3 operator implementation."""

from typing import Optional, Dict, Any, Union

from PIL import Image

from .base_operator import BaseOperator
from ._media import image_or_string_to_png_data_url


class Hailuo2p3Operator(BaseOperator):
    """
    MiniMax Hailuo 2.3 数据处理 Operator
    """

    def __init__(
        self,
        operation_types: list = None,
    ):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super(Hailuo2p3Operator, self).__init__(operation_types)

        self.interaction_template = ["text_prompt", "image_prompt", "multimodal_prompt"]
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
        images: Optional[Union[Image.Image, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        result: Dict[str, Any] = {
            "first_frame_image": None,
            "images": None,
        }

        if images is not None:
            result["first_frame_image"] = image_or_string_to_png_data_url(images)
            result["images"] = images

        return result
