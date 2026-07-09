"""Module for the RunwayGen4p5 operator implementation."""

from typing import Optional, Dict, Any, List, Union

from PIL import Image

from .base_operator import BaseOperator
from ._media import image_or_string_to_png_data_url


class RunwayGen4p5Operator(BaseOperator):
    """
    Runway Gen-4.5 数据处理 Operator
    """

    def __init__(
        self,
        operation_types: list = None,
    ):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super(RunwayGen4p5Operator, self).__init__(operation_types)

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
        last_frame: Optional[Union[Image.Image, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        prompt_image: Optional[Union[str, List[Dict[str, str]]]] = None

        if images is not None and last_frame is not None:
            prompt_image = [
                {
                    "uri": image_or_string_to_png_data_url(images),
                    "position": "first",
                },
                {
                    "uri": image_or_string_to_png_data_url(last_frame),
                    "position": "last",
                },
            ]
        elif images is not None:
            prompt_image = image_or_string_to_png_data_url(images)
        elif last_frame is not None:
            raise ValueError("Runway last_frame requires a first image.")

        return {
            "prompt_image": prompt_image,
            "images": images,
            "last_frame": last_frame,
        }
