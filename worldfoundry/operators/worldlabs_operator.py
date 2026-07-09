"""Module for the WorldLabs operator implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from .base_operator import BaseOperator
from ._media import pil_to_png_base64


def _image_to_data_base64(image_input: Image.Image) -> str:
    """Image to data base64 implementation."""
    return pil_to_png_base64(image_input)


class WorldLabsOperator(BaseOperator):
    """
    World Labs / Marble 数据处理 Operator。

    支持 text / image / multi-image / video 四类 world prompt。
    """

    def __init__(
        self,
        operation_types: list = None,
    ):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["prompt_processing", "image_processing", "video_processing"]
        super(WorldLabsOperator, self).__init__(operation_types)

        self.interaction_template = ["text_prompt", "image_prompt", "multi_image_prompt", "video_prompt"]
        self.interaction_template_init()

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction):
        """Validate the given interaction sequence or parameters."""
        if interaction is not None and not isinstance(interaction, str):
            raise TypeError(f"Interaction must be a string or None, got {type(interaction)}")
        return True

    def process_interaction(self, **kwargs) -> Dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        now_interaction = self.current_interaction[-1] if self.current_interaction else ""
        if now_interaction is not None:
            self.interaction_history.append(now_interaction)
        return {
            "processed_prompt": now_interaction or ""
        }

    def _normalize_content(self, value: Union[Image.Image, str, Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize content implementation."""
        if isinstance(value, dict):
            return value

        if isinstance(value, Image.Image):
            return {
                "source": "data_base64",
                "data_base64": _image_to_data_base64(value),
                "extension": "png",
            }

        if isinstance(value, str):
            path = Path(value)
            if value.startswith(("http://", "https://")):
                return {
                    "source": "uri",
                    "uri": value,
                }
            if path.exists():
                return {
                    "source": "local_file",
                    "path": str(path.resolve()),
                }
            raise ValueError(
                "String image/video input must be a public URL or an existing local path."
            )

        raise TypeError(f"Unsupported content type: {type(value)}")

    def _build_multi_image_prompt(
        self,
        images: List[Union[Image.Image, str, Dict[str, Any]]],
        image_azimuths: Optional[List[float]],
    ) -> List[Dict[str, Any]]:
        """Build multi image prompt implementation."""
        if image_azimuths is not None and len(image_azimuths) != len(images):
            raise ValueError("image_azimuths length must match images length.")

        default_azimuths = [index * (360 / len(images)) for index in range(len(images))]
        azimuths = image_azimuths or default_azimuths

        return [
            {
                "azimuth": azimuths[index],
                "content": self._normalize_content(image),
            }
            for index, image in enumerate(images)
        ]

    def process_perception(
        self,
        images: Optional[Union[Image.Image, str, Dict[str, Any], List[Union[Image.Image, str, Dict[str, Any]]]]] = None,
        video: Optional[Union[str, Dict[str, Any]]] = None,
        prompt_type: str = "auto",
        image_azimuths: Optional[List[float]] = None,
        disable_recaption: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process perception inputs like images, videos, and reference frames."""
        result: Dict[str, Any] = {
            "world_prompt": None,
            "prompt_type": prompt_type,
            "images": images,
            "video": video,
        }

        if prompt_type == "auto":
            if video is not None:
                prompt_type = "video"
            elif isinstance(images, list):
                prompt_type = "multi-image"
            elif images is not None:
                prompt_type = "image"
            else:
                prompt_type = "text"

        if prompt_type == "text":
            result["world_prompt"] = {
                "type": "text",
            }
        elif prompt_type == "image":
            if images is None:
                raise ValueError("image prompt_type requires images input.")
            prompt_data: Dict[str, Any] = {
                "type": "image",
                "image_prompt": self._normalize_content(images),
            }
            if disable_recaption is not None:
                prompt_data["disable_recaption"] = disable_recaption
            result["world_prompt"] = prompt_data
        elif prompt_type in {"multi-image", "multi_image"}:
            if not isinstance(images, list) or len(images) == 0:
                raise ValueError("multi-image prompt_type requires a non-empty images list.")
            prompt_data = {
                "type": "multi-image",
                "multi_image_prompt": self._build_multi_image_prompt(images, image_azimuths),
            }
            if disable_recaption is not None:
                prompt_data["disable_recaption"] = disable_recaption
            result["world_prompt"] = prompt_data
        elif prompt_type == "video":
            if video is None:
                raise ValueError("video prompt_type requires video input.")
            prompt_data = {
                "type": "video",
                "video_prompt": self._normalize_content(video),
            }
            if disable_recaption is not None:
                prompt_data["disable_recaption"] = disable_recaption
            result["world_prompt"] = prompt_data
        else:
            raise ValueError(f"Unsupported prompt_type: {prompt_type}")

        result["prompt_type"] = prompt_type
        return result
