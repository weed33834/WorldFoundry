"""Module for the WoW operator implementation."""

from PIL import Image
from pathlib import Path
import cv2
from typing import Union, Optional, Dict, Any
from .base_operator import BaseOperator


def extract_first_frame(video_path):

    """Extract first frame implementation."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read video: {video_path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def load_input_image(input_path: Union[str, Path]) -> Image.Image:

    """Load input image implementation."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if input_path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
        return extract_first_frame(input_path)
    else:
        return Image.open(input_path).convert("RGB")


class WoWOperator(BaseOperator):
    """
    WoW 数据处理 Operator
    
    负责图像/视频加载、数据预处理等数据预处理工作
    不涉及模型推理和API调用
    """
    
    def __init__(self,
                 operation_types=None):

        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super(WoWOperator, self).__init__(operation_types=operation_types)
        self.interaction_template = ["text_prompt", "image_prompt"]
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
        input_path: Optional[Union[str, Path, Image.Image]] = None,
        **kwargs
    ) -> Dict[str, Any]:

        """Process perception inputs like images, videos, and reference frames."""
        if input_path is None:
            raise ValueError("input_path cannot be None")
        
        original_input_path = None
        if isinstance(input_path, Image.Image):
            input_image = input_path.convert("RGB")
        else:
            original_input_path = str(input_path)
            input_image = load_input_image(input_path)
        
        print(f"[WoWOperator] 成功加载图片: {input_path}, 图片尺寸: {input_image.size}")
        
        return {
            "input_image": input_image,
            "input_path": original_input_path,
        }
