"""Module for the Sora2 operator implementation."""

from PIL import Image
from typing import Optional, Dict, Any, Tuple

from .base_operator import BaseOperator
from ._media import pil_to_png_bytes



class Sora2Operator(BaseOperator):
    """
    Sora2 数据处理 Operator
    
    负责图像编码、数据预处理等数据预处理工作
    不涉及模型推理和API调用
    """
    
    def __init__(
        self,
        operation_types: list = None
    ):
        """
        初始化 Sora2Operator
        
        Args:
            operation_types: 操作类型列表
        """
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super(Sora2Operator, self).__init__(operation_types)
        
        # 初始化交互模板   
        self.interaction_template = ["text_prompt", "image_prompt", "multimodal_prompt"]
        self.interaction_template_init()
    
    def process_image(self, image_input: Image.Image) -> Tuple[str, bytes, str]:
        """
        处理图像，返回文件名、字节和mime类型（API所需格式）
        
        Args:
            image_input: PIL.Image 对象
            
        Returns:
            Tuple[str, bytes, str]: (文件名, 图像字节, mime类型)
        """
        image_bytes = pil_to_png_bytes(image_input)
        mime_type = 'image/png'
        filename = 'reference.png'
        
        return (filename, image_bytes, mime_type)

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
        images: Optional[Image.Image] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理交互输入，生成模型所需的输入格式
        
        Args:
            images: 参考图像（PIL.Image，可选）
            **kwargs: 其他参数
            
        Returns:
            Dict 包含处理后的输入数据：
                - encoded_image: 图像元组 (filename, bytes, mime_type)（如果有）
                - images: 原始参考图像（如果有）
        """
        result: Dict[str, Any] = {
            "encoded_image": None,
            "images": None
        }
        
        if images is not None:
            if not isinstance(images, Image.Image):
                raise TypeError(f"images must be PIL.Image, got {type(images)}")
            result["encoded_image"] = self.process_image(images)
            result["images"] = images
        
        return result
