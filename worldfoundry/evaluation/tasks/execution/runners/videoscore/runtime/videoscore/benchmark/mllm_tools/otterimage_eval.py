import os
from typing import List
from .mllm_utils import merge_images
from worldfoundry.base_models.llm_mllm_core.mllm.otter import OtterImage as OtterImageModel


class OtterImage():
    support_multi_image = False
    merged_image_files = []
    def __init__(self, model_path:str="luodian/OTTER-Image-MPT7B") -> None:
        """Llava model wrapper

        Args:
            model_path (str): Llava model name, e.g. "liuhaotian/llava-v1.5-7b" or "llava-hf/vip-llava-13b-hf"
        """
        self.model = OtterImageModel(model_path=model_path)
        
    def __call__(self, inputs: List[dict]) -> str:
        """
        Args:
            inputs (List[dict]): [
                {
                    "type": "image",
                    "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"
                },
                {
                    "type": "image",
                    "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_337180_3.jpg"
                },
                {
                    "type": "text",
                    "content": "What is difference between two images?"
                }
            ]
            Supports any form of interleaved format of image and text.
        """
        image_links = [x["content"] for x in inputs if x["type"] == "image"]
        if self.support_multi_image:
            
            raise NotImplementedError
                    
        else:
            merge_image = merge_images(image_links)
            text_prompt = "\n".join([x["content"] for x in inputs if x["type"] == "text"])
            generated_text = self.model.generate(text_prompt, merge_image)
            return generated_text
        
    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)
