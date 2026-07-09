import os
from typing import List

from .mllm_utils import load_images
from worldfoundry.base_models.llm_mllm_core.mllm.otter import OtterVideo as OtterVideoModel

class MyOtterVideoModel(OtterVideoModel):
    def __init__(self, model_path:str="luodian/OTTER-Video-LLaMA7B-DenseCaption") -> None:
        super().__init__(model_path=model_path)
        
    def eval_forward(self, text_prompt: str, image_path: str):
        # Similar to the Idefics' eval_forward but adapted for Fuyu
        pass

class OtterVideo():
    support_multi_image = True
    merged_image_files = []
    def __init__(self, model_path:str="luodian/OTTER-Video-LLaMA7B-DenseCaption") -> None:
        """Llava model wrapper

        Args:
            model_path (str): Llava model name, e.g. "liuhaotian/llava-v1.5-7b" or "llava-hf/vip-llava-13b-hf"
        """
        self.model = MyOtterVideoModel(model_path=model_path)

        
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
            images = load_images(image_links)
            text_prompt = "\n".join([x["content"] for x in inputs if x["type"] == "text"])
            
            generated_text = self.model.get_response(images, text_prompt)
            return generated_text
                    
        else:
            raise NotImplementedError
        
    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)
