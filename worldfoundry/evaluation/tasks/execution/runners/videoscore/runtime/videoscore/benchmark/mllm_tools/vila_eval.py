from __future__ import annotations

import os
from pathlib import Path
from typing import List

from transformers import GenerationConfig

from worldfoundry.base_models.llm_mllm_core.mllm.vila import Image, load_vila_model


class VILA:
    support_multi_image = True
    merged_image_files: list[str] = []

    def __init__(
        self,
        model_path: str = "Efficient-Large-Model/Llama-3-VILA1.5-8b",
        model_base: str | None = None,
        *,
        device_map: str | dict | None = "auto",
        torch_dtype: str = "float16",
        attn_implementation: str | None = "flash_attention_2",
        max_new_tokens: int = 512,
    ) -> None:
        del model_base
        self.model = load_vila_model(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.max_new_tokens = max_new_tokens

    def __call__(self, inputs: List[dict]) -> str:
        prompt_parts: list[object] = []
        for message in inputs:
            message_type = message.get("type")
            content = message.get("content")
            if message_type == "image":
                prompt_parts.append(Image(str(content)))
            elif message_type == "text":
                prompt_parts.append(str(content))
            else:
                raise ValueError(f"Unsupported VILA input type: {message_type!r}")

        config = GenerationConfig.from_dict(self.model.default_generation_config.to_dict())
        config.max_new_tokens = self.max_new_tokens
        config.do_sample = False
        config.temperature = 0
        config.top_p = 1.0
        return self.model.generate_content(prompt_parts, generation_config=config).strip()

    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)
