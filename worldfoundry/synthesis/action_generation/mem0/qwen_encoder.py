# Copyright 2025 MemoryMatters Team. All rights reserved.
# Licensed under the MIT License.
# Implemented by Yuran Wang / Peking University in 2025.
"""Inference-only Qwen3-VL feature encoder used by Mem-0."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast


class Qwen3VL_Encapsulation(nn.Module):
    """Checkpoint-compatible local-only Qwen3-VL wrapper."""

    def __init__(
        self,
        model_path: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        attention_implementation: str,
    ) -> None:
        super().__init__()
        device_map = {"": device} if device.type == "cuda" else None
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation=attention_implementation,
            device_map=device_map,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.processor.tokenizer.padding_side = "right"
        self.hidden_size = self.model.config.vision_config.out_hidden_size
        self.vision_start_token_id = self.processor.vision_start_token_id
        self.vision_end_token_id = self.processor.vision_end_token_id
        self.im_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.image_token_id = self.processor.image_token_id

    def forward(self, **kwargs: Any) -> CausalLMOutputWithPast:
        return self.model(**kwargs)

    def build_qwenvl_inputs(
        self,
        images: list[list[Any]],
        instructions: list[str],
        *,
        max_length: int,
    ) -> Any:
        if len(images) != len(instructions):
            raise ValueError("Mem-0 images and instructions must have the same batch size")
        messages = []
        for sample_images, instruction in zip(images, instructions, strict=True):
            content = [{"type": "image", "image": value} for value in sample_images]
            content.append({"type": "text", "text": instruction})
            messages.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
            padding="max_length",
            max_length=max_length,
        )
        return inputs.to(self.model.device)

    def extract_features(
        self,
        input_ids: torch.Tensor,
        last_hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean-pool the official vision and instruction token spans."""

        _, _, hidden_size = last_hidden_state.shape
        device = last_hidden_state.device
        image_features_list = []
        text_features_list = []
        for batch_input_ids, batch_hidden in zip(input_ids, last_hidden_state, strict=True):
            vision_starts = (batch_input_ids == self.vision_start_token_id).nonzero(as_tuple=True)[0]
            vision_ends = (batch_input_ids == self.vision_end_token_id).nonzero(as_tuple=True)[0]
            instruction_ends = (batch_input_ids == self.im_end_token_id).nonzero(as_tuple=True)[0]

            if vision_starts.numel() and vision_ends.numel():
                start = int(vision_starts[0]) + 1
                end = int(vision_ends[0])
                image_features = batch_hidden[start:end]
            else:
                image_features = torch.empty((0, hidden_size), device=device, dtype=batch_hidden.dtype)

            if vision_ends.numel() and instruction_ends.numel():
                start = int(vision_ends[0]) + 1
                end = int(instruction_ends[0])
                text_features = batch_hidden[start:end]
            else:
                text_features = torch.empty((0, hidden_size), device=device, dtype=batch_hidden.dtype)

            if image_features.shape[0]:
                image_features = image_features.mean(dim=0, keepdim=True)
            else:
                image_features = torch.zeros((1, hidden_size), device=device, dtype=batch_hidden.dtype)
            if text_features.shape[0]:
                text_features = text_features.mean(dim=0, keepdim=True)
            else:
                text_features = torch.zeros((1, hidden_size), device=device, dtype=batch_hidden.dtype)
            image_features_list.append(image_features)
            text_features_list.append(text_features)

        return (
            torch.cat(image_features_list, dim=0).unsqueeze(1),
            torch.cat(text_features_list, dim=0).unsqueeze(1),
        )


__all__ = ["Qwen3VL_Encapsulation"]
