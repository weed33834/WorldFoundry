"""Inference state containers shared by streaming generation runtimes."""

from __future__ import annotations


class InferenceParams:
    """State container used to cache key/value tensors during inference."""

    def __init__(self, max_batch_size: int, max_sequence_length: int):
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.key_value_memory_dict = {}
        self.update_kv_cache = False

    def swap_key_value_dict(self, batch_idx) -> None:
        if len(self.key_value_memory_dict) == 0:
            raise ValueError("should not swap when dict is empty")

        for layer_number in self.key_value_memory_dict.keys():
            inference_key_memory, inference_value_memory = self.key_value_memory_dict[layer_number]
            assert len(batch_idx) == inference_key_memory.shape[1]
            new_inference_key_memory = inference_key_memory[:, batch_idx]
            new_inference_value_memory = inference_value_memory[:, batch_idx]
            self.key_value_memory_dict[layer_number] = (new_inference_key_memory, new_inference_value_memory)


__all__ = ["InferenceParams"]
