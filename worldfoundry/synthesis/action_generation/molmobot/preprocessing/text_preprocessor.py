"""Inference-only text/vision token interleaving for MolmoBot."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import BaseConfig


@dataclasses.dataclass
class TextPreprocessorConfig(BaseConfig):
    # Serialized by released checkpoints; loss-related values are accepted for
    # strict config compatibility but no label/loss path is implemented.
    max_answer_len: Optional[int] = None
    last_message_loss_only: bool = False
    max_text_tokens: Optional[int] = None
    loss_token_weighting: Optional[str] = None

    def build_text_preprocessor(self, tokenizer, max_seq_len):
        return InterleavedTextPreprocessor(
            tokenizer=tokenizer,
            max_text_tokens=self.max_text_tokens,
            max_sequence_length=max_seq_len,
        )


@dataclasses.dataclass
class InterleavedTextPreprocessor:
    tokenizer: Any
    max_text_tokens: Optional[int] = None
    max_sequence_length: Optional[int] = None

    def _tokenize_prompt(self, messages: List[str]) -> np.ndarray:
        if not messages or len(messages) % 2 != 1:
            raise ValueError("MolmoBot inference expects an odd, non-empty message list.")
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        token_ids = [bos]
        for index, message in enumerate(messages):
            token_ids.extend(self.tokenizer.encode(message))
            if index % 2 == 1:
                token_ids.append(self.tokenizer.eos_token_id)
        return np.asarray(token_ids, dtype=np.int64)

    def tokenize_and_interleave(
        self,
        message_list: List[str],
        multimodal_tokens: List[np.ndarray],
        multimodal_position_ids: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        text = self._tokenize_prompt(message_list)
        prompt_positions = np.flatnonzero(text == self.tokenizer.image_prompt_token_id)
        if len(prompt_positions) not in {0, len(multimodal_tokens)}:
            raise ValueError(
                f"Found {len(prompt_positions)} image placeholders for "
                f"{len(multimodal_tokens)} visual inputs."
            )
        insertion_points = (
            prompt_positions.tolist()
            if len(prompt_positions)
            else [1] * len(multimodal_tokens)
        )

        chunks: List[np.ndarray] = []
        positions: List[np.ndarray] = []
        text_offset = 0
        absolute_position = 0
        for index, insertion in enumerate(insertion_points):
            before = text[text_offset:insertion]
            chunks.append(before)
            positions.append(np.arange(absolute_position, absolute_position + len(before), dtype=np.int64))
            absolute_position += len(before)

            visual = np.asarray(multimodal_tokens[index], dtype=np.int64)
            chunks.append(visual)
            if multimodal_position_ids is None:
                positions.append(np.arange(absolute_position, absolute_position + len(visual), dtype=np.int64))
                absolute_position += len(visual)
            else:
                visual_positions = np.asarray(multimodal_position_ids[index], dtype=np.int64)
                if len(visual_positions) != len(visual):
                    raise ValueError("Visual token and position-id lengths differ.")
                positions.append(visual_positions + absolute_position)
                absolute_position += int(visual_positions.max()) + 1 if len(visual_positions) else 0
            text_offset = insertion + (1 if len(prompt_positions) else 0)

        remainder = text[text_offset:]
        chunks.append(remainder)
        positions.append(np.arange(absolute_position, absolute_position + len(remainder), dtype=np.int64))
        input_tokens = np.concatenate(chunks) if chunks else text
        position_ids = np.concatenate(positions) if positions else np.arange(len(text), dtype=np.int64)

        if self.max_sequence_length is not None and len(input_tokens) > self.max_sequence_length:
            raise ValueError(
                f"MolmoBot input has {len(input_tokens)} tokens, exceeding max_sequence_length="
                f"{self.max_sequence_length}."
            )
        return {
            "input_tokens": input_tokens,
            "position_ids": position_ids,
        }


__all__ = ["InterleavedTextPreprocessor", "TextPreprocessorConfig"]
