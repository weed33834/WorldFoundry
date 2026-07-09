from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import torch


class Tokenization(ABC):
    @abstractmethod
    def __call__(self, conversations: List[Dict], has_image: bool) -> dict[str, torch.Tensor]:
        pass


class DummyTokenization(Tokenization):
    def __call__(self, conversations: List[Dict], has_image: bool) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([0]),
            "labels": torch.tensor([0]),
        }
