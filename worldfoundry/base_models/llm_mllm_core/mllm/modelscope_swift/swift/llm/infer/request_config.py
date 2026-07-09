# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RequestConfig:
    """Generation options shared by local and server-backed inference engines."""

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    num_beams: int = 1
    stop: Optional[List[str]] = field(default_factory=list)

    seed: Optional[int] = None
    stream: bool = False
    logprobs: bool = False
    top_logprobs: Optional[int] = None

    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    length_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.stop is None:
            self.stop = []


__all__ = ["RequestConfig"]
