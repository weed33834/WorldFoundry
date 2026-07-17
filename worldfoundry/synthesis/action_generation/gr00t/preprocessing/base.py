# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal processor interface required by GR00T inference."""

from abc import abstractmethod
from typing import Any

import numpy as np
from transformers import ProcessorMixin

from .types import EmbodimentTag, ModalityConfig


class BaseProcessor(ProcessorMixin):
    @abstractmethod
    def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert one inference request into model inputs."""

    @abstractmethod
    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Decode normalized model output into embodiment actions."""

    @property
    @abstractmethod
    def collator(self):
        """Return the batch collation callable."""

    @abstractmethod
    def set_statistics(self, statistics: dict[str, Any], override: bool = False) -> None:
        """Set normalization statistics loaded from the checkpoint."""

    def eval(self):
        """Processor operations are deterministic; retain the familiar API."""
        return self

    def get_modality_configs(self) -> dict[str, dict[str, ModalityConfig]]:
        return self.modality_configs
