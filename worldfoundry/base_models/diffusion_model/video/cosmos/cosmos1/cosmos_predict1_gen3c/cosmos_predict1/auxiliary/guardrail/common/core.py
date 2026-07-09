# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos1 -> cosmos_predict1_gen3c -> cosmos_predict1 -> auxiliary -> guardrail -> common -> core.py functionality."""

from typing import Any, Tuple

import numpy as np

from cosmos_predict1.utils import log


class ContentSafetyGuardrail:
    """Content safety guardrail implementation."""
    def is_safe(self, **kwargs) -> Tuple[bool, str]:
        """Is safe.

        Returns:
            The return value.
        """
        raise NotImplementedError("Child classes must implement the is_safe method")


class PostprocessingGuardrail:
    """Postprocessing guardrail implementation."""
    def postprocess(self, frames: np.ndarray) -> np.ndarray:
        """Postprocess.

        Args:
            frames: The frames.

        Returns:
            The return value.
        """
        raise NotImplementedError("Child classes must implement the postprocess method")


class GuardrailRunner:
    """Guardrail runner implementation."""
    def __init__(
        self,
        safety_models: list[ContentSafetyGuardrail] | None = None,
        generic_block_msg: str = "",
        generic_safe_msg: str = "",
        postprocessors: list[PostprocessingGuardrail] | None = None,
    ):
        """Init.

        Args:
            safety_models: The safety models.
            generic_block_msg: The generic block msg.
            generic_safe_msg: The generic safe msg.
            postprocessors: The postprocessors.
        """
        self.safety_models = safety_models
        self.generic_block_msg = generic_block_msg
        self.generic_safe_msg = generic_safe_msg if generic_safe_msg else "Prompt is safe"
        self.postprocessors = postprocessors

    def run_safety_check(self, input: Any) -> Tuple[bool, str]:
        """Run the safety check on the input."""
        if not self.safety_models:
            log.warning("No safety models found, returning safe")
            return True, self.generic_safe_msg

        for guardrail in self.safety_models:
            guardrail_name = str(guardrail.__class__.__name__).upper()
            log.debug(f"Running guardrail: {guardrail_name}")
            safe, message = guardrail.is_safe(input)
            if not safe:
                reasoning = self.generic_block_msg if self.generic_block_msg else f"{guardrail_name}: {message}"
                return False, reasoning
        return True, self.generic_safe_msg

    def postprocess(self, frames: np.ndarray) -> np.ndarray:
        """Run the postprocessing on the video frames."""
        if not self.postprocessors:
            log.warning("No postprocessors found, returning original frames")
            return frames

        for guardrail in self.postprocessors:
            guardrail_name = str(guardrail.__class__.__name__).upper()
            log.debug(f"Running guardrail: {guardrail_name}")
            frames = guardrail.postprocess(frames)
        return frames
