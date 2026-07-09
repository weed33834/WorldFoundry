"""WorldFoundry synthesis adapter for DualCamCtrl."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from worldfoundry.synthesis.base_synthesis import BaseSynthesis

from .runtime import (
    DEFAULT_BASE_REPO,
    DEFAULT_CONFIG,
    DEFAULT_DUALCAMCTRL_CHECKPOINT,
    DEFAULT_DUALCAMCTRL_REPO,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_TEST_CASE_DIR,
    DualCamCtrlRuntime,
    OFFICIAL_SOURCE_REPO,
)


class DualCamCtrlSynthesis(BaseSynthesis):
    """Synthesis adapter delegating inference to :class:`DualCamCtrlRuntime`."""

    MODEL_ID = DualCamCtrlRuntime.MODEL_ID
    DISPLAY_NAME = DualCamCtrlRuntime.DISPLAY_NAME

    def __init__(self, runtime: DualCamCtrlRuntime) -> None:
        super().__init__()
        self.runtime = runtime
        self.model_id = runtime.model_id
        self.model_name = runtime.model_name
        self.generation_type = runtime.generation_type
        self.device = runtime.device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "DualCamCtrlSynthesis":
        del args
        runtime = DualCamCtrlRuntime.from_pretrained(
            pretrained_model_path,
            device=device,
            model_id=model_id,
            **kwargs,
        )
        return cls(runtime)

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.runtime.predict(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            **kwargs,
        )


__all__ = [
    "DEFAULT_BASE_REPO",
    "DEFAULT_CONFIG",
    "DEFAULT_DUALCAMCTRL_CHECKPOINT",
    "DEFAULT_DUALCAMCTRL_REPO",
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_TEST_CASE_DIR",
    "DualCamCtrlSynthesis",
    "OFFICIAL_SOURCE_REPO",
]
