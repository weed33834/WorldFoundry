from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ...base_synthesis import BaseSynthesis
from .runtime import CausalForcingRuntime, RollingForcingRuntime, SelfForcingRuntime


class _BaseForcingSynthesis(BaseSynthesis):
    """Shared synthesis facade over an official forcing-family runtime."""

    MODEL_ID = ""
    DISPLAY_NAME = ""
    RUNTIME_CLS = SelfForcingRuntime

    def __init__(
        self,
        runtime: SelfForcingRuntime | CausalForcingRuntime | RollingForcingRuntime,
    ) -> None:
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
    ) -> "_BaseForcingSynthesis":
        del args
        runtime = cls.RUNTIME_CLS.from_pretrained(
            pretrained_model_path,
            device=device,
            model_id=model_id or cls.MODEL_ID,
            **kwargs,
        )
        return cls(runtime)

    def runtime_plan(self) -> dict[str, Any]:
        return self.runtime.runtime_plan()

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] = (),
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


class SelfForcingSynthesis(_BaseForcingSynthesis):
    """Synthesis facade for Self-Forcing."""

    MODEL_ID = "self-forcing"
    DISPLAY_NAME = "Self-Forcing"
    RUNTIME_CLS = SelfForcingRuntime


class CausalForcingSynthesis(_BaseForcingSynthesis):
    """Synthesis facade for Causal-Forcing."""

    MODEL_ID = "causal-forcing"
    DISPLAY_NAME = "Causal-Forcing"
    RUNTIME_CLS = CausalForcingRuntime


__all__ = ["CausalForcingSynthesis", "SelfForcingSynthesis"]
