"""Forcing visual generation pipeline module."""

from __future__ import annotations

from typing import Any

from ...operators.forcing_operator import CausalForcingOperator, SelfForcingOperator
from ...synthesis.visual_generation.forcing import CausalForcingSynthesis, SelfForcingSynthesis
from ..pipeline_utils import PipelineABC


class _BaseForcingPipeline(PipelineABC):
    """Shared pipeline assembly for forcing-family official runners."""

    MODEL_ID = ""
    OPERATOR_CLS = SelfForcingOperator
    SYNTHESIS_CLS = SelfForcingSynthesis
    MEMORY_TARGET = None

    @classmethod
    def _uses_component_contract(cls) -> bool:
        """Determine whether the pipeline class implements the component contract."""
        return True

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "_BaseForcingPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        options = cls._runtime_options(model_path, required_components, kwargs)
        resolved_model_id = cls._resolve_model_id(options, model_id=model_id)
        if resolved_model_id != cls.MODEL_ID:
            raise ValueError(f"{cls.__name__} cannot load {resolved_model_id!r}.")
        synthesis_model = cls.SYNTHESIS_CLS.from_pretrained(
            {**options, "model_id": resolved_model_id},
            device=device,
        )
        return cls(
            model_id=resolved_model_id,
            operators=cls.OPERATOR_CLS(),
            synthesis_model=synthesis_model,
            memory_module=None,
            device=device,
        )


class SelfForcingPipeline(_BaseForcingPipeline):
    """WorldFoundry pipeline for Self-Forcing video generation."""

    MODEL_ID = "self-forcing"
    OPERATOR_CLS = SelfForcingOperator
    SYNTHESIS_CLS = SelfForcingSynthesis


class CausalForcingPipeline(_BaseForcingPipeline):
    """WorldFoundry pipeline for Causal-Forcing video generation."""

    MODEL_ID = "causal-forcing"
    OPERATOR_CLS = CausalForcingOperator
    SYNTHESIS_CLS = CausalForcingSynthesis


__all__ = ["CausalForcingPipeline", "SelfForcingPipeline"]
