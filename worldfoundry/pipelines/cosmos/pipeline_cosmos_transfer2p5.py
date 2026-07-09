"""Cosmos Transfer2P5 visual generation pipeline module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ...synthesis.visual_generation.cosmos.cosmos_transfer2p5_synthesis import CosmosTransfer2p5Synthesis
from ..pipeline_utils import PipelineABC


class CosmosTransfer2p5Pipeline(PipelineABC):
    """WorldFoundry pipeline wrapper for the in-tree Cosmos Transfer 2.5 plan surface."""

    MODEL_ID = "cosmos-transfer-2.5"

    def __init__(
        self,
        synthesis_model: CosmosTransfer2p5Synthesis | None = None,
        device: str = "cuda",
        model_id: str | None = None,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        super().__init__(
            model_id=model_id or self.MODEL_ID,
            synthesis_model=synthesis_model,
            device=device,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "CosmosTransfer2p5Pipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        options: dict[str, Any] = {}
        if isinstance(model_path, Mapping):
            options.update(model_path)
            model_path = (
                options.pop("model_path", None)
                or options.pop("pretrained_model_path", None)
                or options.pop("repo_id", None)
            )
        options.update(required_components or {})
        options.update(kwargs)
        synthesis_model = CosmosTransfer2p5Synthesis.from_pretrained(
            model_path=str(model_path or options.pop("checkpoint_path", "nvidia/Cosmos-Transfer2.5-2B")),
            controlnet_variant=str(options.pop("controlnet_variant", "edge")),
            device=device,
            **options,
        )
        return cls(
            synthesis_model=synthesis_model,
            device=device,
            model_id=model_id or cls.MODEL_ID,
        )

    def __call__(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise RuntimeError("CosmosTransfer2p5Pipeline synthesis_model is not initialized.")
        result = self.synthesis_model.predict(
            prompt=prompt or "",
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            **kwargs,
        )
        if return_dict:
            return result
        return result.get("artifact_path") or result


__all__ = ["CosmosTransfer2p5Pipeline"]
