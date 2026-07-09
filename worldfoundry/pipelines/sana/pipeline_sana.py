"""Sana visual generation pipeline module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ...synthesis.visual_generation.memory.video import VideoArtifactMemory
from ...operators.runtime_video_operator import RuntimeVideoOperator
from ...synthesis.visual_generation.sana.sana_synthesis import SanaSynthesis
from ...base_models.diffusion_model.image.sana.variants import get_sana_variant
from ..pipeline_utils import PipelineABC


class SanaPipeline(PipelineABC):
    """WorldFoundry pipeline wrapper for the Sana family."""

    MODEL_ID = "sana"
    SYNTHESIS_CLS = SanaSynthesis

    def __init__(
        self,
        *,
        model_id: str | None = None,
        operator: Optional[RuntimeVideoOperator] = None,
        synthesis_model: Optional[SanaSynthesis] = None,
        memory_module: Optional[VideoArtifactMemory] = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        requested_model_id = model_id or self.MODEL_ID
        self.variant = get_sana_variant(requested_model_id)
        self.model_id = self.variant.model_id
        self.synthesis_model = synthesis_model
        self.generation_type = self.variant.task
        self.model_name = self.variant.display_name
        self.operator = operator or RuntimeVideoOperator(
            generation_type="i2v" if self.variant.runner == "controlnet" else "t2v"
        )
        self.operators = self.operator
        self.memory_module = memory_module or VideoArtifactMemory(model_id=self.model_id)
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        model_id: str | None = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> "SanaPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        del lazy
        options: Dict[str, Any] = {}
        if isinstance(model_path, dict):
            options.update(model_path)
        elif model_path is not None:
            options["model_path"] = str(model_path)
        options.update(required_components or {})
        options.update(kwargs)
        resolved_model_id = str(
            options.get("model_id")
            or options.get("variant")
            or options.get("profile_id")
            or model_id
            or cls.MODEL_ID
        )
        synthesis_model = cls.SYNTHESIS_CLS.from_pretrained(
            {**options, "model_id": resolved_model_id},
            device=device,
        )
        variant = get_sana_variant(resolved_model_id)
        return cls(
            model_id=resolved_model_id,
            operator=RuntimeVideoOperator(generation_type="i2v" if variant.runner == "controlnet" else "t2v"),
            synthesis_model=synthesis_model,
            memory_module=VideoArtifactMemory(model_id=variant.model_id),
            device=device,
        )

    def process(self, prompt: str | None = None, images: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        if prompt is None:
            prompt = ""
        self.operator.get_interaction(prompt)
        try:
            interaction = self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()
        return {
            "prompt": interaction["processed_prompt"],
            "images": images,
            "extra_inputs": dict(kwargs),
        }

    def __call__(
        self,
        prompt: str | None = None,
        images: Any = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ):
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise RuntimeError("Sana synthesis model is not loaded. Use from_pretrained() first.")
        processed = self.process(prompt=prompt, images=images, **kwargs.pop("operator_kwargs", {}))
        result = self.synthesis_model.predict(
            prompt=processed["prompt"],
            images=processed["images"],
            output_path=output_path,
            fps=fps,
            **processed["extra_inputs"],
            **kwargs,
        )
        self.memory_module.record(result, metadata={"type": "sana_result", "model_id": self.model_id})
        if return_dict:
            return result
        return result.get("artifact_path") or result.get("generated_video_path") or result.get("generated_image_path") or result

    def stream(
        self,
        prompt: str | None = None,
        images: Any = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ):
        """Stream visual generation outputs chunk by chunk."""
        if images is None and self.variant.runner == "controlnet":
            previous = self.memory_module.select(prefer_type="image")
            if previous is not None:
                images = previous
        return self(
            prompt=prompt,
            images=images,
            output_path=output_path,
            fps=fps,
            return_dict=return_dict,
            **kwargs,
        )

    def get_operator(self) -> RuntimeVideoOperator:
        """Get operator for SanaPipeline."""
        return self.operator

    def get_synthesis_model(self) -> SanaSynthesis:
        """Get synthesis model for SanaPipeline."""
        if self.synthesis_model is None:
            raise RuntimeError("Sana synthesis model is not loaded.")
        return self.synthesis_model


class Sana600M512pxPipeline(SanaPipeline):
    """Pipeline implementation for Sana600M512px visual generation."""
    MODEL_ID = "sana-600m-512px"


class Sana600M1024pxPipeline(SanaPipeline):
    """Pipeline implementation for Sana600M1024px visual generation."""
    MODEL_ID = "sana-600m-1024px"


class Sana1600M512pxPipeline(SanaPipeline):
    """Pipeline implementation for Sana1600M512px visual generation."""
    MODEL_ID = "sana-1600m-512px"


class Sana1600M512pxMultilingPipeline(SanaPipeline):
    """Pipeline implementation for Sana1600M512pxMultiling visual generation."""
    MODEL_ID = "sana-1600m-512px-multiling"


class Sana1600M1024pxPipeline(SanaPipeline):
    """Pipeline implementation for Sana1600M1024px visual generation."""
    MODEL_ID = "sana-1600m-1024px"


class Sana1600M1024pxMultilingPipeline(SanaPipeline):
    """Pipeline implementation for Sana1600M1024pxMultiling visual generation."""
    MODEL_ID = "sana-1600m-1024px-multiling"


class Sana1600M1024pxBf16Pipeline(SanaPipeline):
    """Pipeline implementation for Sana1600M1024pxBf16 visual generation."""
    MODEL_ID = "sana-1600m-1024px-bf16"


class Sana1600M2kBf16Pipeline(SanaPipeline):
    """Pipeline implementation for Sana1600M2kBf16 visual generation."""
    MODEL_ID = "sana-1600m-2k-bf16"


class Sana1600M4kBf16Pipeline(SanaPipeline):
    """Pipeline implementation for Sana1600M4kBf16 visual generation."""
    MODEL_ID = "sana-1600m-4k-bf16"


class Sana1p51600M1024pxPipeline(SanaPipeline):
    """Pipeline implementation for Sana1p51600M1024px visual generation."""
    MODEL_ID = "sana1p5-1600m-1024px"


class Sana1p54800M1024pxPipeline(SanaPipeline):
    """Pipeline implementation for Sana1p54800M1024px visual generation."""
    MODEL_ID = "sana1p5-4800m-1024px"


class SanaSprint600M1024pxPipeline(SanaPipeline):
    """Pipeline implementation for SanaSprint600M1024px visual generation."""
    MODEL_ID = "sana-sprint-600m-1024px"


class SanaSprint1600M1024pxPipeline(SanaPipeline):
    """Pipeline implementation for SanaSprint1600M1024px visual generation."""
    MODEL_ID = "sana-sprint-1600m-1024px"


class SanaControlnet600M1024pxPipeline(SanaPipeline):
    """Pipeline implementation for SanaControlnet600M1024px visual generation."""
    MODEL_ID = "sana-controlnet-600m-1024px"


class SanaControlnet1600M1024pxBf16Pipeline(SanaPipeline):
    """Pipeline implementation for SanaControlnet1600M1024pxBf16 visual generation."""
    MODEL_ID = "sana-controlnet-1600m-1024px-bf16"


class SanaVideo2b480pPipeline(SanaPipeline):
    """Pipeline implementation for SanaVideo2b480p visual generation."""
    MODEL_ID = "sana-video-2b-480p"


class SanaVideo2b720pPipeline(SanaPipeline):
    """Pipeline implementation for SanaVideo2b720p visual generation."""
    MODEL_ID = "sana-video-2b-720p"


class LongsanaVideo2b480pPipeline(SanaPipeline):
    """Pipeline implementation for LongsanaVideo2b480p visual generation."""
    MODEL_ID = "longsana-video-2b-480p"


__all__ = [
    "LongsanaVideo2b480pPipeline",
    "Sana1600M1024pxBf16Pipeline",
    "Sana1600M1024pxMultilingPipeline",
    "Sana1600M1024pxPipeline",
    "Sana1600M2kBf16Pipeline",
    "Sana1600M4kBf16Pipeline",
    "Sana1600M512pxMultilingPipeline",
    "Sana1600M512pxPipeline",
    "Sana1p51600M1024pxPipeline",
    "Sana1p54800M1024pxPipeline",
    "Sana600M1024pxPipeline",
    "Sana600M512pxPipeline",
    "SanaControlnet1600M1024pxBf16Pipeline",
    "SanaControlnet600M1024pxPipeline",
    "SanaPipeline",
    "SanaSprint1600M1024pxPipeline",
    "SanaSprint600M1024pxPipeline",
    "SanaVideo2b480pPipeline",
    "SanaVideo2b720pPipeline",
]
