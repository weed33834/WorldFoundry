"""Skyreels V3 visual generation pipeline module."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ...base_models.diffusion_model.video.skyreels_v3.worldfoundry_runtime import SkyReelsV3Runtime
from ...synthesis.visual_generation.memory.video import VideoArtifactMemory
from ...operators.runtime_video_operator import RuntimeVideoOperator
from ..pipeline_utils import PipelineABC


class SkyReelsV3Pipeline(PipelineABC):
    """WorldFoundry pipeline for the vendored SkyReels V3 runtime entrypoint."""

    MODEL_ID = "skyreels-v3"
    DEFAULT_TASK_TYPE = "reference_to_video"

    def __init__(
        self,
        operator: Optional[RuntimeVideoOperator] = None,
        runtime: Optional[SkyReelsV3Runtime] = None,
        memory_module: Optional[VideoArtifactMemory] = None,
        task_type: str = DEFAULT_TASK_TYPE,
        device: str = "cuda",
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.operator = operator or RuntimeVideoOperator(generation_type="i2v")
        self.runtime = runtime
        self.memory_module = memory_module or VideoArtifactMemory(model_id=self.MODEL_ID)
        self.task_type = task_type
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> "SkyReelsV3Pipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        options = dict(required_components or {})
        if isinstance(model_path, dict):
            options.update(model_path)
            model_path = options.pop("model_path", None) or options.pop("repo_id", None)
        options.update(kwargs)
        task_type = str(options.pop("task_type", cls.DEFAULT_TASK_TYPE))
        for key in PipelineABC.FRAMEWORK_LOADING_OPTION_KEYS:
            options.pop(key, None)
        runtime = SkyReelsV3Runtime.from_pretrained(
            model_path=str(model_path) if model_path is not None else None,
            task_type=task_type,
            device=device,
            **options,
        )
        return cls(
            operator=RuntimeVideoOperator(generation_type="i2v"),
            runtime=runtime,
            memory_module=VideoArtifactMemory(model_id=cls.MODEL_ID),
            task_type=task_type,
            device=device,
        )

    def process(self, prompt: str, images: Any = None, video: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        self.operator.get_interaction(prompt)
        try:
            interaction = self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()
        return {
            "prompt": interaction["processed_prompt"],
            "images": images,
            "video": video,
            "kwargs": dict(kwargs),
        }

    def __call__(
        self,
        prompt: str,
        images: Any = None,
        video: Any = None,
        task_type: Optional[str] = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute the complete pipeline generation flow."""
        if self.runtime is None:
            raise RuntimeError("SkyReelsV3Pipeline is not loaded. Use from_pretrained() first.")
        processed = self.process(prompt=prompt, images=images, video=video, **kwargs)
        selected_task = task_type or self.task_type
        result = self.runtime.generate(
            selected_task,
            prompt=processed["prompt"],
            images=processed["images"],
            video=processed["video"],
            **processed["kwargs"],
        )
        if return_dict:
            return {
                "result": result,
                "prompt": processed["prompt"],
                "model_name": self.MODEL_ID,
                "task_type": selected_task,
            }
        return result

    def stream(self, prompt: str, **kwargs: Any) -> Any:
        """Stream visual generation outputs chunk by chunk."""
        result = self(prompt=prompt, return_dict=True, **kwargs)
        self.memory_module.record(
            result["result"],
            metadata={"prompt": prompt, "model_name": self.MODEL_ID, "task_type": result["task_type"]},
        )
        return result["result"]

    def get_operator(self) -> RuntimeVideoOperator:
        """Get operator for SkyReelsV3Pipeline."""
        return self.operator

    def get_synthesis_model(self) -> SkyReelsV3Runtime | None:
        """Get synthesis model for SkyReelsV3Pipeline."""
        return self.runtime


__all__ = ["SkyReelsV3Pipeline"]
