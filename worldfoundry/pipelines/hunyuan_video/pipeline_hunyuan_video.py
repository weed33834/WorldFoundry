"""Hunyuan Video visual generation pipeline module."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Optional

from ...synthesis.visual_generation.memory.video import VideoArtifactMemory
from ...operators.runtime_video_operator import RuntimeVideoOperator
from ...synthesis.visual_generation.official_video_runtime import OfficialVideoRuntime
from ...synthesis.visual_generation.runtime_video_synthesis import _frames_to_uint8_array
from ..pipeline_utils import PipelineABC


def _is_diffusers_checkpoint(path: Any) -> bool:
    """Is diffusers checkpoint helper function."""
    try:
        candidate = Path(str(path)).expanduser()
    except TypeError:
        return False
    return candidate.is_dir() and (candidate / "model_index.json").is_file()


def _is_official_checkpoint(path: Any) -> bool:
    """Is official checkpoint helper function."""
    try:
        candidate = Path(str(path)).expanduser()
    except TypeError:
        return False
    return candidate.is_dir() and (candidate / "hunyuan-video-t2v-720p").is_dir()


class HunyuanVideoT2VPipeline(PipelineABC):
    """WorldFoundry wrapper for the in-tree HunyuanVideo text-to-video runtime."""

    MODEL_ID = "hunyuanvideo-t2v"

    def __init__(
        self,
        operator: Optional[RuntimeVideoOperator] = None,
        runtime_pipeline: Any = None,
        model_path: Any = None,
        load_options: Optional[Dict[str, Any]] = None,
        memory_module: Optional[VideoArtifactMemory] = None,
        device: str = "cuda",
        lazy: bool = True,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.operator = operator or RuntimeVideoOperator(generation_type="t2v")
        self.runtime_pipeline = runtime_pipeline
        self.model_path = model_path
        self.load_options = dict(load_options or {})
        self.memory_module = memory_module or VideoArtifactMemory(model_id=self.MODEL_ID)
        self.device = device
        if not lazy:
            self._ensure_runtime_pipeline()

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        lazy: bool = True,
        **kwargs: Any,
    ) -> "HunyuanVideoT2VPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        load_options = dict(required_components or {})
        if isinstance(model_path, dict):
            load_options.update(model_path)
            model_path = (
                load_options.pop("model_path", None)
                or load_options.pop("pretrained_model_path", None)
                or load_options.pop("repo_id", None)
            )
        load_options.update(kwargs)
        for key in PipelineABC.FRAMEWORK_LOADING_OPTION_KEYS:
            load_options.pop(key, None)
        model_path = model_path or load_options.pop("pretrained_model_name_or_path", None)
        model_path = model_path or "tencent/HunyuanVideo"
        return cls(
            operator=RuntimeVideoOperator(generation_type="t2v"),
            model_path=model_path,
            load_options=load_options,
            memory_module=VideoArtifactMemory(model_id=cls.MODEL_ID),
            device=device,
            lazy=lazy,
        )

    def _ensure_runtime_pipeline(self) -> Any:
        """Ensure runtime pipeline for HunyuanVideoT2VPipeline."""
        if self.runtime_pipeline is not None:
            return self.runtime_pipeline
        if _is_official_checkpoint(self.model_path) and not _is_diffusers_checkpoint(self.model_path):
            self.runtime_pipeline = OfficialVideoRuntime.from_model_id(
                self.MODEL_ID,
                device=self.device,
                checkpoint_path=str(Path(str(self.model_path)).expanduser()),
            )
            return self.runtime_pipeline
        # Dynamically load target module to prevent eager-import overhead
        module = importlib.import_module(
            "worldfoundry.base_models.diffusion_model.video.hunyuan_video.diffusion.pipelines.pipeline_hunyuan_video"
        )
        runtime_cls = getattr(module, "HunyuanVideoPipeline")
        self.runtime_pipeline = runtime_cls.from_pretrained(self.model_path, **self.load_options)
        if hasattr(self.runtime_pipeline, "to"):
            self.runtime_pipeline = self.runtime_pipeline.to(self.device)
        return self.runtime_pipeline

    def process(self, prompt: str, images: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        if images is not None:
            raise ValueError("HunyuanVideoT2VPipeline is text-to-video and does not accept image conditioning.")
        self.operator.get_interaction(prompt)
        try:
            interaction = self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()
        return {"prompt": interaction["processed_prompt"], "kwargs": dict(kwargs)}

    def __call__(
        self,
        prompt: str,
        images: Any = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute the complete pipeline generation flow."""
        processed = self.process(prompt=prompt, images=images, **kwargs)
        runtime = self._ensure_runtime_pipeline()
        if isinstance(runtime, OfficialVideoRuntime):
            target = Path(output_path or f"tmp/pipeline_eval/{self.MODEL_ID}.mp4").expanduser().resolve()
            result = runtime.generate(
                prompt=processed["prompt"],
                output_path=target,
                **processed["kwargs"],
            )
            if return_dict:
                return {
                    **result,
                    "generated_video_path": result.get("artifact_path"),
                    "model_name": self.MODEL_ID,
                    "generation_type": "t2v",
                }
            return result.get("artifact_path") or result
        result = runtime(prompt=processed["prompt"], **processed["kwargs"])
        video = _frames_to_uint8_array(getattr(result, "videos", result))
        save_path = None
        if output_path is not None:
            save_path = Path(output_path).expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            import imageio

            imageio.mimsave(str(save_path), video, fps=int(kwargs.get("fps") or 16))
        if return_dict:
            return {
                "video": video,
                "prompt": processed["prompt"],
                "generated_video_path": str(save_path) if save_path is not None else None,
                "model_name": self.MODEL_ID,
                "generation_type": "t2v",
            }
        return video

    def stream(self, prompt: str, **kwargs: Any) -> Any:
        """Stream visual generation outputs chunk by chunk."""
        result = self(prompt=prompt, return_dict=True, **kwargs)
        self.memory_module.record(
            result["video"],
            metadata={"prompt": prompt, "model_name": self.MODEL_ID, "generation_type": "t2v"},
        )
        return result["video"]

    def get_operator(self) -> RuntimeVideoOperator:
        """Get operator for HunyuanVideoT2VPipeline."""
        return self.operator

    def get_synthesis_model(self) -> Any:
        """Get synthesis model for HunyuanVideoT2VPipeline."""
        return self.runtime_pipeline


__all__ = ["HunyuanVideoT2VPipeline"]
