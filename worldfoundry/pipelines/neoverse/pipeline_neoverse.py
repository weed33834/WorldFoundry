"""Neoverse visual generation pipeline module."""

from __future__ import annotations

from ..pipeline_utils import PipelineABC
from typing import Any, Optional, Sequence

import torch

from ...synthesis.visual_generation.memory.stream import VisualFrameMemory
from ...operators.neoverse_operator import NeoVerseOperator


def _neoverse_synthesis_cls():
    """Neoverse synthesis cls helper function."""
    from ...synthesis.visual_generation.neoverse.neoverse_synthesis import NeoVerseSynthesis

    return NeoVerseSynthesis


class NeoVersePipeline(PipelineABC):
    """NeoVerse pipeline adapted to the WorldBench navigation interface."""

    def __init__(
        self,
        operator: Optional[NeoVerseOperator] = None,
        synthesis_model: Optional[Any] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the pipeline and configure runtime components."""
        self.operator = operator or NeoVerseOperator()
        self.synthesis_model = synthesis_model
        self.memory_module = memory_module or VisualFrameMemory(model_id="neoverse")
        self.device = device
        self.weight_dtype = weight_dtype

    @staticmethod
    def _maybe_visual_input(
        images: Optional[Any] = None,
        *,
        videos: Optional[Any] = None,
        video: Optional[Any] = None,
        video_path: Optional[Any] = None,
        input_path: Optional[Any] = None,
    ) -> Any:
        """Maybe visual input for NeoVersePipeline."""
        for candidate in (images, videos, video, video_path, input_path):
            if candidate is not None:
                return candidate
        return None

    @classmethod
    def _resolve_visual_input(
        cls,
        images: Optional[Any] = None,
        *,
        videos: Optional[Any] = None,
        video: Optional[Any] = None,
        video_path: Optional[Any] = None,
        input_path: Optional[Any] = None,
    ) -> Any:
        """Resolve visual input for NeoVersePipeline."""
        visual_input = cls._maybe_visual_input(
            images,
            videos=videos,
            video=video,
            video_path=video_path,
            input_path=input_path,
        )
        if visual_input is not None:
            return visual_input
        raise ValueError("NeoVerse requires an image or video input.")

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = "Yuppie1204/NeoVerse",
        required_components: Optional[dict] = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "NeoVersePipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        component_options = dict(required_components or {})
        if isinstance(model_path, dict):
            component_options.update(model_path)
            model_path = component_options.pop(
                "model_path",
                component_options.pop(
                    "pretrained_model_path",
                    component_options.pop("repo_id", None),
                ),
            )
        runtime_options = cls._strip_framework_loading_options({**component_options, **kwargs})
        if model_path is None:
            model_path = runtime_options.pop("repo_id", "Yuppie1204/NeoVerse")

        operator_kwargs = {}
        for key in [
            "height",
            "width",
            "resize_mode",
            "frames_per_action",
            "translation_distance",
            "rotation_angle_deg",
            "zoom_ratio",
            "trajectory_mode",
        ]:
            if key in runtime_options:
                operator_kwargs[key] = runtime_options.pop(key)

        synthesis_model = _neoverse_synthesis_cls().from_pretrained(
            pretrained_model_path=model_path,
            device=device,
            weight_dtype=weight_dtype,
            **runtime_options,
        )
        operator_kwargs.setdefault("height", synthesis_model.height)
        operator_kwargs.setdefault("width", synthesis_model.width)

        return cls(
            operator=NeoVerseOperator(**operator_kwargs),
            synthesis_model=synthesis_model,
            memory_module=VisualFrameMemory(model_id="neoverse"),
            device=device,
            weight_dtype=weight_dtype,
        )

    def process(
        self,
        *,
        images,
        interactions: Optional[Sequence[str]] = None,
        predefined_trajectory: Optional[str] = None,
        trajectory_file: Optional[str] = None,
        trajectory_data=None,
        prompt: str = "",
        static_scene: Optional[bool] = None,
        zoom_ratio: Optional[float] = None,
        trajectory_mode: Optional[str] = None,
        trajectory_name: Optional[str] = None,
        angle: Optional[float] = None,
        distance: Optional[float] = None,
        orbit_radius: Optional[float] = None,
        use_first_frame: bool = True,
        num_frames: Optional[int] = None,
    ):
        """Process and normalize input arguments and conditions for inference."""
        if self.synthesis_model is None:
            raise RuntimeError("Synthesis model is not loaded. Use from_pretrained() first.")

        perception = self.operator.process_perception(
            images,
            height=self.synthesis_model.height,
            width=self.synthesis_model.width,
            static_scene=static_scene,
            num_frames=num_frames,
        )
        interaction_spec = {
            "actions": list(interactions) if interactions is not None else None,
            "predefined_trajectory": predefined_trajectory,
            "trajectory_file": trajectory_file,
            "trajectory_data": trajectory_data,
            "num_frames": num_frames,
            "zoom_ratio": self.operator.zoom_ratio if zoom_ratio is None else zoom_ratio,
            "trajectory_mode": trajectory_mode or self.operator.trajectory_mode,
            "trajectory_name": trajectory_name or predefined_trajectory or "neoverse_actions",
            "angle": angle,
            "distance": distance,
            "orbit_radius": orbit_radius,
            "use_first_frame": use_first_frame,
        }
        self.operator.get_interaction(interaction_spec)
        try:
            interaction = self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()

        return {
            "prompt": prompt or "",
            "input_frames": perception["input_frames"],
            "static_scene": perception["static_scene"],
            "actions": interaction["actions"],
            "predefined_trajectory": interaction["predefined_trajectory"],
            "keyframes": interaction["keyframes"],
            "trajectory_file": interaction["trajectory_file"],
            "trajectory_data": interaction["trajectory_data"],
            "num_frames": interaction["num_frames"],
            "trajectory_mode": interaction["trajectory_mode"],
            "trajectory_name": interaction["trajectory_name"],
            "zoom_ratio": interaction["zoom_ratio"],
            "angle": interaction["angle"],
            "distance": interaction["distance"],
            "orbit_radius": interaction["orbit_radius"],
            "use_first_frame": interaction["use_first_frame"],
        }

    def __call__(
        self,
        images: Optional[Any] = None,
        interactions: Optional[Sequence[str]] = None,
        *,
        videos: Optional[Any] = None,
        video: Optional[Any] = None,
        video_path: Optional[Any] = None,
        input_path: Optional[Any] = None,
        predefined_trajectory: Optional[str] = None,
        trajectory_file: Optional[str] = None,
        trajectory_data=None,
        prompt: str = "",
        static_scene: Optional[bool] = None,
        zoom_ratio: Optional[float] = None,
        trajectory_mode: Optional[str] = None,
        trajectory_name: Optional[str] = None,
        angle: Optional[float] = None,
        distance: Optional[float] = None,
        orbit_radius: Optional[float] = None,
        use_first_frame: bool = True,
        num_frames: Optional[int] = None,
        return_dict: bool = False,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        visual_input = self._maybe_visual_input(
            images,
            videos=videos,
            video=video,
            video_path=video_path,
            input_path=input_path,
        )
        if (
            interactions is None
            and predefined_trajectory is None
            and trajectory_file is None
            and trajectory_data is None
        ):
            raise ValueError("NeoVerse requires interactions or an explicit trajectory specification.")

        processed = self.process(
            images=visual_input,
            interactions=interactions,
            predefined_trajectory=predefined_trajectory,
            trajectory_file=trajectory_file,
            trajectory_data=trajectory_data,
            prompt=prompt,
            static_scene=static_scene,
            zoom_ratio=zoom_ratio,
            trajectory_mode=trajectory_mode,
            trajectory_name=trajectory_name,
            angle=angle,
            distance=distance,
            orbit_radius=orbit_radius,
            use_first_frame=use_first_frame,
            num_frames=num_frames,
        )
        result = self.synthesis_model.predict(
            images=processed["input_frames"],
            prompt=processed["prompt"],
            keyframes=processed["keyframes"],
            predefined_trajectory=processed["predefined_trajectory"],
            trajectory_file=processed["trajectory_file"],
            trajectory_data=processed["trajectory_data"],
            num_frames=processed["num_frames"],
            trajectory_mode=processed["trajectory_mode"],
            trajectory_name=processed["trajectory_name"],
            zoom_ratio=processed["zoom_ratio"],
            angle=processed["angle"],
            distance=processed["distance"],
            orbit_radius=processed["orbit_radius"],
            use_first_frame=processed["use_first_frame"],
            static_scene=processed["static_scene"],
            return_dict=True,
            **kwargs,
        )
        result.update(
            {
                "actions": processed["actions"],
            }
        )
        if return_dict:
            return result
        return result["video"]

    def stream(
        self,
        images: Optional[Any] = None,
        interactions: Optional[Sequence[str]] = None,
        *,
        videos: Optional[Any] = None,
        video: Optional[Any] = None,
        video_path: Optional[Any] = None,
        input_path: Optional[Any] = None,
        trajectory_file: Optional[str] = None,
        trajectory_data=None,
        prompt: str = "",
        static_scene: Optional[bool] = None,
        reset_memory: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        """Stream visual generation outputs chunk by chunk."""
        if reset_memory:
            self.memory_module.manage(action="reset")

        visual_input = self._resolve_visual_input(
            images,
            videos=videos,
            video=video,
            video_path=video_path,
            input_path=input_path,
        )
        if visual_input is not None:
            self.memory_module.record(visual_input, metadata={"mode": "init", "prompt": prompt})

        current_image = self.memory_module.select()
        if current_image is None:
            raise ValueError("No state in memory. Provide `images` on the first stream turn.")

        result = self.__call__(
            images=current_image,
            interactions=interactions,
            trajectory_file=trajectory_file,
            trajectory_data=trajectory_data,
            prompt=prompt,
            static_scene=True if static_scene is None else static_scene,
            return_dict=True,
            **kwargs,
        )
        self.memory_module.record(
            result["video"],
            metadata={
                "prompt": prompt,
                "interactions": list(interactions) if interactions is not None else None,
            },
        )

        if return_dict:
            return result
        return result["video"]
