"""Gen3C visual generation pipeline module."""

from __future__ import annotations

from ..pipeline_utils import PipelineABC
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch
from PIL import Image

from ...synthesis.visual_generation.memory.stream import VisualFrameMemory
from ...operators.gen3c_operator import Gen3COperator
from ...synthesis.visual_generation.gen3c.gen3c_synthesis import Gen3CSynthesis


class Gen3CPipeline(PipelineABC):
    """GEN3C pipeline for single-image camera-controlled video generation."""

    def __init__(
        self,
        synthesis_model: Optional[Gen3CSynthesis] = None,
        operator: Optional[Gen3COperator] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype=torch.bfloat16,
    ):
        """Initialize the pipeline and configure runtime components."""
        self.synthesis_model = synthesis_model
        self.operator = operator or Gen3COperator()
        self.memory_module = memory_module or VisualFrameMemory(model_id="gen3c")
        self.device = device
        self.weight_dtype = weight_dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype=torch.bfloat16,
        **kwargs,
    ) -> "Gen3CPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        required_components = required_components or {}
        synthesis_model = Gen3CSynthesis.from_pretrained(
            pretrained_model_path=model_path,
            checkpoint_dir=required_components.get("checkpoint_dir"),
            moge_path=required_components.get("moge_path"),
            moge_pretrained=required_components.get("moge_pretrained"),
            device=device,
            **{
                key: value
                for key, value in {**required_components, **kwargs}.items()
                if key not in {"repo_root", "checkpoint_dir", "moge_path", "moge_pretrained"}
            },
        )
        return cls(
            synthesis_model=synthesis_model,
            operator=Gen3COperator(),
            memory_module=VisualFrameMemory(model_id="gen3c"),
            device=device,
            weight_dtype=weight_dtype,
        )

    def process(
        self,
        images=None,
        interactions: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
        prompt: str = "",
        trajectory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        if self.synthesis_model is None:
            raise RuntimeError("Synthesis model is not loaded. Use from_pretrained() first.")

        if interactions is None:
            interactions = [trajectory or self.operator.DEFAULT_TRAJECTORY]

        image = self.operator.process_perception(images)
        self.operator.get_interaction(interactions)
        try:
            operator_condition = self.operator.process_interaction(prompt=prompt)
        finally:
            self.operator.delete_last_interaction()

        return {
            "image": image,
            "prompt": prompt or "",
            **operator_condition,
        }

    def __call__(
        self,
        images=None,
        interactions: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
        prompt: str = "",
        trajectory: Optional[str] = None,
        scene_name: str = "gen3c_scene",
        output_dir: Optional[str] = None,
        return_dict: bool = False,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        input_path = kwargs.pop("input_path", None)
        image_path = kwargs.pop("image_path", None)
        if images is None:
            images = input_path if input_path is not None else image_path
        runtime_image = input_path or image_path or images

        processed = self.process(
            images=images,
            interactions=interactions,
            prompt=prompt,
            trajectory=trajectory,
        )

        result = self.synthesis_model.predict(
            image=runtime_image,
            prompt=processed["prompt"],
            trajectory=processed["trajectory"],
            scene_name=scene_name,
            output_dir=output_dir,
            return_dict=True,
            **kwargs,
        )
        result.update(
            {
                "actions": processed["actions"],
                "mapped_trajectories": processed["mapped_trajectories"],
            }
        )
        if return_dict:
            return result
        return result["video"]

    def stream(
        self,
        interactions: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
        images: Optional[Union[Image.Image, Any]] = None,
        prompt: str = "",
        trajectory: Optional[str] = None,
        scene_name: str = "gen3c_scene",
        output_dir: Optional[str] = None,
        reset_memory: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        """Stream visual generation outputs chunk by chunk."""
        input_path = kwargs.pop("input_path", None)
        image_path = kwargs.pop("image_path", None)
        if images is None:
            images = input_path if input_path is not None else image_path

        if reset_memory:
            self.memory_module.manage(action="reset")

        if images is not None:
            self.memory_module.record(self.operator.process_perception(images), metadata={"mode": "init"})

        current_image = self.memory_module.select()
        if current_image is None:
            raise ValueError("No input image found in memory. Provide `images` on the first stream turn.")

        result = self.__call__(
            images=current_image,
            interactions=interactions,
            prompt=prompt,
            trajectory=trajectory,
            scene_name=scene_name,
            output_dir=output_dir,
            return_dict=True,
            **kwargs,
        )
        self.memory_module.record(
            result,
            metadata={
                "prompt": prompt,
                "interactions": list(interactions) if interactions is not None else None,
            },
        )
        if return_dict:
            return result
        return result["video"]
