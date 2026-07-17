"""Infinite World visual generation pipeline module."""

from ..pipeline_utils import PipelineABC
import math
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image

from ...synthesis.visual_generation.memory.stream import VisualFrameMemory
from ...operators.infinite_world_operator import InfiniteWorldOperator
from ...synthesis.visual_generation.infinite_world.infinite_world_synthesis import InfiniteWorldSynthesis


class InfiniteWorldPipeline(PipelineABC):
    """Infinite-World pipeline for action-conditioned long-horizon video generation."""

    def __init__(
        self,
        operators: Optional[InfiniteWorldOperator] = None,
        synthesis_model: Optional[InfiniteWorldSynthesis] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        weight_dtype=None,
    ):
        """Initialize the pipeline and configure runtime components."""
        self.operators = operators or InfiniteWorldOperator()
        self.synthesis_model = synthesis_model
        self.memory_module = memory_module or VisualFrameMemory(model_id="infinite-world")
        self.device = device
        self.weight_dtype = weight_dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        required_components: Optional[dict] = None,
        device: str = "cuda",
        weight_dtype=None,
        **kwargs,
    ) -> "InfiniteWorldPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        synthesis_model = InfiniteWorldSynthesis.from_pretrained(
            pretrained_model_path=model_path,
            device=device,
            weight_dtype=weight_dtype,
            **kwargs,
        )
        return cls(
            operators=InfiniteWorldOperator(),
            synthesis_model=synthesis_model,
            memory_module=VisualFrameMemory(model_id="infinite-world"),
            device=device,
            weight_dtype=weight_dtype,
        )

    def _resolve_num_chunks(
        self,
        num_chunks: Optional[int],
        num_frames: Optional[int],
        action_count: int,
        condition_frames: int,
    ) -> int:
        """Resolve num chunks for InfiniteWorldPipeline."""
        if num_chunks is not None:
            return max(int(num_chunks), 1)
        # ``chunk_stride`` is measured in VAE latent frames (21 latent frames
        # with one-frame overlap => a stride of 20).  The public ``num_frames``
        # argument and action sequence are measured in decoded video frames.
        # One runtime chunk decodes ``validation_num_frames`` frames and appends
        # all but its overlapping first frame.
        output_stride = max(int(self.synthesis_model.validation_num_frames) - 1, 1)
        if num_frames is not None:
            if num_frames <= condition_frames:
                return 1
            return max(
                1,
                math.ceil((int(num_frames) - int(condition_frames)) / output_stride),
            )
        if action_count <= self.synthesis_model.validation_num_frames:
            return 1
        extra_actions = action_count - self.synthesis_model.validation_num_frames
        return 1 + math.ceil(extra_actions / output_stride)

    @staticmethod
    def _trim_result_frames(result: Dict[str, Any], num_frames: int) -> Dict[str, Any]:
        """Trim a generated chunk sequence to the public decoded-frame target."""
        target = max(int(num_frames), 1)
        video = result.get("video")
        if video is not None:
            result["video"] = video[:target]
        video_uint8 = result.get("video_uint8")
        if video_uint8 is not None:
            result["video_uint8"] = video_uint8[:target]
        video_tensor = result.get("video_tensor")
        if video_tensor is not None:
            result["video_tensor"] = video_tensor[:, :, :target]
        result["num_frames"] = min(int(result.get("num_frames", target)), target)
        return result

    def process(
        self,
        images,
        interactions: Sequence[Union[str, Dict[str, str]]],
        prompt: str = "",
    ):
        """Process and normalize input arguments and conditions for inference."""
        if self.synthesis_model is None:
            raise RuntimeError("Synthesis model is not loaded. Use from_pretrained() first.")

        perception = self.operators.process_perception(
            images,
            bucket_config=self.synthesis_model.bucket_config,
        )
        prefix_length = max(int(perception["num_condition_frames"]) - 1, 0)

        self.operators.get_interaction(interactions)
        try:
            operator_condition = self.operators.process_interaction(prefix_length=prefix_length)
        finally:
            self.operators.delete_last_interaction()

        return {
            "condition_video": perception["condition_video"],
            "target_size": perception["target_size"],
            "num_condition_frames": perception["num_condition_frames"],
            "operator_condition": operator_condition,
            "prompt": prompt or "",
        }

    def __call__(
        self,
        images,
        interactions: Sequence[Union[str, Dict[str, str]]],
        prompt: str = "",
        num_chunks: Optional[int] = None,
        num_frames: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        return_dict: bool = False,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        if not interactions:
            raise ValueError("interactions must be provided for Infinite-World generation.")

        output_dict = self.process(
            images=images,
            interactions=interactions,
            prompt=prompt,
        )
        resolved_num_chunks = self._resolve_num_chunks(
            num_chunks=num_chunks,
            num_frames=num_frames,
            action_count=len(output_dict["operator_condition"]["actions"]),
            condition_frames=output_dict["num_condition_frames"],
        )

        result = self.synthesis_model.predict(
            prompt=output_dict["prompt"],
            condition_video=output_dict["condition_video"],
            move_ids=output_dict["operator_condition"]["move_ids"],
            view_ids=output_dict["operator_condition"]["view_ids"],
            num_chunks=resolved_num_chunks,
            negative_prompt=negative_prompt,
            seed=seed,
            **kwargs,
        )
        # Chunked generation can only produce multiples of the decoded output
        # stride.  Honour ``num_frames`` exactly unless the caller explicitly
        # requested a chunk count, in which case full chunks are intentional.
        if num_chunks is None and num_frames is not None:
            result = self._trim_result_frames(result, num_frames)
        if return_dict:
            return result
        return result["video"]

    def stream(
        self,
        images: Optional[Union[Image.Image, Any]],
        interactions: Sequence[Union[str, Dict[str, str]]],
        prompt: str = "",
        num_chunks: Optional[int] = None,
        num_frames: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        reset_memory: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        """Stream visual generation outputs chunk by chunk."""
        if reset_memory:
            self.memory_module.manage(action="reset")

        if images is not None:
            self.memory_module.record(images, metadata={"prompt": prompt, "mode": "init"})

        current_state = self.memory_module.select()
        if current_state is None:
            raise ValueError("No state in memory. Provide 'images' on the first stream turn.")

        result = self.__call__(
            images=current_state,
            interactions=interactions,
            prompt=prompt,
            num_chunks=num_chunks,
            num_frames=num_frames,
            negative_prompt=negative_prompt,
            seed=seed,
            return_dict=True,
            **kwargs,
        )
        self.memory_module.record(
            result,
            metadata={"prompt": prompt, "interactions": list(interactions)},
        )

        if return_dict:
            return result
        return result["video"]
