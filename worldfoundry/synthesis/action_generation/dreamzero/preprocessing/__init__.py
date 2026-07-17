"""DreamZero observation transforms required for checkpoint inference."""

from pydantic import BaseModel

from .embodiment import EmbodimentTag
from .schema import DatasetMetadata
from .transform_base import ComposedModalityTransform, InvertibleModalityTransform, ModalityTransform
from .transform_concat import ConcatTransform
from .transform_state_action import PerHorizonActionTransform, StateActionToTensor, StateActionTransform
from .transform_video import (
    VideoColorJitter,
    VideoCrop,
    VideoFocusRect,
    VideoGrayscale,
    VideoHorizontalFlip,
    VideoNormalize,
    VideoRandomErasing,
    VideoRandomGrayscale,
    VideoRandomPosterize,
    VideoRandomRotation,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from .dream_transform import DreamTransform


class ModalityConfig(BaseModel):
    delta_indices: list[int]
    eval_delta_indices: list[int] | None = None
    modality_keys: list[str]

    def model_post_init(self, __context) -> None:
        if self.eval_delta_indices is None:
            self.eval_delta_indices = self.delta_indices


__all__ = [
    "ComposedModalityTransform",
    "ConcatTransform",
    "DatasetMetadata",
    "DreamTransform",
    "EmbodimentTag",
    "ModalityConfig",
    "PerHorizonActionTransform",
    "StateActionToTensor",
    "StateActionTransform",
    "VideoColorJitter",
    "VideoCrop",
    "VideoFocusRect",
    "VideoGrayscale",
    "VideoHorizontalFlip",
    "VideoNormalize",
    "VideoRandomErasing",
    "VideoRandomGrayscale",
    "VideoRandomPosterize",
    "VideoRandomRotation",
    "VideoResize",
    "VideoToNumpy",
    "VideoToTensor",
]
