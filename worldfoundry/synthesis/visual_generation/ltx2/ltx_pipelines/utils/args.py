from typing import NamedTuple

from worldfoundry.synthesis.visual_generation.ltx2.ltx_pipelines.utils.constants import DEFAULT_IMAGE_CRF


class ImageConditioningInput(NamedTuple):
    path: str
    frame_idx: int
    strength: float
    crf: int = DEFAULT_IMAGE_CRF
