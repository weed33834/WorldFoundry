"""Token constants used by the in-tree TinyVLA prompt path."""

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

__all__ = [
    "DEFAULT_IMAGE_PATCH_TOKEN",
    "DEFAULT_IMAGE_TOKEN",
    "DEFAULT_IM_END_TOKEN",
    "DEFAULT_IM_START_TOKEN",
    "IGNORE_INDEX",
    "IMAGE_TOKEN_INDEX",
]
