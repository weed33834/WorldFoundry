# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Module for base_models -> perception_core -> general_perception -> dinov3 -> hub -> utils.py functionality."""

import torch


_DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"


def _safe_load_state_dict_from_url(url: str, **kwargs):
    """Helper function to safe load state dict from url.

    Args:
        url: The url.
    """
    if torch.__version__ >= (2, 1):
        local_kwargs = {**kwargs, "weights_only": True}
    else:
        local_kwargs = kwargs
    return torch.hub.load_state_dict_from_url(url, **local_kwargs)
