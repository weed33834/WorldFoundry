# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# lr schedules for sparse ga
# --------------------------------------------------------
"""Module for base_models -> three_dimensions -> general_3d -> mast3r -> mast3r -> cloud_opt -> utils -> schedules.py functionality."""

import numpy as np


def linear_schedule(alpha, lr_base, lr_end=0):
    """Linear schedule.

    Args:
        alpha: The alpha.
        lr_base: The lr base.
        lr_end: The lr end.
    """
    lr = (1 - alpha) * lr_base + alpha * lr_end
    return lr


def cosine_schedule(alpha, lr_base, lr_end=0):
    """Cosine schedule.

    Args:
        alpha: The alpha.
        lr_base: The lr base.
        lr_end: The lr end.
    """
    lr = lr_end + (lr_base - lr_end) * (1 + np.cos(alpha * np.pi)) / 2
    return lr
