"""Module for base_models -> diffusion_model -> video -> hunyuan_video -> utils -> data_utils.py functionality."""

import numpy as np
import math


def align_to(value, alignment):
    """align hight, width according to alignment

    Args:
        value (int): height or width
        alignment (int): target alignment factor

    Returns:
        int: the aligned value
    """
    return int(math.ceil(value / alignment) * alignment)
