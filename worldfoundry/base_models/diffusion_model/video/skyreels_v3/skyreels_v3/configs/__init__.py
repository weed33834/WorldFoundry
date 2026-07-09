# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Module for base_models -> diffusion_model -> video -> skyreels_v3 -> skyreels_v3 -> configs -> __init__.py functionality."""

import copy
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .talking_avatar_19B import talking_avatar_19B

WAN_CONFIGS = {
    "talking-avatar-19B": talking_avatar_19B,
}
