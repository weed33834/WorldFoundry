"""Module for base_models -> diffusion_model -> video -> wan -> variants -> krea_realtime -> wan -> __init__.py functionality."""

from . import configs, distributed, modules
from .image2video import WanI2V
from .text2video import WanT2V
