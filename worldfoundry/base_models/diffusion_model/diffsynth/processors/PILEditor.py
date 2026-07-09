"""Module for base_models -> diffusion_model -> diffsynth -> processors -> PILEditor.py functionality."""

from PIL import ImageEnhance
from .base import VideoProcessor


class ContrastEditor(VideoProcessor):
    """Contrast editor implementation."""
    def __init__(self, rate=1.5):
        """Init.

        Args:
            rate: The rate.
        """
        self.rate = rate

    @staticmethod
    def from_model_manager(model_manager, **kwargs):
        """From model manager.

        Args:
            model_manager: The model manager.
        """
        return ContrastEditor(**kwargs)
    
    def __call__(self, rendered_frames, **kwargs):
        """Call.

        Args:
            rendered_frames: The rendered frames.
        """
        rendered_frames = [ImageEnhance.Contrast(i).enhance(self.rate) for i in rendered_frames]
        return rendered_frames


class SharpnessEditor(VideoProcessor):
    """Sharpness editor implementation."""
    def __init__(self, rate=1.5):
        """Init.

        Args:
            rate: The rate.
        """
        self.rate = rate

    @staticmethod
    def from_model_manager(model_manager, **kwargs):
        """From model manager.

        Args:
            model_manager: The model manager.
        """
        return SharpnessEditor(**kwargs)
    
    def __call__(self, rendered_frames, **kwargs):
        """Call.

        Args:
            rendered_frames: The rendered frames.
        """
        rendered_frames = [ImageEnhance.Sharpness(i).enhance(self.rate) for i in rendered_frames]
        return rendered_frames
