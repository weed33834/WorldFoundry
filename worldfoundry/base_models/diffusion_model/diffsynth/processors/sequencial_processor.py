"""Module for base_models -> diffusion_model -> diffsynth -> processors -> sequencial_processor.py functionality."""

from .base import VideoProcessor


class AutoVideoProcessor(VideoProcessor):
    """Auto video processor implementation."""
    def __init__(self):
        """Init."""
        pass

    @staticmethod
    def from_model_manager(model_manager, processor_type, **kwargs):
        """From model manager.

        Args:
            model_manager: The model manager.
            processor_type: The processor type.
        """
        if processor_type == "FastBlend":
            from .FastBlend import FastBlendSmoother
            return FastBlendSmoother.from_model_manager(model_manager, **kwargs)
        elif processor_type == "Contrast":
            from .PILEditor import ContrastEditor
            return ContrastEditor.from_model_manager(model_manager, **kwargs)
        elif processor_type == "Sharpness":
            from .PILEditor import SharpnessEditor
            return SharpnessEditor.from_model_manager(model_manager, **kwargs)
        elif processor_type == "RIFE":
            from .RIFE import RIFESmoother
            return RIFESmoother.from_model_manager(model_manager, **kwargs)
        else:
            raise ValueError(f"invalid processor_type: {processor_type}")


class SequencialProcessor(VideoProcessor):
    """Sequencial processor implementation."""
    def __init__(self, processors=[]):
        """Init.

        Args:
            processors: The processors.
        """
        self.processors = processors

    @staticmethod
    def from_model_manager(model_manager, configs):
        """From model manager.

        Args:
            model_manager: The model manager.
            configs: The configs.
        """
        processors = [
            AutoVideoProcessor.from_model_manager(model_manager, config["processor_type"], **config["config"])
            for config in configs
        ]
        return SequencialProcessor(processors)
    
    def __call__(self, rendered_frames, **kwargs):
        """Call.

        Args:
            rendered_frames: The rendered frames.
        """
        for processor in self.processors:
            rendered_frames = processor(rendered_frames, **kwargs)
        return rendered_frames
