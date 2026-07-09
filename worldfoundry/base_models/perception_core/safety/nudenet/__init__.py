"""NudeNet ONNX detector used by safety benchmarks."""

from .nudenet import NudeDetector, model_path

__all__ = ["NudeDetector", "model_path"]
