"""Independent in-tree VideoX-Fun camera-model integrations."""

from .video_x_fun_synthesis import (
    Wan21Fun1P3BCameraSynthesis,
    Wan21Fun14BCameraSynthesis,
    Wan22Fun5BCameraSynthesis,
    Wan22FunA14BCameraSynthesis,
)
from .worldfoundry_runtime import (
    Wan21Fun1P3BCameraRuntime,
    Wan21Fun14BCameraRuntime,
    Wan22Fun5BCameraRuntime,
    Wan22FunA14BCameraRuntime,
)

__all__ = [
    "Wan21Fun1P3BCameraRuntime",
    "Wan21Fun1P3BCameraSynthesis",
    "Wan21Fun14BCameraRuntime",
    "Wan21Fun14BCameraSynthesis",
    "Wan22Fun5BCameraRuntime",
    "Wan22Fun5BCameraSynthesis",
    "Wan22FunA14BCameraRuntime",
    "Wan22FunA14BCameraSynthesis",
]
