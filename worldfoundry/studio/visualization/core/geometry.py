"""Small geometry helpers shared by Studio visualization exporters."""

from __future__ import annotations

from typing import Any


def depth_to_world_points(depth: Any, intrinsics: Any, pose: Any) -> Any:
    """Unproject a depth image with an OpenCV camera-to-world transform."""

    import numpy as np

    depth = np.asarray(depth)
    intrinsics = np.asarray(intrinsics)
    pose = np.asarray(pose)
    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    camera_points = np.stack(
        (
            (x - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (y - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ),
        axis=-1,
    )
    return camera_points @ pose[:3, :3].T + pose[:3, 3]
