from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


try:
    from torchvision.io import write_video as _torchvision_write_video
except Exception:  # pragma: no cover - depends on torchvision build features.
    _torchvision_write_video = None


def _to_numpy(video_array: Any) -> np.ndarray:
    if isinstance(video_array, torch.Tensor):
        array = video_array.detach().cpu().numpy()
    else:
        array = np.asarray(video_array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def write_video(
    filename: str | Path,
    video_array: Any,
    fps: int | float,
    video_codec: str | None = None,
    options: dict[str, str] | None = None,
) -> None:
    """Torchvision-compatible video writer with an imageio fallback."""
    if _torchvision_write_video is not None:
        _torchvision_write_video(
            str(filename),
            video_array,
            fps=fps,
            video_codec=video_codec,
            options=options,
        )
        return

    import imageio.v3 as iio

    del video_codec, options
    iio.imwrite(str(filename), _to_numpy(video_array), fps=fps)
