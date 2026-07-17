"""Output serialization for VideoX-Fun inference scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .utils import save_videos_grid


def save_inference_sample(
    sample,
    output_dir: str | Path,
    *,
    video_length: int,
    fps: int,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{len(list(output_dir.iterdir())) + 1:08d}"
    if video_length == 1:
        output_path = output_dir / f"{prefix}.png"
        image = sample[0, :, 0].permute(1, 2, 0)
        image = image.mul(255).cpu().numpy().astype(np.uint8)
        Image.fromarray(image).save(output_path)
    else:
        output_path = output_dir / f"{prefix}.mp4"
        save_videos_grid(sample, str(output_path), fps=fps)
    return output_path


__all__ = ["save_inference_sample"]
