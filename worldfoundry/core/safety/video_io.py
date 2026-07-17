# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video I/O shared by content-safety filters and postprocessors."""

from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np

from worldfoundry.core.distributed.logging import log


@dataclass(frozen=True)
class VideoData:
    frames: np.ndarray
    fps: float
    duration: float


def get_video_filepaths(input_dir: str | Path) -> list[str]:
    root = Path(input_dir)
    paths = sorted(str(path) for suffix in ("*.mp4", "*.avi", "*.mov") for path in root.rglob(suffix))
    log.debug("Found {} videos", len(paths))
    return paths


def read_video(filepath: str | Path) -> VideoData:
    path = str(filepath)
    try:
        reader = imageio.get_reader(path, "ffmpeg")
    except Exception as error:
        raise ValueError(f"Failed to read video file: {path}") from error

    try:
        metadata = reader.get_meta_data()
        try:
            frame_count = reader.count_frames()
        except (AttributeError, NotImplementedError, RuntimeError):
            frame_count = round(float(metadata.get("duration", 0.0)) * float(metadata.get("fps", 0.0)))
        if frame_count <= 0:
            raise ValueError("video contains no decodable frames")
        frames = np.stack([reader.get_data(index) for index in range(frame_count)])
        return VideoData(
            frames=frames,
            fps=float(metadata.get("fps", 0.0)),
            duration=float(metadata.get("duration", 0.0)),
        )
    except Exception as error:
        raise ValueError(f"Failed to decode video file: {path}") from error
    finally:
        reader.close()


def save_video(filepath: str | Path, frames: np.ndarray, fps: float) -> None:
    path = str(filepath)
    writer = None
    try:
        writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
        for frame in frames:
            writer.append_data(frame)
    except Exception as error:
        raise ValueError(f"Failed to save video file to {path}") from error
    finally:
        if writer is not None:
            writer.close()
