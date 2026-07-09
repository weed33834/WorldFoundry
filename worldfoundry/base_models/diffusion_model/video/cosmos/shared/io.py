# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for base_models -> diffusion_model -> video -> cosmos -> shared -> io.py functionality."""

from typing import Dict, List

from worldfoundry.core.io import dump_serialized, load_serialized


def read_prompts_from_file(prompt_file: str) -> List[Dict[str, str]]:
    """Read prompts from a JSONL file where each line is a dict with 'prompt' key and optionally 'visual_input' key.

    Args:
        prompt_file (str): Path to JSONL file containing prompts

    Returns:
        List[Dict[str, str]]: List of prompt dictionaries
    """
    return load_serialized(prompt_file, file_format="jsonl")


def save_video(video, fps, H, W, video_save_quality, video_save_path):
    """Save video frames to file.

    Args:
        grid (np.ndarray): Video frames array [T,H,W,C]
        fps (int): Frames per second
        H (int): Frame height
        W (int): Frame width
        video_save_quality (int): Video encoding quality (0-10)
        video_save_path (str): Output video file path
    """
    dump_serialized(
        video,
        video_save_path,
        file_format="mp4",
        fps=fps,
        quality=video_save_quality,
        ffmpeg_params=["-s", f"{W}x{H}"],
        output_params=["-f", "mp4"],
    )


def load_from_fileobj(filepath: str, format: str = "mp4", mode: str = "rgb", **kwargs):
    """
    Load video from a file-like object using imageio with specified format and color mode.

    Parameters:
        file (IO[bytes]): A file-like object containing video data.
        format (str): Format of the video file (default 'mp4').
        mode (str): Color mode of the video, 'rgb' or 'gray' (default 'rgb').

    Returns:
        tuple: A tuple containing an array of video frames and metadata about the video.
    """
    return load_serialized(filepath, file_format=format, mode=mode, **kwargs)
