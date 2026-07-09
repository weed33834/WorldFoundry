# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os


def get_most_recent_checkpoint(output_dir: str) -> str | None:
    """
    Returns the most recent checkpoint directory from the given output directory.

    Args:
        output_dir: Path to the directory containing checkpoints.

    Returns:
        The name of the most recent checkpoint directory, or None if none exist.
    """
    dirs = os.listdir(output_dir)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    return dirs[-1] if dirs else None
