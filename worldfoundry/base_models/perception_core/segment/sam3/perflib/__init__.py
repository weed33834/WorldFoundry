# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Module for base_models -> perception_core -> segment -> sam3 -> perflib -> __init__.py functionality."""

import os

is_enabled = False
if os.getenv("USE_PERFLIB", "1") == "1":
    # print("Enabled the use of perflib.\n", end="")
    is_enabled = True
