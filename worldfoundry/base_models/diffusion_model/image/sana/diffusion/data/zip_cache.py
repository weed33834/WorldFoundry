# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal zip/json helpers retained for VAE cache loading during inference."""

from __future__ import annotations

import json
from functools import lru_cache
from zipfile import ZipFile


@lru_cache(maxsize=16)
def open_zip_file(path: str) -> ZipFile:
    return ZipFile(path, "r")


@lru_cache(maxsize=16)
def lru_json_load(fpath: str):
    with open(fpath, encoding="utf-8") as fp:
        return json.load(fp)
