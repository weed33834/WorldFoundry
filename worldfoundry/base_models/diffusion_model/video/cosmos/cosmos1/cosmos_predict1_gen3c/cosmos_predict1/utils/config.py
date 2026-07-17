# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for the shared WorldFoundry inference configuration."""

from worldfoundry.core.configuration import Config, make_freezable

__all__ = ["Config", "make_freezable"]
