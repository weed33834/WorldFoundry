# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module for base_models -> diffusion_model -> video -> cosmos -> shared -> omegaconf_patch.py functionality."""

from typing import Any, Dict, List, Union

from omegaconf import OmegaConf
from omegaconf.base import DictKeyType, SCMode
from omegaconf.dictconfig import DictConfig  # pragma: no cover


def to_object(cfg: Any) -> Union[Dict[DictKeyType, Any], List[Any], None, str, Any]:
    """
    Converts an OmegaConf configuration object to a native Python container (dict or list), unless
    the configuration is specifically created by LazyCall, in which case the original configuration
    is returned directly.
    """
    if isinstance(cfg, DictConfig) and "_target_" in cfg.keys():
        return cfg

    return OmegaConf.to_container(
        cfg=cfg,
        resolve=True,
        throw_on_missing=True,
        enum_to_str=False,
        structured_config_mode=SCMode.INSTANTIATE,
    )
