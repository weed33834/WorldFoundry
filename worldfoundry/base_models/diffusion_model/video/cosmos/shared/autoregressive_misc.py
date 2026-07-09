# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module for base_models -> diffusion_model -> video -> cosmos -> shared -> autoregressive_misc.py functionality."""

import torch
from omegaconf import DictConfig, OmegaConf


class CustomSimpleNamespace:
    """A namespace that supports attribute-style and dictionary-style access."""

    def __init__(self, d):
        """Init.

        Args:
            d: The d.
        """
        self._d = d

    def __getattr__(self, attr):
        """Getattr.

        Args:
            attr: The attr.
        """
        try:
            return self._d[attr]
        except KeyError:
            raise AttributeError(f"'CustomSimpleNamespace' object has no attribute '{attr}'")

    def __getitem__(self, key):
        """Getitem.

        Args:
            key: The key.
        """
        return self._d[key]


def maybe_convert_to_namespace(config):
    """Maybe convert to namespace.

    Args:
        config: The config.
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    if isinstance(config, dict):
        return CustomSimpleNamespace(config)
    return config


def random_dropout(embeddings, drop_rate):
    """Random dropout.

    Args:
        embeddings: The embeddings.
        drop_rate: The drop rate.
    """
    num_samples = embeddings.shape[0]
    tensor_shape = (num_samples,) + tuple([1] * (embeddings.ndim - 1))
    zero_flag = torch.ones(tensor_shape).to(embeddings.dtype) * (1 - drop_rate)
    zero_flag = torch.bernoulli(zero_flag).to(embeddings.device)
    embeddings = embeddings * zero_flag
    return embeddings
