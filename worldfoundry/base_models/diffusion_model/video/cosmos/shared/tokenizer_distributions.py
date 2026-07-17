# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The distribution modes to use for continuous image tokenizers."""

import torch


class IdentityDistribution(torch.nn.Module):
    """Identity distribution implementation."""

    def __init__(self):
        """Init."""
        super().__init__()

    def forward(self, parameters):
        """Forward.

        Args:
            parameters: The parameters.
        """
        return parameters, (torch.tensor([0.0]), torch.tensor([0.0]))


class GaussianDistribution(torch.nn.Module):
    """Gaussian distribution implementation."""

    def __init__(self, min_logvar: float = -30.0, max_logvar: float = 20.0):
        """Init.

        Args:
            min_logvar: The min logvar.
            max_logvar: The max logvar.
        """
        super().__init__()
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def sample(self, mean, logvar):
        """Sample.

        Args:
            mean: The mean.
            logvar: The logvar.
        """
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)

    def forward(self, parameters):
        """Forward.

        Args:
            parameters: The parameters.
        """
        mean, logvar = torch.chunk(parameters, 2, dim=1)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return self.sample(mean, logvar), (mean, logvar)
