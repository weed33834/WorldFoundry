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

"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos2 -> runtime -> cosmos_predict2 -> cosmos_predict2 -> _src -> imaginaire -> checkpointer -> dummy.py functionality."""

from typing import Optional

import torch
import torch.distributed

from cosmos_predict2._src.imaginaire.checkpointer.base import AbstractCheckpointer
from cosmos_predict2._src.imaginaire.model import ImaginaireModel


class Checkpointer(AbstractCheckpointer):
    """
    A dummy checkpointer that does not save or load anything. This is useful for debugging jobs or share workload with collobrators.
    """

    def save(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save.

        Args:
            model: The model.
            optimizer: The optimizer.
            scheduler: The scheduler.
            grad_scaler: The grad scaler.
            iteration: The iteration.

        Returns:
            The return value.
        """
        pass

    def load(
        self,
        model: ImaginaireModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        grad_scaler: Optional[torch.amp.GradScaler] = None,
    ) -> int:
        """Load.

        Args:
            model: The model.
            optimizer: The optimizer.
            scheduler: The scheduler.
            grad_scaler: The grad scaler.

        Returns:
            The return value.
        """
        return 0
