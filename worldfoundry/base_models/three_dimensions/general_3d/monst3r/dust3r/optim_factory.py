# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# optimization functions
# --------------------------------------------------------


"""Module for base_models -> three_dimensions -> general_3d -> monst3r -> dust3r -> optim_factory.py functionality."""

def adjust_learning_rate_by_lr(optimizer, lr):
    """Adjust learning rate by lr.

    Args:
        optimizer: The optimizer.
        lr: The lr.
    """
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
