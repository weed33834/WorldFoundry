"""Lightweight shared HunyuanVideo I2V modulation layers."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ModulateDiT(nn.Module):
    """Modulation layer for DiT."""

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: Callable,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(
            hidden_size, factor * hidden_size, bias=True, **factory_kwargs
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, condition_type=None, token_replace_vec=None) -> torch.Tensor:
        x_out = self.linear(self.act(x))

        if condition_type == "token_replace":
            x_token_replace_out = self.linear(self.act(token_replace_vec))
            return x_out, x_token_replace_out
        return x_out


def modulate(x, shift=None, scale=None, condition_type=None, tr_shift=None, tr_scale=None, frist_frame_token_num=None):
    """Apply shift/scale modulation."""

    if condition_type == "token_replace":
        x_zero = x[:, :frist_frame_token_num] * (1 + tr_scale.unsqueeze(1)) + tr_shift.unsqueeze(1)
        x_orig = x[:, frist_frame_token_num:] * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return torch.concat((x_zero, x_orig), dim=1)
    if scale is None and shift is None:
        return x
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    if scale is None:
        return x + shift.unsqueeze(1)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_gate(x, gate=None, tanh=False, condition_type=None, tr_gate=None, frist_frame_token_num=None):
    """Apply modulation gates."""

    if condition_type == "token_replace":
        if gate is None:
            return x
        if tanh:
            x_zero = x[:, :frist_frame_token_num] * tr_gate.unsqueeze(1).tanh()
            x_orig = x[:, frist_frame_token_num:] * gate.unsqueeze(1).tanh()
        else:
            x_zero = x[:, :frist_frame_token_num] * tr_gate.unsqueeze(1)
            x_orig = x[:, frist_frame_token_num:] * gate.unsqueeze(1)
        return torch.concat((x_zero, x_orig), dim=1)
    if gate is None:
        return x
    if tanh:
        return x * gate.unsqueeze(1).tanh()
    return x * gate.unsqueeze(1)


def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        return module(*inputs)

    return ckpt_forward
