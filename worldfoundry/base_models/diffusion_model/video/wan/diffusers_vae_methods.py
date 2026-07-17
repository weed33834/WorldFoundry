"""Diffusers encode/decode bridge shared by canonical Wan VAEs."""

from __future__ import annotations

import torch
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils.accelerate_utils import apply_forward_hook


class WanDiffusersVAEMixin:
    @apply_forward_hook
    def encode(self, value: torch.Tensor, return_dict: bool = True):
        encoded = torch.stack(
            [
                self.model.encode(item.unsqueeze(0), self.scale).squeeze(0)
                for item in value
            ]
        )
        moments = torch.cat((encoded, torch.zeros_like(encoded)), dim=1)
        posterior = DiagonalGaussianDistribution(moments, deterministic=True)
        if return_dict:
            return AutoencoderKLOutput(latent_dist=posterior)
        return (posterior,)

    @apply_forward_hook
    def decode(self, value: torch.Tensor, return_dict: bool = True):
        decoded = torch.stack(
            [
                self.model.decode(item.unsqueeze(0), self.scale)
                .clamp_(-1, 1)
                .squeeze(0)
                for item in value
            ]
        )
        if return_dict:
            return DecoderOutput(sample=decoded)
        return (decoded,)


__all__ = ["WanDiffusersVAEMixin"]
