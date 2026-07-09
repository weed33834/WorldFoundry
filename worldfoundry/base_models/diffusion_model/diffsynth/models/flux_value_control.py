"""Module for base_models -> diffusion_model -> diffsynth -> models -> flux_value_control.py functionality."""

import torch
from .svd_unet import TemporalTimesteps


class MultiValueEncoder(torch.nn.Module):
    """Multi value encoder implementation."""
    def __init__(self, encoders=()):
        """Init.

        Args:
            encoders: The encoders.
        """
        super().__init__()
        self.encoders = torch.nn.ModuleList(encoders)

    def __call__(self, values, dtype):
        """Call.

        Args:
            values: The values.
            dtype: The dtype.
        """
        emb = []
        for encoder, value in zip(self.encoders, values):
            if value is not None:
                value = value.unsqueeze(0)
                emb.append(encoder(value, dtype))
        emb = torch.concat(emb, dim=0)
        return emb


class SingleValueEncoder(torch.nn.Module):
    """Single value encoder implementation."""
    def __init__(self, dim_in=256, dim_out=4096, prefer_len=32, computation_device=None):
        """Init.

        Args:
            dim_in: The dim in.
            dim_out: The dim out.
            prefer_len: The prefer len.
            computation_device: The computation device.
        """
        super().__init__()
        self.prefer_len = prefer_len
        self.prefer_proj = TemporalTimesteps(num_channels=dim_in, flip_sin_to_cos=True, downscale_freq_shift=0, computation_device=computation_device)
        self.prefer_value_embedder = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out), torch.nn.SiLU(), torch.nn.Linear(dim_out, dim_out)
        )
        self.positional_embedding = torch.nn.Parameter(
            torch.randn(self.prefer_len, dim_out) 
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Helper function to initialize weights."""
        last_linear = self.prefer_value_embedder[-1]
        torch.nn.init.zeros_(last_linear.weight)
        torch.nn.init.zeros_(last_linear.bias)

    def forward(self, value, dtype):
        """Forward.

        Args:
            value: The value.
            dtype: The dtype.
        """
        value = value * 1000
        emb = self.prefer_proj(value).to(dtype)
        emb = self.prefer_value_embedder(emb).squeeze(0)
        base_embeddings = emb.expand(self.prefer_len, -1)
        positional_embedding = self.positional_embedding.to(dtype=base_embeddings.dtype, device=base_embeddings.device)
        learned_embeddings = base_embeddings + positional_embedding
        return learned_embeddings

    @staticmethod
    def state_dict_converter():
        """State dict converter."""
        return SingleValueEncoderStateDictConverter()


class SingleValueEncoderStateDictConverter:
    """Single value encoder state dict converter implementation."""
    def __init__(self):
        """Init."""
        pass

    def from_diffusers(self, state_dict):
        """From diffusers.

        Args:
            state_dict: The state dict.
        """
        return state_dict

    def from_civitai(self, state_dict):
        """From civitai.

        Args:
            state_dict: The state dict.
        """
        return state_dict
