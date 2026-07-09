"""Module for base_models -> three_dimensions -> general_3d -> stable_virtual_camera -> stable_virtual_camera_runtime -> seva -> modules -> autoencoder.py functionality."""

import torch
from diffusers.models import AutoencoderKL  # type: ignore
from torch import nn


class AutoEncoder(nn.Module):
    """Auto encoder implementation."""
    scale_factor: float = 0.18215
    downsample: int = 8

    def __init__(self, chunk_size: int | None = None):
        """Init.

        Args:
            chunk_size: The chunk size.
        """
        super().__init__()
        self.module = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            subfolder="vae",
            force_download=False,
            low_cpu_mem_usage=False,
        )
        self.module.eval().requires_grad_(False)  # type: ignore
        self.chunk_size = chunk_size

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function to encode.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        return (
            self.module.encode(x).latent_dist.mean  # type: ignore
            * self.scale_factor
        )

    def encode(self, x: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        """Encode.

        Args:
            x: The x.
            chunk_size: The chunk size.

        Returns:
            The return value.
        """
        chunk_size = chunk_size or self.chunk_size
        if chunk_size is not None:
            return torch.cat(
                [self._encode(x_chunk) for x_chunk in x.split(chunk_size)],
                dim=0,
            )
        else:
            return self._encode(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Helper function to decode.

        Args:
            z: The z.

        Returns:
            The return value.
        """
        return self.module.decode(z / self.scale_factor).sample  # type: ignore

    def decode(self, z: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        """Decode.

        Args:
            z: The z.
            chunk_size: The chunk size.

        Returns:
            The return value.
        """
        chunk_size = chunk_size or self.chunk_size
        if chunk_size is not None:
            return torch.cat(
                [self._decode(z_chunk) for z_chunk in z.split(chunk_size)],
                dim=0,
            )
        else:
            return self._decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        return self.decode(self.encode(x))
