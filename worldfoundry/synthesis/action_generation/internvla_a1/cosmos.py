# SPDX-License-Identifier: Apache-2.0
"""Minimal NVIDIA Cosmos CI8x8 TorchScript encoder used by InternVLA-A1."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class CosmosImageTokenizer(nn.Module):
    """Inference-only wrapper around the official staged CI8x8 JIT assets."""

    def __init__(
        self,
        encoder_checkpoint: str | Path,
        decoder_checkpoint: str | Path,
        *,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        encoder_path = Path(encoder_checkpoint).expanduser().resolve()
        decoder_path = Path(decoder_checkpoint).expanduser().resolve()
        if not encoder_path.is_file():
            raise FileNotFoundError(f"Cosmos image tokenizer encoder is missing: {encoder_path}")
        if not decoder_path.is_file():
            raise FileNotFoundError(f"Cosmos image tokenizer decoder is missing: {decoder_path}")
        # Attribute names intentionally match the official checkpoint state keys.
        self._enc_model = torch.jit.load(str(encoder_path), map_location=device).eval()
        self._dec_model = torch.jit.load(str(decoder_path), map_location=device).eval()
        self._enc_model = self._enc_model.to(device=device, dtype=dtype)
        self._dec_model = self._dec_model.to(device=device, dtype=dtype)
        self.dtype = dtype

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        original_dtype = images.dtype
        output = self._enc_model(images.to(dtype=self.dtype))
        if not isinstance(output, torch.Tensor):
            output = output[0]
        return output.to(dtype=original_dtype)

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        original_dtype = latents.dtype
        output = self._dec_model(latents.to(dtype=self.dtype))
        if not isinstance(output, torch.Tensor):
            output = output[0]
        return output.to(dtype=original_dtype)


__all__ = ["CosmosImageTokenizer"]
