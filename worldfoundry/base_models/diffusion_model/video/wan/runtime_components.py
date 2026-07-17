"""Shared resident Wan 2.1 text and VAE inference components."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .vae.camera_wan2p1 import _video_vae
from .wan_2p1.modules.t5 import umt5_xxl
from .wan_2p1.modules.tokenizers import HuggingfaceTokenizer

WAN_2P1_VAE_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
WAN_2P1_VAE_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


class WanTextEncoder(nn.Module):
    def __init__(
        self,
        model_root: str | Path | None = None,
        *,
        text_encoder_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if model_root is not None:
            root = Path(model_root)
            text_encoder_path = text_encoder_path or (
                root / "models_t5_umt5-xxl-enc-bf16.pth"
            )
            tokenizer_path = tokenizer_path or (root / "google" / "umt5-xxl")
        if text_encoder_path is None or tokenizer_path is None:
            raise ValueError(
                "WanTextEncoder requires model_root or explicit encoder/tokenizer paths."
            )
        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=torch.device("cpu"),
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(
                text_encoder_path,
                map_location="cpu",
                weights_only=False,
            )
        )
        self.tokenizer = HuggingfaceTokenizer(
            name=str(tokenizer_path),
            seq_len=512,
            clean="whitespace",
        )

    @property
    def device(self) -> torch.device:
        return next(self.text_encoder.parameters()).device

    def forward(self, text_prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        ids, mask = self.tokenizer(
            list(text_prompts),
            return_mask=True,
            add_special_tokens=True,
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        sequence_lengths = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        for embedding, length in zip(context, sequence_lengths):
            embedding[length:] = 0.0
        return {"prompt_embeds": context}


class WanVAEWrapper(nn.Module):
    def __init__(
        self,
        model_root: str | Path | None = None,
        *,
        vae_path: str | Path | None = None,
        mean: Sequence[float] = WAN_2P1_VAE_MEAN,
        std: Sequence[float] = WAN_2P1_VAE_STD,
        z_dim: int = 16,
        temporal_downsample: Sequence[bool] | None = None,
        model_factory=None,
    ) -> None:
        super().__init__()
        if vae_path is None:
            if model_root is None:
                raise ValueError("WanVAEWrapper requires model_root or vae_path.")
            vae_path = Path(model_root) / "Wan2.1_VAE.pth"
        self.register_buffer(
            "latent_mean", torch.tensor(tuple(mean)), persistent=False
        )
        self.register_buffer(
            "latent_inverse_std",
            1.0 / torch.tensor(tuple(std)),
            persistent=False,
        )
        kwargs = {"pretrained_path": str(vae_path), "z_dim": int(z_dim)}
        if temporal_downsample is not None:
            kwargs["temperal_downsample"] = list(temporal_downsample)
        self.model = (model_factory or _video_vae)(**kwargs).eval().requires_grad_(False)

    def _scale(self, value: torch.Tensor) -> list[torch.Tensor]:
        return [
            self.latent_mean.to(value),
            self.latent_inverse_std.to(value),
        ]

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        scale = self._scale(pixel)
        encoded = torch.stack(
            [
                self.model.encode(item.unsqueeze(0), scale)
                .float()
                .squeeze(0)
                for item in pixel
            ]
        )
        return encoded.permute(0, 2, 1, 3, 4)

    def encode_to_latent_cached(self, pixel: torch.Tensor) -> torch.Tensor:
        encoded = self.model.cached_encode(pixel, self._scale(pixel)).float()
        return encoded.permute(0, 2, 1, 3, 4)

    def decode_to_pixel(
        self,
        latent: torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        if use_cache and latent.shape[0] != 1:
            raise ValueError("Cached Wan VAE decoding requires batch size 1.")
        decode = self.model.cached_decode if use_cache else self.model.decode
        scale = self._scale(latent)
        decoded = torch.stack(
            [
                decode(item.unsqueeze(0), scale)
                .float()
                .clamp_(-1, 1)
                .squeeze(0)
                for item in latent.permute(0, 2, 1, 3, 4)
            ]
        )
        return decoded.permute(0, 2, 1, 3, 4)
