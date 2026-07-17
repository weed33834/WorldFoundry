"""Diffusers-compatible wrapper around the canonical Wan CLIP encoder."""

from __future__ import annotations

import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from torch.nn import functional as F

from worldfoundry.base_models.diffusion_model.video.wan.wan_2p1.modules.clip import (
    clip_xlm_roberta_vit_h_14,
)
from worldfoundry.core.model_loading import load_model


class CLIPModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    def __init__(self) -> None:
        super().__init__()
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
        )

    def forward(self, videos: list[torch.Tensor]) -> torch.Tensor:
        size = (self.model.image_size, self.model.image_size)
        frames = torch.cat(
            [
                F.interpolate(
                    video.transpose(0, 1),
                    size=size,
                    mode="bicubic",
                    align_corners=False,
                )
                for video in videos
            ]
        )
        frames = self.transforms.transforms[-1](frames.mul_(0.5).add_(0.5))
        return self.model.visual(frames.to(dtype=self.dtype), use_31_block=True)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        transformer_additional_kwargs: dict | None = None,
    ) -> "CLIPModel":
        del transformer_additional_kwargs
        return load_model(
            cls,
            pretrained_model_path,
            torch_dtype=torch.float32,
            device="cpu",
            state_dict_converter=lambda state: {
                (key if key.startswith("model.") else f"model.{key}"): value
                for key, value in state.items()
            },
        )


__all__ = ["CLIPModel"]
