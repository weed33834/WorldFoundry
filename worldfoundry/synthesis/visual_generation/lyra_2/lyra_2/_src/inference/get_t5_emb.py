from __future__ import annotations

import os
from typing import List, Optional, Union

import torch

from worldfoundry.base_models.diffusion_model.video.cosmos.cosmos2.runtime.cosmos_predict2.cosmos_predict2._src.predict2.inference.get_umt5_emb import (
    UMT5EncoderModel,
)

_t5_encoder: Optional[UMT5EncoderModel] = None
_t5_offloaded: Optional[UMT5EncoderModel] = None


def _build_encoder(*, device: str, text_len: int, load_on_cpu: bool) -> UMT5EncoderModel:
    return UMT5EncoderModel(
        text_len=text_len,
        device=device,
        checkpoint_path=os.environ.get(
            "LYRA2_TEXT_ENCODER_CKPT",
            "./checkpoints/text_encoder/encoder.pth",
        ),
        tokenizer_path=os.environ.get("LYRA2_UMT5_TOKENIZER", "google/umt5-xxl"),
        credential_path=None,
        load_on_cpu=load_on_cpu,
    )


def get_umt5_embedding(
    prompts: Union[str, List[str]],
    device: str = "cuda",
    max_length: int = 512,
) -> torch.Tensor:
    global _t5_encoder
    if _t5_encoder is None:
        _t5_encoder = _build_encoder(device=device, text_len=max_length, load_on_cpu=False)
    return _t5_encoder(prompts, device=device)


@torch.no_grad()
def get_umt5_embedding_offloaded(
    prompts: Union[str, List[str]],
    device: str = "cuda",
    max_length: int = 512,
) -> torch.Tensor:
    global _t5_offloaded
    if _t5_offloaded is None:
        _t5_offloaded = _build_encoder(device="cpu", text_len=max_length, load_on_cpu=True)

    _t5_offloaded.model.to(device)
    _t5_offloaded.device = device
    try:
        return _t5_offloaded(prompts, device=device)
    finally:
        _t5_offloaded.model.to("cpu")
        _t5_offloaded.device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
