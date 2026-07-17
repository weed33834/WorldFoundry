"""Checkpoint loading and batched encoding for the FLUX.2 autoencoder."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .autoencoder import AutoEncoder, AutoEncoderParams

FLUX2_REPO_ID = "black-forest-labs/FLUX.2-dev"
FLUX2_AE_FILENAME = "ae.safetensors"


def _resolve_checkpoint(checkpoint_path: str | os.PathLike[str] | None) -> Path:
    configured = (
        checkpoint_path
        or os.environ.get("WORLDFOUNDRY_FLUX2_AE_PATH")
        or os.environ.get("AE_MODEL_PATH")
    )
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"FLUX.2 autoencoder checkpoint not found: {path}")
        return path

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=FLUX2_REPO_ID,
            filename=FLUX2_AE_FILENAME,
            repo_type="model",
        )
    )


def load_autoencoder(
    checkpoint_path: str | os.PathLike[str] | None = None,
    *,
    device: str | torch.device = "cuda",
) -> AutoEncoder:
    """Load the FLUX.2 AE from an explicit path, environment, or HF cache."""

    from safetensors import safe_open

    device = torch.device(device)
    path = _resolve_checkpoint(checkpoint_path)
    with torch.device("meta"):
        model = AutoEncoder(AutoEncoderParams())
    required_keys = set(model.state_dict())
    with safe_open(str(path), framework="pt", device=str(device)) as checkpoint:
        state_dict = {
            key: checkpoint.get_tensor(key)
            for key in checkpoint.keys()
            if key in required_keys
        }
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.to(device).eval()


def encode_video_batch_refs(
    autoencoder: AutoEncoder,
    video_batch: torch.Tensor,
) -> torch.Tensor:
    """Encode ``[B, T, H, W, C]`` normalized frames to ``[B, T, C, h, w]``."""

    if video_batch.ndim != 5:
        raise ValueError(
            f"Expected video_batch with shape [B, T, H, W, C], got {tuple(video_batch.shape)}"
        )
    batch, frames, height, width, channels = video_batch.shape
    if channels != 3:
        raise ValueError(f"Expected three RGB channels, got {channels}")

    images = video_batch.permute(0, 1, 4, 2, 3).reshape(
        batch * frames, channels, height, width
    )
    device = next(autoencoder.parameters()).device
    encoded = autoencoder.encode(images.to(device))
    return encoded.reshape(batch, frames, *encoded.shape[1:])


__all__ = ["encode_video_batch_refs", "load_autoencoder"]
