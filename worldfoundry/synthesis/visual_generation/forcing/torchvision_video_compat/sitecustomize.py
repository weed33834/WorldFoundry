"""Compatibility patch for official forcing runtimes on new torchvision.

Self-Forcing and Causal-Forcing official scripts still import
``torchvision.io.write_video``. Recent torchvision builds removed that helper,
so this sitecustomize module reinstates the narrow video-only behavior required
by those scripts before their module graph imports torchvision.io.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy_video(video_array: Any) -> np.ndarray:
    if hasattr(video_array, "detach"):
        video_array = video_array.detach().cpu().numpy()
    array = np.asarray(video_array)
    if array.ndim != 4:
        raise ValueError(f"write_video expects a 4D video array, got shape {array.shape!r}")
    if array.shape[1] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 1, -1)
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 1.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _install_write_video() -> None:
    try:
        import torchvision.io as tv_io
    except Exception:
        return
    if hasattr(tv_io, "write_video"):
        return

    def write_video(
        filename: str,
        video_array: Any,
        fps: float,
        video_codec: str = "libx264",
        options: dict[str, Any] | None = None,
        audio_array: Any | None = None,
        audio_fps: float | None = None,
        audio_codec: str | None = None,
    ) -> None:
        del audio_array, audio_fps, audio_codec
        import imageio.v2 as imageio

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer_options = dict(options or {})
        if video_codec:
            writer_options.setdefault("codec", video_codec)
        with imageio.get_writer(str(path), fps=fps, **writer_options) as writer:
            for frame in _to_numpy_video(video_array):
                writer.append_data(frame)

    tv_io.write_video = write_video


def _install_flash_attention_fallback() -> None:
    """Route official Wan flash_attention calls to SDPA when flash-attn is absent."""

    try:
        from wan.modules import attention as attention_module
    except Exception:
        return

    if getattr(attention_module, "FLASH_ATTN_2_AVAILABLE", False) or getattr(
        attention_module, "FLASH_ATTN_3_AVAILABLE", False
    ):
        return

    def flash_attention_fallback(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=None,
        version=None,
    ):
        del version
        dtype = dtype if dtype is not None else q.dtype
        return attention_module.attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            fa_version=None,
        )

    attention_module.flash_attention = flash_attention_fallback
    try:
        from wan.modules import model as model_module
    except Exception:
        return
    model_module.flash_attention = flash_attention_fallback


if not Path(sys.argv[0]).name.startswith("pip"):
    _install_write_video()
    _install_flash_attention_fallback()
