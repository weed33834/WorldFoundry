"""Static TensorRT cache metadata for the Matrix-Game 2 VAE decoder.

The original integration materialised every zero tensor at module import time.
Those tensors total roughly two gigabytes on CPU, even though the PyTorch
streaming decoder only needs a list of ``None`` values.  Keep the legacy
sequence API for the optional TensorRT wrapper, but allocate an individual
zero tensor only if that wrapper actually iterates over the sequence.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch


_BASE_W = 80
_BASE_H = 44
VAE_CACHE_SHAPES: tuple[tuple[int, ...], ...] = (
    (1, 16, 2, _BASE_H, _BASE_W),
    *((1, 384, 2, _BASE_H, _BASE_W),) * 11,
    (1, 192, 2, _BASE_H * 2, _BASE_W * 2),
    *((1, 384, 2, _BASE_H * 2, _BASE_W * 2),) * 6,
    *((1, 192, 2, _BASE_H * 4, _BASE_W * 4),) * 6,
    *((1, 96, 2, _BASE_H * 8, _BASE_W * 8),) * 7,
)


class _LazyZeroCache(Sequence[torch.Tensor]):
    """Read-only, allocation-free cache template with legacy list semantics."""

    def __len__(self) -> int:
        return len(VAE_CACHE_SHAPES)

    def __getitem__(self, index: int | slice) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(index, slice):
            return [torch.zeros(shape) for shape in VAE_CACHE_SHAPES[index]]
        return torch.zeros(VAE_CACHE_SHAPES[index])

    def __iter__(self) -> Iterator[torch.Tensor]:
        return (torch.zeros(shape) for shape in VAE_CACHE_SHAPES)


ZERO_VAE_CACHE: Sequence[torch.Tensor] = _LazyZeroCache()
feat_names = [f"vae_cache_{i}" for i in range(len(VAE_CACHE_SHAPES))]
ALL_INPUTS_NAMES = ["z", "use_cache", *feat_names]
