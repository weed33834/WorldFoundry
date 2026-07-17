"""Small helpers for releasing process-local inference runtime caches."""

from __future__ import annotations

import gc
from collections.abc import MutableMapping
from typing import Any


def clear_inference_runtime_cache(cache: MutableMapping[Any, Any]) -> None:
    """Drop cached runtimes and release unreferenced accelerator allocations.

    The torch import is intentionally lazy so model discovery remains cheap in
    processes that never execute a PyTorch policy.  ``empty_cache`` only returns
    already-unreferenced blocks to the CUDA allocator; clearing the strong
    references and collecting cycles must happen first.
    """

    cache.clear()
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = ["clear_inference_runtime_cache"]
