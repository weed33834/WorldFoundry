"""Shared Cosmos Predict2 source package."""
from __future__ import annotations

import sys

from worldfoundry.base_models.diffusion_model.video.cosmos.shared.about import __version__ as __version__

sys.modules["cosmos_predict2"] = sys.modules[__name__]


def _check_cuda_extra():
    """Check if CUDA extra is installed."""
    try:
        import cosmos_cuda
    except ImportError:
        raise RuntimeError("CUDA extra not installed. Please run 'uv sync --extra=<cuda_name>'") from None

    if __version__ != cosmos_cuda.__version__:
        raise RuntimeError(
            f"CUDA extra version mismatch: {cosmos_cuda.__version__} != {__version__}. Please run 'uv sync --extra=<cuda_name>'"
        )


__all__ = ["__version__", "_check_cuda_extra"]
