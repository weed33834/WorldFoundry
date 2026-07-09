"""Lazy exports for the canonical Wan 2.2 S2V module package."""

from __future__ import annotations

from importlib import import_module

__all__ = ["WanModel_S2V", "AudioEncoder"]

_LAZY_EXPORTS = {
    "AudioEncoder": ".audio_encoder",
    "WanModel_S2V": ".model_s2v",
}


def __getattr__(name: str):
    """Getattr.

    Args:
        name: The name.
    """
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
