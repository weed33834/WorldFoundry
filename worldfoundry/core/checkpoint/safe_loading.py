"""Fail-closed helpers for inference-time PyTorch checkpoint loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any, BinaryIO

import torch

DEFAULT_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model", "module")


def load_weights_only(
    source: str | PathLike[str] | BinaryIO,
    *,
    map_location: Any = "cpu",
    mmap: bool | None = None,
) -> object:
    """Deserialize only tensors and PyTorch's allowlisted primitive containers."""

    kwargs: dict[str, Any] = {
        "map_location": map_location,
        "weights_only": True,
    }
    if mmap is not None:
        kwargs["mmap"] = mmap
    return torch.load(source, **kwargs)


def require_tensor(payload: object, *, source: str) -> torch.Tensor:
    """Return one tensor payload or reject the checkpoint shape."""

    if not torch.is_tensor(payload):
        raise TypeError(f"Checkpoint {source!r} must contain a tensor, got {type(payload).__name__}")
    return payload


def require_mapping(payload: object, *, source: str) -> Mapping[str, object]:
    """Return a string-keyed checkpoint mapping or reject its shape."""

    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise TypeError(f"Checkpoint {source!r} must contain a string-keyed mapping")
    return payload


def tensor_state_dict(
    payload: object,
    *,
    source: str,
    wrapper_keys: Sequence[str] = DEFAULT_STATE_DICT_KEYS,
    allow_empty: bool = False,
) -> dict[str, torch.Tensor]:
    """Extract and validate a flat string-to-tensor state dictionary."""

    candidates: list[object] = [payload]
    if isinstance(payload, Mapping):
        candidates.extend(payload[key] for key in wrapper_keys if key in payload)
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or (not candidate and not allow_empty):
            continue
        if all(isinstance(key, str) and torch.is_tensor(value) for key, value in candidate.items()):
            return dict(candidate)
    raise TypeError(
        f"Checkpoint {source!r} does not contain a "
        "flat string-to-tensor state dictionary"
    )


def load_tensor_state_dict(
    source: str | PathLike[str] | BinaryIO,
    *,
    map_location: Any = "cpu",
    wrapper_keys: Sequence[str] = DEFAULT_STATE_DICT_KEYS,
    mmap: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Weights-only load followed by strict tensor-state validation."""

    payload = load_weights_only(source, map_location=map_location, mmap=mmap)
    return tensor_state_dict(
        payload,
        source=str(getattr(source, "name", source)),
        wrapper_keys=wrapper_keys,
    )


__all__ = [
    "DEFAULT_STATE_DICT_KEYS",
    "load_tensor_state_dict",
    "load_weights_only",
    "require_mapping",
    "require_tensor",
    "tensor_state_dict",
]
