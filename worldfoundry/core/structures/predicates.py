"""Lightweight predicates for Python container structures."""

from collections.abc import Mapping, Sequence
from typing import Any


def is_sequence(value: Any) -> bool:
    """Return whether *value* is a non-string sequence."""

    return isinstance(value, Sequence) and not isinstance(value, str)


def is_mapping(value: Any) -> bool:
    """Return whether *value* implements the mapping protocol."""

    return isinstance(value, Mapping)


__all__ = ["is_mapping", "is_sequence"]
