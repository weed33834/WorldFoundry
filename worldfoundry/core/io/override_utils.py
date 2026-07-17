"""Lightweight command-line configuration formatting helpers."""

from __future__ import annotations


def format_override_value(value) -> str:
    """Format a Python value for a Hydra command-line override."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(format_override_value(item) for item in value) + "]"
    if value == "":
        return "''"
    return str(value)


__all__ = ["format_override_value"]
