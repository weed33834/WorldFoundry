"""Utilities required by the retained inference paths."""

from .group_offload import register_auto_device_hook, safe_enable_group_offload

__all__ = ["register_auto_device_hook", "safe_enable_group_offload"]
