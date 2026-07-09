"""Canonical FlowFormer++ optical-flow runtime."""

from __future__ import annotations

from .core.FlowFormer import build_flowformer
from .configs.submissions import get_cfg
from .paths import checkpoint_path

__all__ = ["build_flowformer", "checkpoint_path", "get_cfg"]
