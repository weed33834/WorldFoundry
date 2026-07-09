"""Path helpers for the in-tree ViT human anatomy detector."""

from __future__ import annotations

from pathlib import Path


def config_path() -> Path:
    return Path(__file__).resolve().parent / "simmim_finetune__vit_base__img224__800ep.yaml"
