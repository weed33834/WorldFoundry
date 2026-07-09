from __future__ import annotations


def status_block(message: str) -> str:
    return f"<div class='wa-status'>{message}</div>"


__all__ = ["status_block"]
