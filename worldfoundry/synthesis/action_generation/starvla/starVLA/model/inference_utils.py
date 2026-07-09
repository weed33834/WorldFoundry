from __future__ import annotations

import logging
from contextlib import nullcontext
from logging import LoggerAdapter
from typing import Any, Callable, ClassVar, MutableMapping

from PIL import Image


class _ContextAdapter(LoggerAdapter):
    _PREFIXES: ClassVar[dict[int, str]] = {0: "[*] ", 1: "    |=> ", 2: "        |=> ", 3: "            |=> "}

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:
        ctx_level = int(kwargs.pop("ctx_level", 0))
        return f"{self._PREFIXES.get(ctx_level, '')}{msg}", kwargs


class _InferenceLogger:
    def __init__(self, name: str) -> None:
        self.logger = _ContextAdapter(logging.getLogger(name), extra={})
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

    @staticmethod
    def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        return self._identity

    @property
    def local_zero_only(self) -> Callable[..., Any]:
        return self._identity

    @property
    def rank_zero_first(self):
        return nullcontext

    @property
    def local_zero_first(self):
        return nullcontext

    @staticmethod
    def is_rank_zero() -> bool:
        return True

    @staticmethod
    def rank() -> int:
        return 0

    @staticmethod
    def local_rank() -> int:
        return 0

    @staticmethod
    def world_size() -> int:
        return 1


def initialize_overwatch(name: str) -> _InferenceLogger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    return _InferenceLogger(name)


def resize_images(images, target_size=(224, 224)):
    if isinstance(images, Image.Image):
        return images.resize(target_size)
    if isinstance(images, list):
        return [resize_images(item, target_size) for item in images]
    raise ValueError("Unsupported image type or structure.")
