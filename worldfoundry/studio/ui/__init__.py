"""WorldFoundry Studio UI assets and HTML fragments."""

from __future__ import annotations

from .assets import (
    SPARK_MODULE_PATH,
    SPARK_ROOT,
    STUDIO_ASSET_DIR,
    THREE_CORE_MODULE_PATH,
    THREE_MODULE_PATH,
    THREE_ROOT,
    VENDOR_DIR,
    local_module_url,
)
from .css import CUSTOM_CSS
from .head import HEAD_HTML
from .html import hero_html, profile_html, summary_html
from .urls import file_url

__all__ = [
    "CUSTOM_CSS",
    "HEAD_HTML",
    "SPARK_MODULE_PATH",
    "SPARK_ROOT",
    "STUDIO_ASSET_DIR",
    "THREE_CORE_MODULE_PATH",
    "THREE_MODULE_PATH",
    "THREE_ROOT",
    "VENDOR_DIR",
    "hero_html",
    "file_url",
    "local_module_url",
    "profile_html",
    "summary_html",
]
