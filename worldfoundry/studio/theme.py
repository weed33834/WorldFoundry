"""Public Studio theme facade.

The implementation lives under :mod:`worldfoundry.studio.ui` so JS/CSS/assets can
be maintained independently from the Gradio app assembly code.
"""

from __future__ import annotations

from .ui import (
    CUSTOM_CSS,
    HEAD_HTML,
    SPARK_MODULE_PATH,
    SPARK_ROOT,
    STUDIO_ASSET_DIR,
    THREE_CORE_MODULE_PATH,
    THREE_MODULE_PATH,
    THREE_ROOT,
    VENDOR_DIR,
    hero_html,
    local_module_url,
    profile_html,
    summary_html,
)

_local_module_url = local_module_url

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
    "local_module_url",
    "profile_html",
    "summary_html",
]
