"""Vendored Open-MAGVIT2 runtime package."""

from __future__ import annotations

import sys

from . import src as _src

sys.modules.setdefault("open_magvit2_runtime", sys.modules[__name__])
sys.modules.setdefault("src", _src)
