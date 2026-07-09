"""Vendored Show-O runtime package."""

from __future__ import annotations

import sys

from . import inference_support as _inference_support
from . import models as _models

sys.modules.setdefault("models", _models)
sys.modules.setdefault("inference_support", _inference_support)
