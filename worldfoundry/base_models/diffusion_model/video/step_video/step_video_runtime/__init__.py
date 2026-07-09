"""Vendored Step-Video-T2V runtime package."""

from __future__ import annotations

import sys

from . import benchmark as _benchmark
from . import stepvideo as _stepvideo

sys.modules.setdefault("benchmark", _benchmark)
sys.modules.setdefault("stepvideo", _stepvideo)
