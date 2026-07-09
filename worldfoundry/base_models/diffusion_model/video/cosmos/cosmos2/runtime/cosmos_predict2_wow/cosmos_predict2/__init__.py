"""WoW Cosmos Predict2 fork package."""
from __future__ import annotations

import sys


sys.modules["cosmos_predict2"] = sys.modules[__name__]

__all__: list[str] = []
