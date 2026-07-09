"""In-tree package alias for the vendored DB-CogACT runtime.

The upstream Dexbotic code imports modules as ``dexbotic.*``. WorldFoundry keeps
the inference subset under ``db_cogact_runtime``; extending this package path
lets those imports resolve without depending on an external checkout.
"""

from __future__ import annotations

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent)]
