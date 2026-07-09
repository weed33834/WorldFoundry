"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos1 -> cosmos_predict1_gen3c -> cosmos_predict1 -> __init__.py functionality."""

from __future__ import annotations

import sys
from pathlib import Path

from worldfoundry.base_models.diffusion_model.video.cosmos.shared.predict1_imports import (
    ensure_predict1_import_path,
)


SOURCE_ROOT = Path(__file__).resolve().parent.parent


def ensure_import_paths() -> str:
    """Expose this upstream Cosmos Predict1 tree as the top-level package."""
    return ensure_predict1_import_path(__file__, sys.modules[__name__], alias_top_level=True)


IMPORT_PATH = ensure_import_paths()


__all__ = ["IMPORT_PATH", "SOURCE_ROOT", "ensure_import_paths"]
