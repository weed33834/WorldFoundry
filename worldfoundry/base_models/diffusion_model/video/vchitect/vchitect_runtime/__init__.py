"""Module for base_models -> diffusion_model -> video -> vchitect -> vchitect_runtime -> __init__.py functionality."""

from __future__ import annotations

from dataclasses import dataclass

from worldfoundry.runtime.env import resolve_hfd_root

OFFICIAL_REPO_URL = "https://github.com/Vchitect/Vchitect-2.0"
OFFICIAL_REPO_REVISION = "be56767f1107b4fd6ba7652fe92ec0e7032878fe"
OFFICIAL_CHECKPOINT_REPO = "Vchitect/Vchitect-2.0-2B"
OFFICIAL_CHECKPOINT_REVISION = "37936818734242d8e75685b9ee43d2d2e447f98c"
DEFAULT_CHECKPOINT_DIR = str(resolve_hfd_root() / "Vchitect--Vchitect-2.0-2B")
PREPARE_COMMANDS = [
    (
        "hf download Vchitect/Vchitect-2.0-2B "
        f"--revision {OFFICIAL_CHECKPOINT_REVISION} "
        f"--local-dir {DEFAULT_CHECKPOINT_DIR}"
    ),
]


@dataclass(frozen=True)
class VchitectComponents:
    """Vchitect components implementation."""
    pipeline_cls: object


def load_vchitect_components() -> VchitectComponents:
    """
    Load the in-tree Vchitect-2 implementation components.

    Args:
        None: This loader has no parameters because checkpoints stay external assets.
    """
    from .models.pipeline import VchitectXLPipeline

    return VchitectComponents(pipeline_cls=VchitectXLPipeline)
