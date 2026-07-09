"""Module for base_models -> llm_mllm_core -> mllm -> open_flamingo -> __init__.py functionality."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
IMPORT_ROOT = PACKAGE_ROOT.parent


def _is_under(path: Path, root: Path) -> bool:
    """Helper function to is under.

    Args:
        path: The path.
        root: The root.

    Returns:
        The return value.
    """
    return path == root or root in path.parents


def _assert_top_level_package_not_shadowed() -> None:
    """Helper function to assert top level package not shadowed.

    Returns:
        The return value.
    """
    module = sys.modules.get("open_flamingo")
    if module is None:
        return

    origins: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        origins.append(Path(str(module_file)).resolve())
    module_paths = getattr(module, "__path__", None)
    if module_paths is not None:
        origins.extend(Path(str(path)).resolve() for path in module_paths)
    if origins and any(_is_under(origin, PACKAGE_ROOT) for origin in origins):
        return
    origin_text = ", ".join(str(origin) for origin in origins) or "<unknown>"
    raise RuntimeError(f"open_flamingo is already imported from outside WorldFoundry base_models: {origin_text}")


def ensure_import_paths() -> tuple[Path, ...]:
    """Expose the shared OpenFlamingo source as the top-level ``open_flamingo`` package."""

    _assert_top_level_package_not_shadowed()
    import_root = str(IMPORT_ROOT)
    if import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)
    return (IMPORT_ROOT,)


__all__ = ["IMPORT_ROOT", "PACKAGE_ROOT", "ensure_import_paths"]
