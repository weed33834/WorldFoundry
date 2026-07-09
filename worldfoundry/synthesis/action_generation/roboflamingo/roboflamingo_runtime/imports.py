from __future__ import annotations

import sys
from pathlib import Path

from worldfoundry.base_models.llm_mllm_core.mllm.open_flamingo import (
    PACKAGE_ROOT as OPEN_FLAMINGO_ROOT,
    ensure_import_paths as ensure_open_flamingo_import_paths,
)


RUNTIME_ROOT = Path(__file__).resolve().parent
ROBOT_FLAMINGO_ROOT = RUNTIME_ROOT / "robot_flamingo"


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _assert_top_level_package_not_shadowed(name: str, expected_root: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        return

    origins: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        origins.append(Path(str(module_file)).resolve())
    module_paths = getattr(module, "__path__", None)
    if module_paths is not None:
        origins.extend(Path(str(path)).resolve() for path in module_paths)
    if origins and any(_is_under(origin, expected_root) for origin in origins):
        return
    origin_text = ", ".join(str(origin) for origin in origins) or "<unknown>"
    raise RuntimeError(f"{name} is already imported from outside the WorldFoundry RoboFlamingo runtime: {origin_text}")


def ensure_runtime_import_paths() -> tuple[Path, ...]:
    """Expose RoboFlamingo runtime code and shared OpenFlamingo base code."""

    ensure_open_flamingo_import_paths()
    _assert_top_level_package_not_shadowed("open_flamingo", OPEN_FLAMINGO_ROOT)
    _assert_top_level_package_not_shadowed("robot_flamingo", ROBOT_FLAMINGO_ROOT)
    runtime_root = str(RUNTIME_ROOT)
    if runtime_root in sys.path:
        sys.path.remove(runtime_root)
    sys.path.insert(0, runtime_root)
    return (RUNTIME_ROOT, OPEN_FLAMINGO_ROOT)


__all__ = ["RUNTIME_ROOT", "ROBOT_FLAMINGO_ROOT", "ensure_runtime_import_paths"]
