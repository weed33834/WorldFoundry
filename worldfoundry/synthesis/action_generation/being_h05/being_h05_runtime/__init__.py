from __future__ import annotations

import sys
import types
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent


def _install_namespace_alias(name: str, package_dir: Path) -> None:
    module_path = package_dir.resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        module_file = Path(str(getattr(existing, "__file__", ""))).resolve()
        module_paths = [Path(item).resolve() for item in getattr(existing, "__path__", ())]
        if module_file == module_path / "__init__.py" or module_path in module_paths:
            return
        raise RuntimeError(f"{name} is already imported from outside the Being-H0.5 in-tree runtime: {module_file}")

    module = types.ModuleType(name)
    module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module


def install_aliases() -> None:
    """Expose vendored upstream packages under their official import names.

    Args:
        None. The alias targets are resolved from this in-tree runtime package.
    """

    _install_namespace_alias("BeingH", RUNTIME_ROOT / "BeingH")
    _install_namespace_alias("configs", RUNTIME_ROOT / "configs")


__all__ = ["RUNTIME_ROOT", "install_aliases"]
