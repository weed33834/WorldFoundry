from __future__ import annotations

from pathlib import Path


def _shared_openpi_package_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "openpi" / "openpi_runtime" / "openpi"


def _extend_with_shared_openpi(package_path, *parts: str):
    shared_path = _shared_openpi_package_dir().joinpath(*parts)
    if shared_path.is_dir():
        shared_text = str(shared_path)
        if shared_text not in package_path:
            package_path.append(shared_text)
    return package_path


_extend_with_shared_openpi(__path__)
