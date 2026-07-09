"""Small PathManager shim used by the in-tree lazy config loader."""

from __future__ import annotations

from pathlib import Path


class PathManager:
    """Local-filesystem subset of fvcore's PathManager API."""

    @staticmethod
    def open(path, mode: str = "r", *args, **kwargs):
        file_path = Path(path)
        if any(flag in mode for flag in ("w", "a", "x")):
            file_path.parent.mkdir(parents=True, exist_ok=True)
        return file_path.open(mode, *args, **kwargs)

    @staticmethod
    def isfile(path) -> bool:
        return Path(path).is_file()

    @staticmethod
    def isdir(path) -> bool:
        return Path(path).is_dir()
