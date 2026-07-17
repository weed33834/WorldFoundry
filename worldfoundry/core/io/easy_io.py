"""Minimal local/Hugging Face serialization facade for inference assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .serialization import dump_serialized, load_serialized
from .storage import copy_uri, exists_uri, list_uri


def resolve_checkpoint_path(value: str | Path) -> str:
    text = str(value)
    if text.startswith("hf://"):
        from .hf import resolve_hf_path

        return str(resolve_hf_path(text))
    return str(Path(text).expanduser())


get_checkpoint_path = resolve_checkpoint_path
download_checkpoint = resolve_checkpoint_path


class EasyIO:
    def __init__(self) -> None:
        self._storage_options: dict[str, dict[str, Any]] = {}

    def _options(self, backend_key=None, **kwargs) -> dict[str, Any]:
        options = dict(self._storage_options.get(str(backend_key), {})) if backend_key is not None else {}
        options.update(kwargs)
        for ignored in ("fast_backend", "backend_key"):
            options.pop(ignored, None)
        return options

    def exists(self, path, **kwargs) -> bool:
        try:
            resolved = resolve_checkpoint_path(path)
            return exists_uri(resolved, **self._options(**kwargs))
        except Exception:
            return False

    def load(self, path, *, map_location="cpu", weights_only=True, **kwargs):
        """Load serialized data, defaulting PyTorch formats to safe tensor-only mode."""

        options = {key: value for key, value in kwargs.items() if key in {"encoding", "file_format", "loader"}}
        return load_serialized(
            resolve_checkpoint_path(path),
            map_location=map_location,
            weights_only=weights_only,
            **options,
        )

    def dump(self, value, path, **kwargs):
        options = {key: item for key, item in kwargs.items() if key in {"encoding", "file_format"}}
        return dump_serialized(value, path, **options)

    def copyfile(self, source, destination, **kwargs) -> str:
        return copy_uri(source, destination, **self._options(**kwargs))

    copyfile_from_local = copyfile
    copyfile_to_local = copyfile

    def list_dir_or_file(
        self,
        path,
        *,
        recursive: bool = False,
        list_dir: bool = False,
        list_file: bool = True,
        suffix=None,
        **kwargs,
    ) -> list[str]:
        if list_dir and not list_file:
            root = Path(path)
            iterator = root.rglob("*") if recursive else root.iterdir()
            return sorted(str(item) for item in iterator if item.is_dir())
        return list_uri(
            path,
            recursive=recursive,
            suffix=suffix,
            **self._options(**kwargs),
        )

    def set_s3_backend(self, *, backend_key="default", **kwargs) -> None:
        """Store fsspec-compatible options for subsequent calls using *backend_key*."""

        self._storage_options[str(backend_key)] = dict(kwargs)


easy_io = EasyIO()


__all__ = [
    "EasyIO",
    "download_checkpoint",
    "easy_io",
    "get_checkpoint_path",
    "resolve_checkpoint_path",
]
