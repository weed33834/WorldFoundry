"""Data file I/O helpers used by bundled model runtimes.

This module intentionally stays thin: data-facing callers can import from
``worldfoundry.data.io``, while the actual storage and serialization behavior is
implemented in ``worldfoundry.core.io``.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any
from urllib.parse import urlparse

from worldfoundry.core.io import (
    copy_uri,
    dump_serialized,
    exists_uri,
    load_serialized,
    parse_uri_scheme,
    save_image_or_video_tensor,
    uri_to_local_path,
)


def _strip_storage_compat_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs.pop("backend_args", None)
    kwargs.pop("backend_key", None)
    return kwargs


def _storage_options_from_backend_args(backend_args: dict[str, Any] | None) -> dict[str, Any]:
    if not backend_args:
        return {}
    options = dict(backend_args)
    options.pop("backend", None)
    options.pop("path_mapping", None)
    options.pop("s3_credential_path", None)
    return options


def load(file: str | os.PathLike[str] | IO[Any], *, file_format: str | None = None, **kwargs: Any) -> Any:
    """Load a local/URI data object, inferring format from suffix when needed."""

    kwargs = _strip_storage_compat_kwargs(dict(kwargs))
    if "weights_only" not in kwargs:
        normalized = (file_format or str(file).rsplit(".", 1)[-1]).lower()
        if normalized in {"pt", "pth", "ckpt", "bin"}:
            kwargs["weights_only"] = False
    return load_serialized(file, file_format=file_format, **kwargs)


def dump(
    obj: Any,
    file: str | os.PathLike[str] | IO[Any] | None = None,
    *,
    file_format: str | None = None,
    **kwargs: Any,
) -> Any:
    """Dump a data object, inferring format from suffix when possible."""

    kwargs = _strip_storage_compat_kwargs(dict(kwargs))
    return dump_serialized(obj, file, file_format=file_format, **kwargs)


def exists(file: str | os.PathLike[str], **kwargs: Any) -> bool:
    """Return whether a local/URI data object exists."""

    kwargs = _strip_storage_compat_kwargs(dict(kwargs))
    return exists_uri(file, **kwargs)


def copyfile_to_local(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    dst_type: str = "file",
    **kwargs: Any,
) -> str:
    """Copy a local/URI data object to a local file or directory."""

    kwargs = _strip_storage_compat_kwargs(dict(kwargs))
    destination = Path(dst).expanduser()
    if dst_type == "dir":
        destination = destination / (Path(urlparse(str(src)).path).name or Path(str(src)).name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return copy_uri(src, destination, **kwargs)


def copyfile(src: str | os.PathLike[str], dst: str | os.PathLike[str], **kwargs: Any) -> str:
    """Copy a local/URI data object to another local/URI destination."""

    kwargs = _strip_storage_compat_kwargs(dict(kwargs))
    return copy_uri(src, dst, **kwargs)


def download_from_cache_or_uri(
    source_path: str | os.PathLike[str],
    cache_fp: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    rank_sync: bool = True,
    backend_args: dict[str, Any] | None = None,
    backend_key: str | None = None,
) -> str:
    """Resolve a local/URI source to a local cached file."""

    del backend_key
    cache_path = _cache_path_for_source(source_path, cache_fp, cache_dir)

    if rank_sync:
        _run_on_rank0(lambda: _populate_cache_file(source_path, cache_path, backend_args=backend_args))
        _distributed_barrier()
    else:
        _populate_cache_file(source_path, cache_path, backend_args=backend_args)
    return str(cache_path)


def load_from_cache_or_uri(
    source_path: str | os.PathLike[str],
    cache_fp: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    rank_sync: bool = True,
    backend_args: dict[str, Any] | None = None,
    backend_key: str | None = None,
    easy_io_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Load a local/URI data object, caching remote sources on local disk first."""

    cached_path = download_from_cache_or_uri(
        source_path,
        cache_fp=cache_fp,
        cache_dir=cache_dir,
        rank_sync=rank_sync,
        backend_args=backend_args,
        backend_key=backend_key,
    )
    return load(cached_path, **(easy_io_kwargs or {}))


def load_from_s3_with_cache(*args: Any, **kwargs: Any) -> Any:
    """Compatibility alias for bundled runtimes that used S3-specific naming."""

    return load_from_cache_or_uri(*args, **kwargs)


def set_s3_backend(*args: Any, **kwargs: Any) -> None:
    """Compatibility no-op; URI handling is centralized in ``worldfoundry.core.io``."""

    del args, kwargs


def save_img_or_video(
    sample_c_t_h_w_in01,
    save_fp_wo_ext: str | os.PathLike[str] | IO[Any],
    fps: int = 24,
    quality: int | None = None,
    ffmpeg_params: list[str] | None = None,
) -> None:
    """Save a ``[C,T,H,W]`` tensor as ``.jpg`` for one frame or ``.mp4`` otherwise."""

    save_image_or_video_tensor(
        sample_c_t_h_w_in01,
        save_fp_wo_ext,
        fps=fps,
        quality=quality,
        ffmpeg_params=ffmpeg_params,
        value_range="0,1",
    )


def _cache_path_for_source(
    source_path: str | os.PathLike[str],
    cache_fp: str | os.PathLike[str] | None,
    cache_dir: str | os.PathLike[str] | None,
) -> Path:
    if cache_dir is None:
        cache_dir = os.environ.get("TORCH_HOME") or os.environ.get("IMAGINAIRE_CACHE_DIR", "~/.cache/imaginaire")
    cache_root = Path(os.path.expanduser(str(cache_dir)))
    if cache_fp is None:
        source_text = str(source_path).replace("://", "/").lstrip("/")
        cache_fp = cache_root / source_text
    cache_path = Path(os.path.expanduser(str(cache_fp)))
    if not cache_path.is_absolute():
        cache_path = cache_root / cache_path
    return cache_path


def _populate_cache_file(
    source_path: str | os.PathLike[str],
    cache_path: Path,
    *,
    backend_args: dict[str, Any] | None,
) -> None:
    if parse_uri_scheme(source_path) == "file":
        local_source = uri_to_local_path(source_path)
        if local_source.resolve() == cache_path.resolve():
            return
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return
    if cache_path.exists():
        cache_path.unlink()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    copy_uri(source_path, cache_path, **_storage_options_from_backend_args(backend_args))


def _run_on_rank0(func) -> None:
    try:
        from worldfoundry.core.distributed import torch_process_group as distributed
    except Exception:
        func()
        return
    if distributed.get_rank() == 0:
        func()


def _distributed_barrier() -> None:
    try:
        from worldfoundry.core.distributed import torch_process_group as distributed
    except Exception:
        return
    distributed.barrier()


easy_io = SimpleNamespace(
    copyfile=copyfile,
    copyfile_to_local=copyfile_to_local,
    dump=dump,
    exists=exists,
    load=load,
    set_s3_backend=set_s3_backend,
)


__all__ = [
    "copyfile",
    "copyfile_to_local",
    "download_from_cache_or_uri",
    "dump",
    "easy_io",
    "exists",
    "load",
    "load_from_cache_or_uri",
    "load_from_s3_with_cache",
    "save_img_or_video",
    "set_s3_backend",
]
