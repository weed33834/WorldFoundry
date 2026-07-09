from __future__ import annotations

from pathlib import Path
from urllib.parse import quote


def file_url(path: str | Path | None) -> str:
    if not path:
        return ""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return ""
    return f"/gradio_api/file={quote(resolved.as_posix(), safe='/')}"


_file_url = file_url

__all__ = ["file_url"]
