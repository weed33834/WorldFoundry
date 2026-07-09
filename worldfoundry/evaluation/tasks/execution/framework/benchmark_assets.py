"""Bundled benchmark asset path helpers.

Official prompts, rubrics, and manifests ship under
``worldfoundry/data/benchmarks/assets/<benchmark_id>/``. Runners resolve bundled
assets by default; ``WORLDFOUNDRY_*`` env vars and explicit kwargs override only
when the caller sets them.
"""

from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.utils import worldfoundry_data_path


def bundled_benchmark_assets_root(benchmark_id: str) -> Path:
    """Return the bundled asset directory for ``benchmark_id``."""
    return worldfoundry_data_path("benchmarks", "assets", benchmark_id)


def bundled_benchmark_asset(benchmark_id: str, *relative: str | Path) -> Path:
    """Return a path under the bundled asset tree for ``benchmark_id``."""
    return bundled_benchmark_assets_root(benchmark_id).joinpath(*relative)


def first_existing_path(*candidates: Path | None) -> Path | None:
    """Return the first candidate path that exists on disk."""
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.expanduser().resolve()
    return None
