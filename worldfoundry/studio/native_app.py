"""Gradio-free launcher for native WorldFoundry Studio frontends."""

from __future__ import annotations

import os
from typing import Sequence

from .catalog import CatalogEntry, discover_catalog, find_entry
from .visualization.backends.frontends import NATIVE_FRONTENDS, UNIFIED_FRONTEND, resolve_frontend_mode, serve_native_frontend
from .launch_config import StudioLaunchConfig, parse_launch_config as _parse_launch_config_core
from .variants import resolve_cli_variant_id as _resolve_shared_cli_variant_id


def _studio_catalog(entries: Sequence[CatalogEntry] | None = None) -> tuple[CatalogEntry, ...]:
    return tuple(entries) if entries is not None else tuple(discover_catalog())


def _resolve_cli_variant_id(entry: CatalogEntry, raw_variant: str | None) -> str | None:
    """Resolve native-frontend variant aliases without importing Gradio app code."""

    return _resolve_shared_cli_variant_id(entry, raw_variant)


def parse_launch_config(argv: Sequence[str] | None = None) -> StudioLaunchConfig:
    return _parse_launch_config_core(
        argv,
        studio_catalog=_studio_catalog,
        resolve_cli_variant_id=_resolve_cli_variant_id,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Launch a native Studio frontend without importing Gradio."""

    os.environ.setdefault("WORLDFOUNDRY_STUDIO_SKIP_RUNTIME_PROFILES", "1")
    launch_config = parse_launch_config(argv)
    entry = find_entry(launch_config.model_id)
    frontend_mode = resolve_frontend_mode(entry, launch_config.frontend)
    if frontend_mode == UNIFIED_FRONTEND:
        raise SystemExit("The `unified` frontend requires Gradio. Install `worldfoundry[ui]` or use --frontend world.")
    if frontend_mode not in NATIVE_FRONTENDS:
        raise SystemExit(f"Unsupported native Studio frontend: {frontend_mode}")
    serve_native_frontend(entry, launch_config, frontend_mode)


if __name__ == "__main__":
    main()
