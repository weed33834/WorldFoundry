"""API-based runtime configuration and management helpers for remote pipelines."""

from __future__ import annotations

import os
from collections.abc import Sequence


_PLACEHOLDER_API_KEYS = {"", "your_api_key", "your api key"}


def resolve_api_key(api_key: str | None, env_names: Sequence[str], service_name: str) -> str:
    """Resolve an API key from explicit input or documented environment variables.

    Args:
        api_key: API key passed by the caller.
        env_names: Environment variable names checked in priority order.
        service_name: Human-readable service name used in error messages.
    """

    value = (api_key or "").strip()
    if value and value not in _PLACEHOLDER_API_KEYS:
        return value
    for env_name in env_names:
        # Attempt to retrieve from environment variables as a fallback resolution
        env_value = os.getenv(env_name)
        if env_value:
            return env_value
    joined = "/".join(env_names)
    raise ValueError(f"{service_name} API key is required. Pass api_key or set {joined}.")
