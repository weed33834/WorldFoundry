"""Shared MCP tool response helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

_T = TypeVar("_T")


def success_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach ``ok=True`` when the payload does not already declare status."""

    if "ok" in payload:
        return payload
    return {"ok": True, **payload}


def error_payload(message: str, *, error_type: str = "error") -> dict[str, Any]:
    """Return a structured MCP tool error response."""

    return {"ok": False, "error": message, "error_type": error_type}


def invoke_tool(fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a payload builder and normalize success/error envelopes."""

    try:
        result = fn(*args, **kwargs)
    except (ValueError, KeyError, FileNotFoundError, LookupError) as exc:
        return error_payload(str(exc))
    except RuntimeError as exc:
        return error_payload(str(exc), error_type="runtime")
    if isinstance(result, dict):
        return success_payload(result)
    return success_payload({"result": result})


async def invoke_tool_async(
    fn: Callable[..., Awaitable[_T]],
    /,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Async variant of :func:`invoke_tool`."""

    try:
        result = await fn(*args, **kwargs)
    except (ValueError, KeyError, FileNotFoundError, LookupError) as exc:
        return error_payload(str(exc))
    except RuntimeError as exc:
        return error_payload(str(exc), error_type="runtime")
    if isinstance(result, dict):
        return success_payload(result)
    return success_payload({"result": result})


__all__ = ["error_payload", "invoke_tool", "invoke_tool_async", "success_payload"]
