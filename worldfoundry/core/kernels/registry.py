"""Runtime registry for optional in-tree accelerator kernels.

The registry deliberately keeps the PyTorch implementation as the semantic
source of truth.  Accelerator kernels are selected per workload signature and
failed signatures are quarantined, so one unsupported shape does not disable a
backend globally.  CUDA out-of-memory errors are never converted into a silent
fallback because retrying usually increases peak memory.
"""

from __future__ import annotations

import os
import threading
import warnings
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

KernelCallable = Callable[..., Any]
KernelPredicate = Callable[..., bool]


class KernelNotSupported(RuntimeError):
    """Raised by a kernel wrapper when a workload is outside its contract."""


@dataclass(frozen=True)
class KernelCandidate:
    """One implementation of a logical operator."""

    op: str
    backend: str
    name: str
    priority: int
    implementation: KernelCallable
    predicate: KernelPredicate


class KernelRegistry:
    """Select accelerator kernels while retaining an explicit fallback."""

    def __init__(self, *, failure_limit: int = 1024, selection_limit: int = 4096) -> None:
        self._candidates: dict[str, list[KernelCandidate]] = defaultdict(list)
        self._failures: set[tuple[object, ...]] = set()
        self._failure_order: deque[tuple[object, ...]] = deque()
        self._failure_messages: dict[tuple[object, ...], str] = {}
        self._failure_limit = int(failure_limit)
        self._selection_cache: OrderedDict[tuple[object, ...], KernelCandidate | None] = OrderedDict()
        self._selection_limit = int(selection_limit)
        self._lock = threading.Lock()

    def register(
        self,
        op: str,
        *,
        backend: str,
        name: str,
        implementation: KernelCallable,
        predicate: KernelPredicate,
        priority: int = 0,
    ) -> KernelCandidate:
        candidate = KernelCandidate(
            op=str(op),
            backend=str(backend),
            name=str(name),
            priority=int(priority),
            implementation=implementation,
            predicate=predicate,
        )
        entries = self._candidates[candidate.op]
        if any(item.name == candidate.name for item in entries):
            raise ValueError(f"kernel {candidate.name!r} is already registered for {candidate.op!r}")
        entries.append(candidate)
        entries.sort(key=lambda item: (-item.priority, item.name))
        return candidate

    def dispatch(
        self,
        op: str,
        fallback: KernelCallable,
        *args: Any,
        signature: tuple[object, ...],
        **kwargs: Any,
    ) -> Any:
        requested = os.getenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto").strip().casefold() or "auto"
        if requested in {"torch", "pytorch", "native", "off", "disabled"}:
            return fallback(*args, **kwargs)

        selection_key = (op, requested, *signature)
        cached, cache_hit = self._cached_selection(selection_key)
        skipped: set[str] = set()
        if cache_hit:
            if cached is None:
                return fallback(*args, **kwargs)
            failure_key = (cached.name, *signature)
            if failure_key not in self._failures:
                try:
                    return cached.implementation(*args, **kwargs)
                except BaseException as exc:
                    if _is_out_of_memory(exc) or isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    if not _is_optional_kernel_failure(exc):
                        raise
                    self._drop_selection(selection_key)
                    self._remember_failure(failure_key, exc)
                    warnings.warn(
                        f"Kernel {cached.name!r} failed for this workload; using the next eligible fallback: {exc}",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    skipped.add(cached.name)

        for candidate in self._candidates.get(op, ()):
            if candidate.name in skipped:
                continue
            if requested != "auto" and requested not in {candidate.backend.casefold(), candidate.name.casefold()}:
                continue
            failure_key = (candidate.name, *signature)
            if failure_key in self._failures:
                continue
            try:
                if not candidate.predicate(*args, **kwargs):
                    continue
                result = candidate.implementation(*args, **kwargs)
                self._remember_selection(selection_key, candidate)
                return result
            except BaseException as exc:
                if _is_out_of_memory(exc) or isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                if not _is_optional_kernel_failure(exc):
                    raise
                self._remember_failure(failure_key, exc)
                warnings.warn(
                    f"Kernel {candidate.name!r} failed for this workload; using the PyTorch fallback: {exc}",
                    RuntimeWarning,
                    stacklevel=3,
                )
        self._remember_selection(selection_key, None)
        return fallback(*args, **kwargs)

    def _cached_selection(self, key: tuple[object, ...]) -> tuple[KernelCandidate | None, bool]:
        with self._lock:
            if key not in self._selection_cache:
                return None, False
            selected = self._selection_cache.pop(key)
            self._selection_cache[key] = selected
            return selected, True

    def _remember_selection(self, key: tuple[object, ...], selected: KernelCandidate | None) -> None:
        with self._lock:
            self._selection_cache.pop(key, None)
            self._selection_cache[key] = selected
            while len(self._selection_cache) > self._selection_limit:
                self._selection_cache.popitem(last=False)

    def _drop_selection(self, key: tuple[object, ...]) -> None:
        with self._lock:
            self._selection_cache.pop(key, None)

    def _remember_failure(self, key: tuple[object, ...], exc: BaseException) -> None:
        with self._lock:
            if key in self._failures:
                return
            if len(self._failure_order) >= self._failure_limit:
                expired = self._failure_order.popleft()
                self._failures.discard(expired)
                self._failure_messages.pop(expired, None)
            self._failures.add(key)
            self._failure_order.append(key)
            self._failure_messages[key] = f"{type(exc).__name__}: {exc}"

    def report(self) -> dict[str, object]:
        return {
            "requested_backend": os.getenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto"),
            "operators": {
                op: [
                    {"name": item.name, "backend": item.backend, "priority": item.priority}
                    for item in candidates
                ]
                for op, candidates in sorted(self._candidates.items())
            },
            "selection_cache_entries": len(self._selection_cache),
            "failed_signatures": len(self._failures),
            "failures": tuple(self._failure_messages.values()),
        }

    def clear_failures(self) -> None:
        with self._lock:
            self._failures.clear()
            self._failure_order.clear()
            self._failure_messages.clear()
            self._selection_cache.clear()


def _is_out_of_memory(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return "out of memory" in message or "alloc_failed" in message or type(exc).__name__ == "OutOfMemoryError"


def _is_optional_kernel_failure(exc: BaseException) -> bool:
    if isinstance(exc, (ImportError, OSError, KernelNotSupported)):
        return True
    module = type(exc).__module__.casefold()
    if module.startswith("triton"):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).casefold()
    markers = (
        "not supported",
        "unsupported",
        "no kernel image",
        "invalid device function",
        "out of resource",
        "ptxas",
        "triton",
        "requires sm",
    )
    return any(marker in message for marker in markers)


KERNEL_REGISTRY = KernelRegistry()


def kernel_dispatch_report() -> dict[str, object]:
    """Return registered implementations and quarantined failure counts."""

    return KERNEL_REGISTRY.report()


def clear_kernel_dispatch_cache() -> None:
    """Clear workload selections and runtime failure quarantine."""

    KERNEL_REGISTRY.clear_failures()


__all__ = [
    "KERNEL_REGISTRY",
    "KernelCandidate",
    "KernelNotSupported",
    "KernelRegistry",
    "clear_kernel_dispatch_cache",
    "kernel_dispatch_report",
]
