"""Persistent, in-tree torch.compile cache and wrapper reuse.

The module imports torch lazily so control-plane and CPU-only processes can
configure WorldFoundry without loading an accelerator runtime.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import re
import sys
import tempfile
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping

from worldfoundry.runtime.env import resolve_cache_dir


@dataclass(frozen=True, slots=True)
class CompilePolicy:
    """Stable options that identify one compiled module variant."""

    backend: str = "inductor"
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool | None = None
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class CompileCacheLayout:
    """Resolved persistent compiler cache directories."""

    root: Path
    inductor: Path
    triton: Path
    fingerprint: str


_CONFIGURE_LOCK = threading.Lock()
_COMPILE_LOCK = threading.RLock()


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value or "unknown")).strip("-.")
    if not token:
        return "unknown"
    if len(token) <= 96:
        return token
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    return f"{token[:79]}-{digest}"


def _torch_fingerprint() -> str:
    """Return a cache namespace that separates incompatible generated code."""

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return "torch-unavailable"

    version = _safe_token(getattr(torch, "__version__", "unknown"))
    try:
        triton_version = _safe_token(importlib.metadata.version("triton"))
    except importlib.metadata.PackageNotFoundError:
        triton_version = "unavailable"
    python_tag = _safe_token(getattr(sys.implementation, "cache_tag", "python-unknown"))
    runtime = getattr(torch, "version", None)
    hip = getattr(runtime, "hip", None)
    cuda_version = getattr(runtime, "cuda", None)
    try:
        cudnn_version = getattr(getattr(torch, "backends", None), "cudnn", None)
        cudnn_version = cudnn_version.version() if cudnn_version is not None else None
    except Exception:
        cudnn_version = None
    accelerator = "cpu"
    try:
        device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if hip and torch.cuda.is_available():
            devices = set()
            for device_index in range(device_count):
                properties = torch.cuda.get_device_properties(device_index)
                devices.add(
                    f"{getattr(properties, 'gcnArchName', 'unknown')}-"
                    f"{getattr(properties, 'name', 'gpu')}-"
                    f"cus{getattr(properties, 'multi_processor_count', 'unknown')}-"
                    f"smem{getattr(properties, 'shared_memory_per_multiprocessor', 'unknown')}"
                )
            accelerator = f"rocm-{hip}-" + "_and_".join(sorted(devices))
        elif torch.cuda.is_available():
            # The cache directory is process-global. Fingerprint the unique
            # visible hardware set instead of LOCAL_RANK so a heterogeneous
            # multi-GPU process cannot accidentally namespace all generated
            # code under whichever device happened to share its rank index.
            devices = set()
            for device_index in range(device_count):
                major, minor = torch.cuda.get_device_capability(device_index)
                properties = torch.cuda.get_device_properties(device_index)
                devices.add(
                    f"sm{major}{minor}-{getattr(properties, 'name', 'gpu')}-"
                    f"sms{getattr(properties, 'multi_processor_count', 'unknown')}-"
                    f"smem{getattr(properties, 'shared_memory_per_multiprocessor', 'unknown')}"
                )
            accelerator = f"cuda-{cuda_version}-" + "_and_".join(sorted(devices))
        elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            accelerator = "xpu"
        elif getattr(getattr(torch, "backends", None), "mps", None) is not None:
            if torch.backends.mps.is_available():
                accelerator = "mps"
    except Exception:
        accelerator = "accelerator-unknown"
    return _safe_token(
        f"torch-{version}-triton-{triton_version}-cudnn-{cudnn_version or 'unknown'}-"
        f"{python_tag}-{accelerator}"
    )


def _ensure_cache_directory(directory: Path, *, fingerprint: str, kind: str) -> Path:
    """Create a compiler cache directory with a writable local fallback."""

    try:
        directory.mkdir(parents=True, exist_ok=True)
        # ``mkdir(exist_ok=True)`` says nothing about an existing read-only
        # directory. Exercise the same create/delete permission compiler caches
        # require before committing the process-global environment to it.
        with tempfile.NamedTemporaryFile(prefix=".worldfoundry-write-test-", dir=directory):
            pass
        return directory
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / "worldfoundry-compile" / fingerprint / kind
        fallback.mkdir(parents=True, exist_ok=True)
        warnings.warn(
            f"Cannot use compiler cache directory {directory}: {exc}. "
            f"Falling back to {fallback}.",
            RuntimeWarning,
            stacklevel=3,
        )
        return fallback


def configure_persistent_compile_cache(
    *,
    namespace: str = "default",
    environ: MutableMapping[str, str] | None = None,
) -> CompileCacheLayout:
    """Configure persistent Inductor and Triton caches once per process.

    Explicit ``TORCHINDUCTOR_CACHE_DIR`` and ``TRITON_CACHE_DIR`` values are
    respected. Otherwise caches live below ``WORLDFOUNDRY_CACHE_DIR`` and are
    partitioned by torch/runtime/accelerator compatibility. Compatible model
    namespaces intentionally share the binary/autotune cache; Inductor's own
    graph keys prevent collisions, while a process-global environment variable
    cannot safely point at a different directory for every compiled module.
    """

    del namespace
    env = os.environ if environ is None else environ
    fingerprint = _torch_fingerprint()
    configured_root = env.get("WORLDFOUNDRY_COMPILE_CACHE_DIR", "").strip()
    base = Path(configured_root).expanduser() if configured_root else resolve_cache_dir(env) / "compile"
    root = base / fingerprint

    with _CONFIGURE_LOCK:
        inductor = Path(env.get("TORCHINDUCTOR_CACHE_DIR") or root / "inductor").expanduser()
        triton = Path(env.get("TRITON_CACHE_DIR") or root / "triton").expanduser()
        inductor = _ensure_cache_directory(inductor, fingerprint=fingerprint, kind="inductor")
        triton = _ensure_cache_directory(triton, fingerprint=fingerprint, kind="triton")
        # Assignment is intentional: when an explicitly configured path is not
        # writable, _ensure_cache_directory has selected a valid fallback.
        env["TORCHINDUCTOR_CACHE_DIR"] = str(inductor)
        env["TRITON_CACHE_DIR"] = str(triton)
        env.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        env.setdefault("TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE", "1")
        env.setdefault("TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE", "0")

    return CompileCacheLayout(
        root=root,
        inductor=inductor,
        triton=triton,
        fingerprint=fingerprint,
    )


def compile_module_cached(
    module: Any,
    *,
    policy: CompilePolicy | None = None,
    namespace: str = "modules",
) -> Any:
    """Return one cached ``torch.compile`` wrapper per module and policy."""

    selected = CompilePolicy() if policy is None else policy
    if not selected.enabled:
        return module
    if hasattr(module, "_orig_mod"):
        return module

    configure_persistent_compile_cache(namespace=namespace)
    torch = importlib.import_module("torch")
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        return module

    key = (
        selected.backend,
        selected.mode,
        selected.fullgraph,
        selected.dynamic,
    )
    with _COMPILE_LOCK:
        variants = getattr(module, "_worldfoundry_compiled_variants", None)
        if not isinstance(variants, dict):
            variants = {}
            try:
                setattr(module, "_worldfoundry_compiled_variants", variants)
            except Exception:
                variants = {}
        cached = variants.get(key)
        if cached is not None:
            return cached
        try:
            compiled = compile_fn(
                module,
                backend=selected.backend,
                mode=selected.mode,
                fullgraph=selected.fullgraph,
                dynamic=selected.dynamic,
            )
        except Exception:
            return module
        variants[key] = compiled
        return compiled


def compile_callable_cached(
    function: Any,
    *,
    policy: CompilePolicy | None = None,
    namespace: str = "functions",
) -> Any:
    """Compile a callable once without selecting compilation implicitly."""

    selected = CompilePolicy() if policy is None else policy
    if not selected.enabled:
        return function
    configure_persistent_compile_cache(namespace=namespace)
    torch = importlib.import_module("torch")
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        return function

    key = (
        selected.backend,
        selected.mode,
        selected.fullgraph,
        selected.dynamic,
    )
    with _COMPILE_LOCK:
        variants = getattr(function, "_worldfoundry_compiled_variants", None)
        if not isinstance(variants, dict):
            variants = {}
            try:
                setattr(function, "_worldfoundry_compiled_variants", variants)
            except Exception:
                # Bound methods cannot always carry attributes. Their owner
                # should retain the returned wrapper when repeated reuse matters.
                variants = {}
        cached = variants.get(key)
        if cached is not None:
            return cached
        try:
            compiled = compile_fn(
                function,
                backend=selected.backend,
                mode=selected.mode,
                fullgraph=selected.fullgraph,
                dynamic=selected.dynamic,
            )
        except Exception:
            return function
        variants[key] = compiled
        return compiled


__all__ = [
    "CompileCacheLayout",
    "CompilePolicy",
    "compile_callable_cached",
    "compile_module_cached",
    "configure_persistent_compile_cache",
]
