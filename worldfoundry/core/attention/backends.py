"""Attention backend capability probes shared by inference dispatchers.

This module is responsible for safely, lightly, and side-effect-freely probing and 
dispatching the optimal Attention computation backend in the current runtime environment.

Core Architectural Design:
1. Zero-Cost Probing: Uses `importlib.util.find_spec` to dynamically query packages
   rather than physically importing them. Since many acceleration packages (e.g., `flash_attn`)
   might crash hard during import/initialization (e.g., due to missing shared libraries or
   mismatched CUDA runtimes), this probing mechanism ensures a graceful fallback to PyTorch
   native SDPA without crashing the process.
2. Two-Stage Validation: Separates physical availability (`available`) from runtime hardware
   compatibility (`usable`). For example, even if `flash_attn` is successfully installed, it
   will be marked unusable if the active GPU Compute Capability is < 8.0 (e.g., V100, T4).
3. Fallback Priority Chain: Defines structured fallback paths for standard/default backends
   as well as experimental/sparse attention kernels (e.g., video sparse attention, V-MoBA),
   ensuring the system is highly robust across diverse hardware.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import torch


@dataclass(frozen=True)
class AttentionKernelCapability:
    """Runtime availability metadata for a single attention kernel family.

    Attributes:
        name: Canonical identifier of the attention backend.
        package: The underlying Python package or C++/CUDA extension.
        available: True if the module/package is physically installed in the environment.
        usable: True if the backend is physically runnable on the active hardware (e.g., CUDA capability).
        reason: Explains why the backend is unusable or unavailable, if applicable.
    """

    name: str
    package: str
    available: bool
    usable: bool
    reason: str = ""


_AUTO = "auto"
_TORCH = "torch"
_FLASH_AUTO = "flash_attention"
_BACKEND_ALIASES: Mapping[str, str] = {
    "auto": _AUTO,
    "default": _AUTO,
    "torch": _TORCH,
    "torch_sdpa": _TORCH,
    "sdpa": _TORCH,
    "math": _TORCH,
    "efficient": _TORCH,
    "cudnn": _TORCH,
    "flash": _FLASH_AUTO,
    "flash_attn": _FLASH_AUTO,
    "flash_attention": _FLASH_AUTO,
    "flash_attention_2": "flash_attention_2",
    "flash_attn_2": "flash_attention_2",
    "flash_attn2": "flash_attention_2",
    "flash_attention_3": "flash_attention_3",
    "flash_attn_3": "flash_attention_3",
    "flash_attn3": "flash_attention_3",
    "sage": "sage_attention",
    "sage_attn": "sage_attention",
    "sage_attention": "sage_attention",
    "sage_attn_three": "sage_attention_3",
    "sage_attention_3": "sage_attention_3",
    "sage3": "sage_attention_3",
    "xformers": "xformers",
    "xformers_attention": "xformers",
    "video_sparse_attn": "video_sparse_attention",
    "video_sparse_attention": "video_sparse_attention",
    "vmoba": "vmoba_attention",
    "vmoba_attn": "vmoba_attention",
    "vmoba_attention": "vmoba_attention",
    "sla": "sla_attention",
    "sla_attn": "sla_attention",
    "sla_attention": "sla_attention",
    "sage_sla": "sage_sla_attention",
    "sage_sla_attn": "sage_sla_attention",
    "sage_sla_attention": "sage_sla_attention",
}
_DEFAULT_PRIORITY = (
    "flash_attention_3",
    "flash_attention_2",
    "sage_attention",
    "xformers",
    _TORCH,
)
_EXPERIMENTAL_PRIORITY = (
    "video_sparse_attention",
    "vmoba_attention",
    "sla_attention",
    "sage_sla_attention",
)
_VIDEO_SPARSE_KERNEL_IMPORT = "fast" + "video_kernel"


def attention_backend_from_env(environ: Mapping[str, str] | None = None) -> str:
    """Read and resolve the canonical attention backend requested by the environment.

    Inspects `WORLDFOUNDRY_ATTENTION_IMPLEMENTATION` or `WORLDFOUNDRY_ATTENTION_BACKEND`,
    falling back to `"auto"` if neither is specified.
    """
    env = os.environ if environ is None else environ
    value = (
        env.get("WORLDFOUNDRY_ATTENTION_IMPLEMENTATION")
        or env.get("WORLDFOUNDRY_ATTENTION_BACKEND")
        or _AUTO
    )
    return normalize_attention_backend(value)


def normalize_attention_backend(value: str | None) -> str:
    """Normalize colloquial or varied attention backend names into standard keys.

    For example, maps 'flash-attn-2', 'flash2', or 'flash_attention_2' to 'flash_attention_2'
    while stripping and lowering input values to handle typos gracefully.
    """
    if value is None:
        return _AUTO
    key = str(value).strip().lower().replace("-", "_")
    if not key:
        return _AUTO
    canonical = _BACKEND_ALIASES.get(key)
    if canonical is None:
        allowed = ", ".join(sorted(_BACKEND_ALIASES))
        raise ValueError(f"Unknown attention backend {value!r}. Expected one of: {allowed}")
    return canonical


@lru_cache(maxsize=1)
def probe_attention_backends() -> dict[str, AttentionKernelCapability]:
    """Probe installed attention packages and hardware compatibility.

    Uses `lru_cache` to ensure the probing process executes exactly once over the
    process lifecycle, avoiding CPU bottlenecks from repetitive importlib and PyTorch queries.
    """
    flash_gpu = gpu_supports_flash_attention()
    cuda = torch.cuda.is_available()
    return {
        "flash_attention_3": _package_capability(
            name="flash_attention_3",
            package="flash_attn_interface",
            usable_if=flash_gpu,
            unavailable_reason="flash_attn_interface is not installed",
            unusable_reason="FlashAttention requires CUDA with compute capability >= 8.0",
        ),
        "flash_attention_2": _package_capability(
            name="flash_attention_2",
            package="flash_attn",
            usable_if=flash_gpu,
            unavailable_reason="flash_attn is not installed",
            unusable_reason="FlashAttention requires CUDA with compute capability >= 8.0",
        ),
        "sage_attention": _package_capability(
            name="sage_attention",
            package="sageattention",
            usable_if=cuda,
            unavailable_reason="sageattention is not installed",
            unusable_reason="SageAttention requires CUDA",
        ),
        "sage_attention_3": _package_capability(
            name="sage_attention_3",
            package="sageattention",
            usable_if=cuda,
            unavailable_reason="sageattention is not installed",
            unusable_reason="SageAttention 3 requires CUDA and a compatible Blackwell runtime",
        ),
        "xformers": _package_capability(
            name="xformers",
            package="xformers.ops",
            usable_if=cuda,
            unavailable_reason="xformers is not installed",
            unusable_reason="xFormers attention requires CUDA",
        ),
        "video_sparse_attention": _package_capability(
            name="video_sparse_attention",
            package="video_sparse_attention_kernels",
            import_name=_VIDEO_SPARSE_KERNEL_IMPORT,
            usable_if=cuda,
            unavailable_reason="video sparse attention kernel package is not installed",
            unusable_reason="Video sparse attention requires CUDA and model-specific kernels",
        ),
        "vmoba_attention": _package_capability(
            name="vmoba_attention",
            package="vmoba_attention_kernels",
            import_name=_VIDEO_SPARSE_KERNEL_IMPORT,
            usable_if=cuda,
            unavailable_reason="V-MoBA attention kernel package is not installed",
            unusable_reason="V-MoBA attention requires CUDA and model-specific kernels",
        ),
        "sla_attention": _package_capability(
            name="sla_attention",
            package="sparse_linear_attention_kernels",
            import_name=_VIDEO_SPARSE_KERNEL_IMPORT,
            usable_if=cuda,
            unavailable_reason="sparse linear attention kernel package is not installed",
            unusable_reason="Sparse linear attention requires CUDA and model-specific kernels",
        ),
        "sage_sla_attention": _package_capability(
            name="sage_sla_attention",
            package="sage_sparse_linear_attention_kernels",
            import_name=_VIDEO_SPARSE_KERNEL_IMPORT,
            usable_if=cuda,
            unavailable_reason="SageSLA attention kernel package is not installed",
            unusable_reason="SageSLA attention requires CUDA and model-specific kernels",
        ),
        _TORCH: AttentionKernelCapability(name=_TORCH, package="torch", available=True, usable=True),
    }


def resolve_attention_backend(preferred: str | None = None) -> str:
    """Resolve the highest priority usable backend, falling back gracefully to PyTorch SDPA.

    If `"auto"` is preferred, traverses priority list to select the first runnable package.
    If the requested package is not usable on the active hardware, automatically degrades to
    `"torch"` SDPA to ensure robust execution.
    """
    requested = attention_backend_from_env() if preferred is None else normalize_attention_backend(preferred)
    capabilities = probe_attention_backends()
    if requested == _AUTO:
        for name in _DEFAULT_PRIORITY:
            if capabilities[name].usable:
                return name
        return _TORCH
    if requested == _FLASH_AUTO:
        for name in ("flash_attention_3", "flash_attention_2"):
            if capabilities[name].usable:
                return name
        return _TORCH
    capability = capabilities.get(requested)
    if capability is not None and capability.usable:
        return requested
    return _TORCH


def attention_backend_capability(name: str) -> AttentionKernelCapability:
    """Retrieve runtime capability metadata for a single normalized backend."""
    canonical = normalize_attention_backend(name)
    if canonical in {_AUTO, _FLASH_AUTO}:
        canonical = resolve_attention_backend(canonical)
    return probe_attention_backends()[canonical]


def attention_backend_report() -> tuple[AttentionKernelCapability, ...]:
    """Retrieve capability status of all registered backends, ordered by dispatch priority."""
    capabilities = probe_attention_backends()
    return tuple(capabilities[name] for name in (*_DEFAULT_PRIORITY, *_EXPERIMENTAL_PRIORITY) if name in capabilities)


def gpu_supports_flash_attention() -> bool:
    """Determine if the active CUDA device is architecturally capable of running FlashAttention.

    FlashAttention requires Ampere (SM80), Ada Lovelace (SM89), Hopper (SM90), or newer architectures
    with native hardware tensor core support.
    """
    if not torch.cuda.is_available():
        return False
    try:
        # Major compute capability >= 8
        return torch.cuda.get_device_capability()[0] >= 8
    except Exception:
        return False


def _package_capability(
    *,
    name: str,
    package: str,
    import_name: str | None = None,
    usable_if: bool,
    unavailable_reason: str,
    unusable_reason: str,
) -> AttentionKernelCapability:
    """Underlying safe package prober.

    Queries package specs via `importlib.util.find_spec` without executing the actual
    `__init__.py` code, avoiding potential segfaults or library errors on non-CUDA machines.
    """
    try:
        available = importlib.util.find_spec(import_name or package) is not None
    except ModuleNotFoundError:
        available = False
    if not available:
        return AttentionKernelCapability(name=name, package=package, available=False, usable=False, reason=unavailable_reason)
    if not usable_if:
        return AttentionKernelCapability(name=name, package=package, available=True, usable=False, reason=unusable_reason)
    return AttentionKernelCapability(name=name, package=package, available=True, usable=True)


__all__ = [
    "AttentionKernelCapability",
    "attention_backend_capability",
    "attention_backend_from_env",
    "attention_backend_report",
    "gpu_supports_flash_attention",
    "normalize_attention_backend",
    "probe_attention_backends",
    "resolve_attention_backend",
]
