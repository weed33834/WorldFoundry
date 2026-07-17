"""Accelerator capability and policy profiles for in-tree kernels.

The profiles are deliberately feature-based rather than a whitelist of product
names.  This keeps mixed GPU nodes correct and makes future CUDA devices fall
back safely when their exact architecture has not been tuned yet.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True, slots=True)
class KernelDeviceProfile:
    """Hardware facts and conservative eager-kernel thresholds."""

    device: str
    name: str
    runtime: str
    family: str
    compute_capability: tuple[int, int] | None
    total_memory: int | None
    multiprocessors: int | None
    shared_memory_per_multiprocessor: int | None
    warp_size: int | None
    supports_triton: bool
    supports_fp16: bool
    supports_bf16: bool
    supports_fp8: bool
    supports_fp4: bool
    residual_gate_min_elements: int
    adaln_min_elements: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.compute_capability is not None:
            payload["compute_capability"] = list(self.compute_capability)
        return payload


def _cuda_family(capability: tuple[int, int] | None) -> str:
    if capability is None:
        return "cuda-unknown"
    major, minor = capability
    if major == 7:
        return "volta" if minor == 0 else "turing"
    if major == 8:
        return "ada" if minor == 9 else "ampere"
    if major == 9:
        return "hopper"
    if major == 10:
        return "blackwell-datacenter"
    if major == 12:
        return "blackwell-consumer"
    return f"cuda-sm{major}{minor}"


def _default_thresholds(family: str) -> tuple[int, int]:
    mib = 1024 * 1024
    if family in {"volta", "turing", "cuda-unknown", "rocm"} or family.startswith("cuda-sm"):
        # Legacy/untested targets prefer vendor kernels until enough work is
        # available to amortise a custom launch.
        return 16 * mib, 8 * mib
    if family in {"blackwell-datacenter", "blackwell-consumer"}:
        # Native Blackwell pointwise/reduction kernels are very fast.  These
        # thresholds remain conservative until real B200/SM120 data is cached.
        return 16 * mib, 8 * mib
    if family == "hopper":
        return 12 * mib, 6 * mib
    # Ampere and Ada share the currently validated baseline.
    return 8 * mib, 4 * mib


def default_kernel_thresholds(capability: tuple[int, int] | None) -> tuple[int, int]:
    """Return ``(residual_gate, AdaLN)`` thresholds for a CUDA capability."""

    return _default_thresholds(_cuda_family(capability))


def _normalise_cuda_device(device: torch.device | str | int | None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    if isinstance(device, int):
        return torch.device("cuda", device)
    parsed = torch.device(device)
    if parsed.type == "cuda" and parsed.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return parsed


def kernel_device_profile(device: torch.device | str | int | None = None) -> KernelDeviceProfile:
    """Return a cached profile for one concrete accelerator device."""

    parsed = _normalise_cuda_device(device)
    if parsed.type != "cuda" or not torch.cuda.is_available():
        return KernelDeviceProfile(
            device=str(parsed),
            name=parsed.type,
            runtime=parsed.type,
            family=parsed.type,
            compute_capability=None,
            total_memory=None,
            multiprocessors=None,
            shared_memory_per_multiprocessor=None,
            warp_size=None,
            supports_triton=False,
            supports_fp16=parsed.type not in {"cpu", "mps"},
            supports_bf16=False,
            supports_fp8=False,
            supports_fp4=False,
            residual_gate_min_elements=2**63 - 1,
            adaln_min_elements=2**63 - 1,
        )
    index = torch.cuda.current_device() if parsed.index is None else int(parsed.index)
    allow_untested = os.getenv("WORLDFOUNDRY_ALLOW_UNTESTED_GPU_KERNELS", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return _cuda_profile_cached(index, bool(getattr(torch.version, "hip", None)), allow_untested)


@lru_cache(maxsize=64)
def _cuda_profile_cached(index: int, hip: bool, allow_untested: bool) -> KernelDeviceProfile:
    properties = torch.cuda.get_device_properties(index)
    name = str(getattr(properties, "name", f"GPU {index}"))
    total_memory = int(getattr(properties, "total_memory", 0)) or None
    multiprocessors = int(getattr(properties, "multi_processor_count", 0)) or None
    shared_memory_per_multiprocessor = (
        int(getattr(properties, "shared_memory_per_multiprocessor", 0)) or None
    )
    warp_size = int(getattr(properties, "warp_size", 0)) or None
    if hip:
        family = "rocm"
        capability = None
        residual_min, adaln_min = _default_thresholds(family)
        return KernelDeviceProfile(
            device=f"cuda:{index}",
            name=name,
            runtime="rocm",
            family=family,
            compute_capability=capability,
            total_memory=total_memory,
            multiprocessors=multiprocessors,
            shared_memory_per_multiprocessor=shared_memory_per_multiprocessor,
            warp_size=warp_size,
            # The present kernels use CUDA-specific validation and have not
            # yet been qualified on CDNA/RDNA. PyTorch remains the safe path.
            supports_triton=False,
            supports_fp16=True,
            supports_bf16=bool(torch.cuda.is_bf16_supported()),
            supports_fp8=False,
            supports_fp4=False,
            residual_gate_min_elements=residual_min,
            adaln_min_elements=adaln_min,
        )

    major, minor = torch.cuda.get_device_capability(index)
    capability = (int(major), int(minor))
    family = _cuda_family(capability)
    residual_min, adaln_min = _default_thresholds(family)
    supports_bf16 = capability >= (8, 0)
    supports_fp8 = capability >= (8, 9)
    supports_fp4 = capability >= (10, 0)
    known_kernel_capabilities = {
        (7, 0),
        (7, 5),
        (8, 0),
        (8, 6),
        (8, 7),
        (8, 9),
        (9, 0),
        (10, 0),
        (10, 3),
        (12, 0),
        (12, 1),
    }
    return KernelDeviceProfile(
        device=f"cuda:{index}",
        name=name,
        runtime="cuda",
        family=family,
        compute_capability=capability,
        total_memory=total_memory,
        multiprocessors=multiprocessors,
        shared_memory_per_multiprocessor=shared_memory_per_multiprocessor,
        warp_size=warp_size,
        supports_triton=capability in known_kernel_capabilities or allow_untested,
        supports_fp16=capability >= (5, 3),
        supports_bf16=supports_bf16,
        supports_fp8=supports_fp8,
        supports_fp4=supports_fp4,
        residual_gate_min_elements=residual_min,
        adaln_min_elements=adaln_min,
    )


def profile_supports_dtype(profile: KernelDeviceProfile, dtype: torch.dtype) -> bool:
    if dtype == torch.float16:
        return profile.supports_fp16
    if dtype == torch.bfloat16:
        return profile.supports_bf16
    if dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
        return profile.supports_fp8
    return dtype == torch.float32


def triton_tensor_eligible(tensor: torch.Tensor) -> bool:
    """Return whether the current in-tree Triton kernels support ``tensor``."""

    if not tensor.is_cuda:
        return False
    profile = kernel_device_profile(tensor.device)
    return profile.supports_triton and profile_supports_dtype(profile, tensor.dtype)


def detected_kernel_device_profiles() -> tuple[KernelDeviceProfile, ...]:
    """Describe all visible CUDA/ROCm devices without assuming homogeneity."""

    if not torch.cuda.is_available():
        return (kernel_device_profile(torch.device("cpu")),)
    return tuple(kernel_device_profile(index) for index in range(torch.cuda.device_count()))


__all__ = [
    "KernelDeviceProfile",
    "default_kernel_thresholds",
    "detected_kernel_device_profiles",
    "kernel_device_profile",
    "profile_supports_dtype",
    "triton_tensor_eligible",
]
