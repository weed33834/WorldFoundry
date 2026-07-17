"""Lightweight, in-tree accelerator platform abstraction."""

from .base import BasePlatformProvider, PlatformProvider
from .detect import detect_accelerators
from .providers import (
    CpuPlatformProvider,
    CudaPlatformProvider,
    MpsPlatformProvider,
    RocmPlatformProvider,
    XpuPlatformProvider,
    builtin_accelerator_providers,
)
from .types import AcceleratorDescriptor, CapabilitySet, MemoryInfo, PlatformKind

__all__ = [
    "AcceleratorDescriptor",
    "BasePlatformProvider",
    "CapabilitySet",
    "CpuPlatformProvider",
    "CudaPlatformProvider",
    "MemoryInfo",
    "MpsPlatformProvider",
    "PlatformKind",
    "PlatformProvider",
    "RocmPlatformProvider",
    "XpuPlatformProvider",
    "builtin_accelerator_providers",
    "detect_accelerators",
]
