"""Device detection and torch backend helpers for CPU, CUDA, and NPU."""

from typing import Any

import importlib
import os

import torch


def is_torch_npu_available() -> bool:
    """Return True when ``torch_npu`` is installed and NPU runtime is available."""
    return importlib.util.find_spec("torch_npu") is not None


IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_NPU_AVAILABLE = is_torch_npu_available() and hasattr(torch, "npu") and torch.npu.is_available()

if IS_NPU_AVAILABLE:
    import torch_npu

    torch.npu.config.allow_internal_format = False


def get_device_type() -> str:
    """Get device type based on current machine, currently only support CPU, CUDA, NPU."""
    if IS_CUDA_AVAILABLE:
        device = "cuda"
    elif IS_NPU_AVAILABLE:
        device = "npu"
    else:
        device = "cpu"

    return device


def get_torch_device() -> Any:
    """Get torch attribute based on device type, e.g. torch.cuda or torch.npu"""
    device_name = get_device_type()

    try:
        return getattr(torch, device_name)
    except AttributeError:
        print(f"Device namespace '{device_name}' not found in torch, try to load 'torch.cuda'.")
        return torch.cuda


def get_device_id() -> int:
    """Get current device id based on device type."""
    return get_torch_device().current_device()


def get_device_name() -> str:
    """Get current device name based on device type."""
    return f"{get_device_type()}:{get_device_id()}"


def synchronize() -> None:
    """Execute torch synchronize operation."""
    get_torch_device().synchronize()


def empty_cache() -> None:
    """Execute torch empty cache operation."""
    get_torch_device().empty_cache()


def get_nccl_backend() -> str:
    """Return distributed communication backend type based on device type."""
    if IS_CUDA_AVAILABLE:
        return "nccl"
    elif IS_NPU_AVAILABLE:
        return "hccl"
    else:
        raise RuntimeError(f"No available distributed communication backend found on device type {get_device_type()}.")


def enable_high_precision_for_bf16() -> None:
    """Disable reduced-precision bf16 matmul accumulation on CUDA and NPU."""
    if IS_CUDA_AVAILABLE:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    if IS_NPU_AVAILABLE:
        torch.npu.matmul.allow_tf32 = False
        torch.npu.matmul.allow_bf16_reduced_precision_reduction = False


def parse_device_type(device) -> str:
    """Normalize a device string or :class:`torch.device` to ``cpu``, ``cuda``, or ``npu``."""
    if isinstance(device, str):
        if device.startswith("cuda"):
            return "cuda"
        elif device.startswith("npu"):
            return "npu"
        else:
            return "cpu"
    elif isinstance(device, torch.device):
        return device.type
    return "cpu"


def cuda_visible_devices_from_device(
    device: str | torch.device | None,
    *,
    inherited: str | None = None,
    map_inherited: bool = True,
    default_cuda: str = "0",
) -> str | None:
    """Convert a device string into a ``CUDA_VISIBLE_DEVICES`` value.

    ``cuda:N`` is interpreted as a local index into an inherited
    ``CUDA_VISIBLE_DEVICES`` list by default, which is the behavior expected by
    subprocess launchers nested under a scheduler or torchrun process.
    """

    if device is None:
        return None
    normalized = str(device).strip().lower()
    if not normalized:
        return None
    inherited_devices = os.environ.get("CUDA_VISIBLE_DEVICES") if inherited is None else inherited
    if normalized == "cuda":
        return inherited_devices if inherited_devices else default_cuda
    if normalized.startswith("cuda:"):
        suffix = normalized.split(":", 1)[1].strip()
        if not suffix:
            return None
        if "," in suffix:
            return suffix if all(part.strip().isdigit() for part in suffix.split(",") if part.strip()) else None
        if not suffix.isdigit():
            return None
        visible_index = int(suffix)
        if map_inherited and inherited_devices:
            inherited_parts = [part.strip() for part in inherited_devices.split(",") if part.strip()]
            if visible_index < len(inherited_parts):
                return inherited_parts[visible_index]
        return str(visible_index)
    if normalized.isdigit() or (
        "," in normalized and all(part.strip().isdigit() for part in normalized.split(",") if part.strip())
    ):
        return normalized
    return None


def parse_nccl_backend(device_type: str) -> str:
    """Return the distributed backend name (``nccl`` or ``hccl``) for *device_type*."""
    if device_type == "cuda":
        return "nccl"
    elif device_type == "npu":
        return "hccl"
    else:
        raise RuntimeError(f"No available distributed communication backend found on device type {device_type}.")


def get_available_device_type() -> str:
    """Return the best available device type on this host."""
    return get_device_type()


__all__ = [
    "IS_CUDA_AVAILABLE",
    "IS_NPU_AVAILABLE",
    "cuda_visible_devices_from_device",
    "empty_cache",
    "enable_high_precision_for_bf16",
    "get_available_device_type",
    "get_device_id",
    "get_device_name",
    "get_device_type",
    "get_nccl_backend",
    "get_torch_device",
    "is_torch_npu_available",
    "parse_device_type",
    "parse_nccl_backend",
    "synchronize",
]
