"""Runtime inventory for Sol-Engine-style optimization families."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from worldfoundry.core.acceleration.nvfp4 import _nvfp4_hardware_eligible
from worldfoundry.core.attention.piecewise import piecewise_attention_available
from worldfoundry.core.kernels import kernel_device_profile, kernel_dispatch_report


@dataclass(frozen=True, slots=True)
class AccelerationTechnology:
    family: str
    name: str
    implementation: str
    in_tree: bool
    approximate: bool
    hardware_active: bool
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _resolved_device(device: torch.device | str | int | None) -> torch.device:
    if isinstance(device, int):
        return torch.device("cuda", device)
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    return torch.device(device)


def acceleration_technology_report(
    device: torch.device | str | int | None = None,
) -> tuple[AccelerationTechnology, ...]:
    """Report real in-tree capabilities without treating optional packages as integration."""

    resolved = _resolved_device(device)
    profile = kernel_device_profile(resolved)
    kernels = kernel_dispatch_report().get("operators", {})
    fp8_active = (
        resolved.type == "cuda"
        and profile.supports_fp8
        and callable(getattr(torch, "_scaled_mm", None))
    )
    fp4_active = _nvfp4_hardware_eligible(resolved)
    pisa_active = piecewise_attention_available(resolved) if resolved.type == "cuda" else False
    fusion_names = ", ".join(sorted(str(name) for name in kernels))
    return (
        AccelerationTechnology(
            "cache",
            "fixed-step output cache",
            "worldfoundry.core.acceleration.FixedStepCache",
            True,
            True,
            True,
        ),
        AccelerationTechnology(
            "cache",
            "adaptive residual cache",
            "worldfoundry.core.acceleration.AdaptiveResidualCache",
            True,
            True,
            True,
        ),
        AccelerationTechnology(
            "token_pruning",
            "feature-norm token pruning",
            "worldfoundry.core.acceleration.TokenPruner",
            True,
            True,
            True,
        ),
        AccelerationTechnology(
            "sparse_attention",
            "PISA piecewise attention",
            "worldfoundry.core.attention.piecewise_attention",
            True,
            True,
            pisa_active,
            "TMA kernel requires SM90 or data-center Blackwell; other GPUs use exact SDPA"
            if not pisa_active
            else "",
        ),
        AccelerationTechnology(
            "quantization",
            "FP8 dynamic activation/weight linear",
            "worldfoundry.core.acceleration.Float8Linear",
            True,
            True,
            fp8_active,
            "native FP8 GEMM requires Ada/Hopper/Blackwell; dense fallback is retained"
            if not fp8_active
            else "",
        ),
        AccelerationTechnology(
            "quantization",
            "NVFP4 dynamic activation/weight linear",
            "worldfoundry.core.acceleration.NVFP4Linear",
            True,
            True,
            fp4_active,
            "native NVFP4 GEMM requires a known Blackwell target and CUDA >= 12.8; dense fallback is retained"
            if not fp4_active
            else "",
        ),
        AccelerationTechnology(
            "kernel_fusion",
            "DiT fused glue kernels",
            fusion_names or "PyTorch exact fallbacks",
            True,
            False,
            bool(kernels),
        ),
        AccelerationTechnology(
            "compilation",
            "persistent Inductor/Triton cache",
            "worldfoundry.runtime.compile_cache",
            True,
            False,
            True,
        ),
    )


__all__ = ["AccelerationTechnology", "acceleration_technology_report"]
