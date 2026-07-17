"""In-tree low-precision linear layers with portable dense fallback."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.core.kernels.capabilities import kernel_device_profile

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _quantize_tensorwise_fp8(
    value: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dtype not in _FP8_DTYPES:
        raise ValueError(f"unsupported FP8 dtype: {dtype}")
    fp8_max = float(torch.finfo(dtype).max)
    scale = (value.detach().float().abs().amax() / fp8_max).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (value.float() / scale).clamp(-fp8_max, fp8_max).to(dtype)
    return quantized, scale.reshape(1).float()


def _fp8_hardware_eligible(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    profile = kernel_device_profile(device)
    return profile.supports_fp8 and callable(getattr(torch, "_scaled_mm", None))


def _fp8_linear_eligible(
    input: torch.Tensor,
    out_features: int,
    hardware_eligible: bool,
) -> bool:
    if input.device.type != "cuda" or torch.is_grad_enabled():
        return False
    if input.dtype not in {torch.float16, torch.bfloat16}:
        return False
    if input.shape[-1] % 16 or out_features % 16:
        return False
    return hardware_eligible


class Float8Linear(nn.Module):
    """Inference-only tensorwise-FP8 linear with an exact dense fallback.

    Quantized weights are stored in row-major ``[out, in]`` form and viewed as
    the column-major right GEMM operand at execution. Dynamic activation
    quantization happens on-device. A dense weight copy is retained by default
    so the same checkpoint remains runnable on V100, T4, A100, ROCm and CPU.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        keep_dense_fallback: bool = True,
    ) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("weight must have [out_features, in_features] shape")
        if fp8_dtype not in _FP8_DTYPES:
            raise ValueError(f"unsupported FP8 dtype: {fp8_dtype}")
        quantized, scale = _quantize_tensorwise_fp8(weight.detach(), fp8_dtype)
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.fp8_dtype = fp8_dtype
        self.low_precision_enabled = True
        self.register_buffer("weight_fp8", quantized.contiguous())
        self.register_buffer("weight_scale", scale.to(device=weight.device))
        dense = weight.detach().clone() if keep_dense_fallback else None
        self.register_buffer("weight", dense)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self._hardware_eligible = _fp8_hardware_eligible(weight.device)

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        keep_dense_fallback: bool = True,
    ) -> "Float8Linear":
        return cls(
            layer.weight,
            layer.bias,
            fp8_dtype=fp8_dtype,
            keep_dense_fallback=keep_dense_fallback,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {input.shape[-1]}")
        if input.numel() == 0:
            return input.new_empty((*input.shape[:-1], self.out_features))
        if not self.low_precision_enabled or not _fp8_linear_eligible(
            input,
            self.out_features,
            self._hardware_eligible,
        ):
            if self.weight is None:
                raise RuntimeError("FP8 is unavailable for this workload and no dense fallback was retained")
            bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
            return F.linear(input, self.weight.to(dtype=input.dtype), bias)

        original_shape = input.shape
        flattened = input.reshape(-1, self.in_features).contiguous()
        input_fp8, input_scale = _quantize_tensorwise_fp8(flattened, self.fp8_dtype)
        # A contiguous [N, K] weight transposes to the column-major [K, N]
        # layout required by torch._scaled_mm/cuBLASLt without another copy.
        weight_mat = self.weight_fp8.t()
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        output = torch._scaled_mm(
            input_fp8,
            weight_mat,
            input_scale,
            self.weight_scale,
            bias=bias,
            out_dtype=input.dtype,
            use_fast_accum=False,
        )
        return output.reshape(*original_shape[:-1], self.out_features)

    def _apply(self, fn, recurse: bool = True):
        # ``Module.to(dtype=...)`` applies its dtype conversion to every
        # floating buffer.  FP8 payloads and their FP32 decode scale are
        # storage-format metadata, not model-compute tensors, so preserve the
        # original values/dtypes while still following the requested device.
        weight_fp8 = self.weight_fp8
        weight_scale = self.weight_scale
        result = super()._apply(fn, recurse=recurse)
        target_device = self.weight_fp8.device
        self.weight_fp8 = weight_fp8.to(device=target_device)
        self.weight_scale = weight_scale.to(device=target_device)
        self._hardware_eligible = _fp8_hardware_eligible(self.weight_fp8.device)
        return result

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, fp8_dtype={self.fp8_dtype}, "
            f"dense_fallback={self.weight is not None}"
        )


def replace_linear_with_float8(
    module: nn.Module,
    *,
    min_features: int = 1024,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    keep_dense_fallback: bool = True,
    exclude: tuple[str, ...] = (),
) -> int:
    """Replace eligible child ``nn.Linear`` modules in-place.

    Apply it after loading the dense checkpoint and placing its modules.  This
    explicit inference transform turns quantized weights into buffers, avoids
    hidden runtime monkey-patching, and returns the number of replacements.
    """

    replaced = 0
    for name, child in tuple(module.named_children()):
        if any(pattern and pattern in name for pattern in exclude):
            continue
        if isinstance(child, nn.Linear):
            if (
                child.in_features >= min_features
                and child.out_features >= min_features
                and child.in_features % 16 == 0
                and child.out_features % 16 == 0
            ):
                setattr(
                    module,
                    name,
                    Float8Linear.from_linear(
                        child,
                        fp8_dtype=fp8_dtype,
                        keep_dense_fallback=keep_dense_fallback,
                    ),
                )
                replaced += 1
            continue
        replaced += replace_linear_with_float8(
            child,
            min_features=min_features,
            fp8_dtype=fp8_dtype,
            keep_dense_fallback=keep_dense_fallback,
            exclude=exclude,
        )
    return replaced


def set_low_precision_enabled(module: nn.Module, enabled: bool) -> int:
    """Toggle in-tree FP8/NVFP4 modules for dense boundary denoising steps."""

    updated = 0
    for child in module.modules():
        if hasattr(child, "low_precision_enabled"):
            child.low_precision_enabled = bool(enabled)
            updated += 1
    return updated


__all__ = ["Float8Linear", "replace_linear_with_float8", "set_low_precision_enabled"]
