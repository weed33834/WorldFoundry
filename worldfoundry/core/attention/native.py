# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SDPA-backed attention with selectable QKV layout and kernel backend."""

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

try:
    from torch.distributed.tensor.device_mesh import DeviceMesh
except ImportError:  # torch 2.4 exposes DeviceMesh from this module.
    from torch.distributed.device_mesh import DeviceMesh

try:
    from torch.distributed.tensor.experimental import context_parallel as _context_parallel
except ImportError:
    _context_parallel = None


@dataclass(frozen=True)
class AttentionBackendInfo:
    """Resolved attention backend metadata."""

    backend: str
    uses_torch_sdpa: bool


def attention_backend_info() -> AttentionBackendInfo:
    """Return the available generic PyTorch attention backend."""

    has_sdpa = callable(getattr(F, "scaled_dot_product_attention", None))
    backend = "torch.scaled_dot_product_attention" if has_sdpa else "torch.einsum"
    return AttentionBackendInfo(backend=backend, uses_torch_sdpa=has_sdpa)


def attention_backend_context(
    backend: Literal["math", "efficient", "cudnn", "flash"] | Any | None = None,
    *,
    backends: Any = None,
) -> Any:
    """Return a context manager that selects PyTorch SDPA backends via core."""

    return _sdpa_kernel_context(backend=backend, backends=backends)


def scaled_dot_product_attention(
    query: Any,
    key: Any,
    value: Any,
    *args: Any,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    backend: Literal["math", "efficient", "cudnn", "flash"] | Any | None = None,
    backends: Any = None,
) -> Any:
    """Compute scaled dot-product attention with PyTorch SDPA when available."""

    if args:
        if len(args) > 4:
            raise TypeError(f"scaled_dot_product_attention expected at most 4 positional options, got {len(args)}")
        if len(args) >= 1:
            attn_mask = args[0]
        if len(args) >= 2:
            dropout_p = args[1]
        if len(args) >= 3:
            is_causal = args[2]
        if len(args) >= 4:
            scale = args[3]

    sdpa = getattr(F, "scaled_dot_product_attention", None)
    if callable(sdpa):
        kwargs: dict[str, Any] = {
            "attn_mask": attn_mask,
            "dropout_p": float(dropout_p),
            "is_causal": bool(is_causal),
        }
        if scale is not None:
            kwargs["scale"] = float(scale)
        if enable_gqa:
            kwargs["enable_gqa"] = True
        context = _sdpa_kernel_context(backend=backend, backends=backends)
        with context:
            try:
                return sdpa(query, key, value, **kwargs)
            except TypeError:
                if "enable_gqa" not in kwargs:
                    raise
                key, value = _repeat_key_value_for_gqa(key, value, query)
                kwargs.pop("enable_gqa", None)
                return sdpa(query, key, value, **kwargs)

    if enable_gqa:
        key, value = _repeat_key_value_for_gqa(key, value, query)
    dim = int(query.shape[-1])
    factor = (dim**-0.5) if scale is None else float(scale)
    scores = torch.matmul(query, key.transpose(-2, -1)) * factor
    if attn_mask is not None:
        if getattr(attn_mask, "dtype", None) is torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    if is_causal:
        q_len = int(query.shape[-2])
        k_len = int(key.shape[-2])
        causal = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).tril(diagonal=k_len - q_len)
        scores = scores.masked_fill(~causal, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    if dropout_p:
        weights = F.dropout(weights, p=float(dropout_p), training=True)
    return torch.matmul(weights, value)


def _repeat_key_value_for_gqa(key: Any, value: Any, query: Any) -> tuple[Any, Any]:
    query_heads = int(query.shape[1])
    key_value_heads = int(key.shape[1])
    if key_value_heads == query_heads:
        return key, value
    if query_heads % key_value_heads:
        raise ValueError(f"Cannot expand {key_value_heads} KV heads to {query_heads} query heads.")
    repeats = query_heads // key_value_heads
    return (
        key.repeat_interleave(repeats, dim=1),
        value.repeat_interleave(repeats, dim=1),
    )


def _sdpa_kernel_context(*, backend: Any = None, backends: Any = None) -> Any:
    requested = backends if backends is not None else backend
    if requested is None:
        return nullcontext()
    attention = getattr(torch.nn, "attention", None)
    sdpa_kernel = getattr(attention, "sdpa_kernel", None) if attention is not None else None
    if not callable(sdpa_kernel):
        return nullcontext()
    resolved = _resolve_sdpa_backends(requested)
    if not resolved:
        return nullcontext()
    try:
        return sdpa_kernel(backends=resolved, set_priority_order=True)
    except TypeError:
        try:
            return sdpa_kernel(backends=resolved)
        except TypeError:
            return sdpa_kernel(resolved)


def _resolve_sdpa_backends(requested: Any) -> list[Any]:
    if isinstance(requested, (str, bytes)) or not isinstance(requested, (list, tuple, set, frozenset)):
        values = [requested]
    else:
        values = list(requested)

    backend_type = getattr(getattr(torch.nn, "attention", None), "SDPBackend", None)
    backend_map = {
        "math": getattr(backend_type, "MATH", None) if backend_type is not None else None,
        "efficient": getattr(backend_type, "EFFICIENT_ATTENTION", None) if backend_type is not None else None,
        "cudnn": getattr(backend_type, "CUDNN_ATTENTION", None) if backend_type is not None else None,
        "flash": getattr(backend_type, "FLASH_ATTENTION", None) if backend_type is not None else None,
    }
    resolved: list[Any] = []
    for value in values:
        if isinstance(value, str):
            value = backend_map.get(value.strip().lower().replace("_", "-"))
        if value is not None:
            resolved.append(value)
    return resolved


class NativeAttention(torch.nn.Module):
    """Native attention module with configurable QKV layout and SDPA backend."""

    def __init__(
        self,
        qkv_format: Literal["bhsd", "bshd"] = "bhsd",
        backend: Literal["math", "efficient", "cudnn", "flash"] = "cudnn",
    ) -> None:
        """Configure attention format and backend.

        Args:
            qkv_format: Layout of the QKV tensors; ``"bhsd"`` is ``(B, H, S, D)``,
                ``"bshd"`` is ``(B, S, H, D)``.
            backend: SDPA backend selected via ``sdpa_kernel``.
        """
        super().__init__()
        assert qkv_format in ["bhsd", "bshd"], f"Invalid qkv format: {qkv_format}"
        assert backend in ["math", "efficient", "cudnn", "flash"], (
            f"Invalid backend: {backend}"
        )
        self.qkv_format = qkv_format
        self.backend = backend
        self.device_mesh: DeviceMesh | None = None

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Enable or disable context parallelism for ring attention.

        Args:
            cp_group: Process group for context parallel; use None to disable.
        """
        if cp_group is None:
            self.device_mesh = None
        else:
            if _context_parallel is None:
                raise RuntimeError(
                    "NativeAttention context parallel requires "
                    "torch.distributed.tensor.experimental.context_parallel."
                )
            self.device_mesh = DeviceMesh.from_group(cp_group, device_type="cuda")

            # Need to disable load balance for torch context parallel to work.
            from torch.distributed.tensor.experimental._attention import (
                _cp_options,
                set_rotate_method,
            )

            _cp_options.enable_load_balance = False
            set_rotate_method("allgather")

    def is_context_parallel_enabled(self) -> bool:
        """Return True if context parallelism is active."""
        return self.device_mesh is not None

    def context_parallel_size(self) -> int:
        """Return the context parallel world size, or 1 if disabled."""
        return self.device_mesh.size() if self.device_mesh is not None else 1

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Run context-parallel SDPA (or single-rank SDPA when CP is disabled).

        Args:
            query: Query tensor in configured ``qkv_format``.
            key: Key tensor in configured ``qkv_format``.
            value: Value tensor in configured ``qkv_format``.

        Returns:
            Attention output in the same format as inputs.
        """
        # SDPA / low-level ops expect (B, H, S, D). "bshd" is (B, S, H, D) → transpose once.
        if self.qkv_format == "bshd":
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        out = self._impl(query=query, key=key, value=value)
        if self.qkv_format == "bshd":
            out = out.transpose(1, 2)
        return out

    def _impl(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """Attention implementation.

        Args:
            query: Query tensor, shape ``[B, H, S, D]`` (CP-shared).
            key: Key tensor, shape ``[B, H, S, D]`` (CP-sharded).
            value: Value tensor, shape ``[B, H, S, D]`` (CP-sharded).

        Returns:
            Attention output.
        """
        sdpa_backend = {
            "math": torch.nn.attention.SDPBackend.MATH,
            "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            "cudnn": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
            "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        }[self.backend]

        with torch.nn.attention.sdpa_kernel(sdpa_backend):
            if self.device_mesh is not None:
                # Pass a dummy buffer to satisfy context_parallel's buffers[0].device
                # check (required in PyTorch 2.9+ where buffers cannot be empty).
                _dummy = torch.empty(self.device_mesh.size(), device=query.device)
                with _context_parallel(
                    self.device_mesh,
                    buffers=[
                        _dummy,
                    ],
                    buffer_seq_dims=[
                        0,
                    ],
                    no_restore_buffers={_dummy},
                ):
                    out = F.scaled_dot_product_attention(query, key, value)
            else:
                out = F.scaled_dot_product_attention(query, key, value)

        return out


__all__ = [
    "AttentionBackendInfo",
    "NativeAttention",
    "attention_backend_info",
    "scaled_dot_product_attention",
]
