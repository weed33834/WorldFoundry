from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover - defensive import hook.
    torch = None

if torch is not None and not hasattr(torch.library, "wrap_triton"):
    def _wrap_triton(kernel):
        return kernel

    torch.library.wrap_triton = _wrap_triton
