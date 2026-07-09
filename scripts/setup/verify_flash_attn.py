from __future__ import annotations

import importlib
import sys


def main() -> int:
    try:
        flash_attn = importlib.import_module("flash_attn")
    except Exception as exc:
        print(f"flash_attn import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        import torch
        from flash_attn import flash_attn_func
    except Exception as exc:
        print(f"flash_attn kernel imports failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    version = getattr(flash_attn, "__version__", "unknown")
    print(f"flash_attn import ok: {version}")
    if not torch.cuda.is_available():
        print("CUDA is not available; skipped flash_attn GPU kernel check.")
        return 0

    try:
        q = torch.randn(1, 16, 4, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"flash_attn GPU kernel failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if out.shape != q.shape:
        print(f"flash_attn GPU kernel returned unexpected shape: {tuple(out.shape)}", file=sys.stderr)
        return 1
    print("flash_attn GPU kernel ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
