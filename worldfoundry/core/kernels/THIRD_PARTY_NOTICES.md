# Third-party kernel notices

## LingBot-VLA v2 packed routed MoE

`triton_moe.py` is adapted from the grouped routed-MoE inference path in
Robbyant LingBot-VLA v2. The WorldFoundry version exposes a model-independent
packed SwiGLU interface, uses the shared GPU capability policy, and retains a
portable PyTorch fallback for unsupported devices.

Source: https://github.com/Robbyant/lingbot-vla-v2

Copyright 2026 Robbyant Team and/or its affiliates. Licensed under the Apache
License 2.0.

## SGLang GroupNorm + SiLU

`triton_group_norm_silu.py` is adapted from
`sglang/jit_kernel/diffusion/triton/group_norm_silu.py` in SGLang. SGLang is
licensed under the Apache License 2.0. The WorldFoundry version removes the
SGLang custom-op/runtime dependency, renames internal symbols, and exposes the
kernel through WorldFoundry's workload registry and PyTorch fallback.

Source: https://github.com/sgl-project/sglang

## NVIDIA Sol-Engine / SGLang diffusion acceleration policies

The generic fixed-step cache, accumulated-change residual cache, and
feature-norm token-pruning policies in `worldfoundry/core/acceleration/` are
adapted from NVIDIA Sol-Engine's SGLang runtime:

- https://github.com/NVlabs/Sana/tree/sol-engine
- `python/sglang/multimodal_gen/runtime/efficiency/techniques/step_cache.py`
- `python/sglang/multimodal_gen/runtime/efficiency/techniques/token_prune.py`

Copyright 2025 SGLang authors. These sources are licensed under the Apache
License 2.0. WorldFoundry's adaptations are self-contained and do not import
SGLang or the Sana repository at runtime.

## PISA piecewise sparse attention

`worldfoundry/core/attention/triton_piecewise_attention.py` is adapted
from Sol-Engine's `piecewise_attn.py`, which in turn implements piecewise sparse
attention by Haopeng Li. The WorldFoundry copy retains the Triton/TMA routing,
forward, and backward kernels while removing SGLang backend/runtime coupling.

Copyright (c) 2025-2026, Haopeng Li. Licensed under the MIT License.

Sources:

- https://github.com/NVlabs/Sana/tree/sol-engine
- https://github.com/xie-lab-ml/piecewise-sparse-attention

## TorchAO NVFP4 reference algorithms

`worldfoundry/core/acceleration/nvfp4.py` adapts the FP4 E2M1 round-to-nearest-
even encoder, nibble packing, NVFP4 two-level scaling, and 128x4 scale swizzle
from TorchAO commit `bfbc842047452e13e3292646656b307f5947e815`.

Source: https://github.com/pytorch/ao

Copyright 2023 Meta

All contributions by Arm: Copyright (c) 2024-2026 Arm Limited and/or its
affiliates

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
