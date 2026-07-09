# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# Sparse (2:4) linear layers — optional, requires xformers with cusparselt backend.
# Import only when 2:4 sparsity is explicitly enabled.

"""Module for base_models -> perception_core -> general_perception -> dinov3 -> layers -> sparse_linear.py functionality."""

try:
    import logging
    from typing import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import xformers.ops as xops

    from ..utils import named_apply, named_replace

    logger = logging.getLogger("dinov3")

    class LinearW24(torch.nn.Linear):
        """Linear implementation."""
        ALGO = "largest_abs_values_greedy"

        def __init__(self, *args, **kwargs) -> None:
            """Init.

            Returns:
                The return value.
            """
            super().__init__(*args, **kwargs)
            self.sparsity_enabled = False

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Forward.

            Args:
                input: The input.

            Returns:
                The return value.
            """
            if not self.sparsity_enabled:
                return super().forward(input)
            input_shape = input.shape
            input = input.flatten(end_dim=-2)
            dim0 = input.shape[0]
            if dim0 % 8 != 0:
                input = F.pad(input, [0, 0, 0, -dim0 % 8])
            w_sparse = xops.sparsify24(
                self.weight,
                algo=self.ALGO,
                gradient="ste",
                backend="cusparselt",
            )
            return F.linear(input, w_sparse, self.bias)[:dim0].unflatten(dim=0, sizes=input_shape[:-1])

    def replace_linears_with_sparse_linear(root_module: nn.Module, *, filter_fn: Callable[[str], bool]) -> nn.Module:
        """Replace linears with sparse linear.

        Args:
            root_module: The root module.

        Returns:
            The return value.
        """
        total_count = 0

        def replace(module: nn.Module, name: str) -> nn.Module:
            """Replace.

            Args:
                module: The module.
                name: The name.

            Returns:
                The return value.
            """
            nonlocal total_count
            if not isinstance(module, nn.Linear) or not filter_fn(name):
                return module
            assert type(module) == nn.Linear, "Subtypes not supported"
            new_module = LinearW24(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                dtype=module.weight.dtype,
                device=module.weight.device,
            )
            new_module.weight = module.weight
            new_module.bias = module.bias
            total_count += 1
            return new_module

        out = named_replace(replace, root_module)
        assert total_count > 0, "2:4 sparsity: no layer found to sparsify"
        return out

    def update_24sparsity(root_module: nn.Module, enabled: bool) -> int:
        """Update 24sparsity.

        Args:
            root_module: The root module.
            enabled: The enabled.

        Returns:
            The return value.
        """
        num_modified = 0

        def maybe_apply_sparsity(module: nn.Module, name: str) -> nn.Module:
            """Maybe apply sparsity.

            Args:
                module: The module.
                name: The name.

            Returns:
                The return value.
            """
            nonlocal num_modified
            if not isinstance(module, LinearW24):
                return module
            num_modified += 1
            module.sparsity_enabled = enabled
            logger.info(f"- {'' if module.sparsity_enabled else 'de'}sparsifying {name}")
            return module

        named_apply(maybe_apply_sparsity, root_module)
        torch._dynamo.reset_code_caches()
        return num_modified

except ImportError:
    # xformers not available — provide stubs that raise at runtime
    class _SparseStub:
        """Sparse stub implementation."""
        def __getattr__(self, name):
            """Getattr.

            Args:
                name: The name.
            """
            raise RuntimeError(
                "2:4 sparse linear layers require xformers with cusparselt backend. "
                "Install xformers or disable 2:4 sparsity."
            )

    LinearW24 = _SparseStub()
    replace_linears_with_sparse_linear = _SparseStub()
    update_24sparsity = _SparseStub()
