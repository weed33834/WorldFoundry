"""Minimal model base and state helpers required by inference runtimes."""

from __future__ import annotations

from typing import Any

import torch


class InferenceModel(torch.nn.Module):
    """Compatibility base containing inference lifecycle hooks only."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def prepare_inference(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        """Initialize weights/tokenizers; retained because upstream models use this hook for loading."""


def instantiate_inference_network(
    config: Any,
    *,
    device: str = "cuda",
    device_mesh: Any = None,
    internal_mixed_precision_policy: Any = None,
    root_mixed_precision_policy: Any = None,
    shard_before_materialize: bool = False,
) -> torch.nn.Module:
    """Instantiate, materialize, initialize, and optionally shard an inference network."""

    from worldfoundry.core.configuration.lazy_config import instantiate

    with torch.device("meta"):
        network = instantiate(config)

    internal_kwargs = (
        {"mp_policy": internal_mixed_precision_policy} if internal_mixed_precision_policy is not None else {}
    )
    root_kwargs = {"mp_policy": root_mixed_precision_policy} if root_mixed_precision_policy is not None else {}

    def shard(module: torch.nn.Module) -> torch.nn.Module:
        from torch.distributed._composable.fsdp import fully_shard

        module.fully_shard(mesh=device_mesh, **internal_kwargs)
        return fully_shard(
            module,
            mesh=device_mesh,
            reshard_after_forward=True,
            **root_kwargs,
        )

    if device_mesh is not None and shard_before_materialize:
        network = shard(network)
    network.to_empty(device=device)
    network.init_weights()
    if device_mesh is not None:
        if not shard_before_materialize:
            network = shard(network)
        from torch.distributed.tensor import DTensor

        from worldfoundry.core.distributed.device_mesh_collectives import (
            broadcast_dtensor_model_states,
        )

        broadcast_dtensor_model_states(network, device_mesh)
        for name, parameter in network.named_parameters():
            if not isinstance(parameter, DTensor):
                raise TypeError(f"expected DTensor parameter {name}, got {type(parameter)}")
    return network


__all__ = [
    "InferenceModel",
    "instantiate_inference_network",
]
