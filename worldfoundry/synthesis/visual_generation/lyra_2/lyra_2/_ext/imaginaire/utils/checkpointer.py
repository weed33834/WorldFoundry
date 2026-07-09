from __future__ import annotations

from typing import List, NamedTuple, Tuple

import torch

from lyra_2._ext.imaginaire.utils import log


TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 11):
    from torch.ao import quantization
    from torch.ao.quantization import FakeQuantizeBase, ObserverBase
elif (
    TORCH_VERSION >= (1, 8)
    and hasattr(torch.quantization, "FakeQuantizeBase")
    and hasattr(torch.quantization, "ObserverBase")
):
    from torch import quantization
    from torch.quantization import FakeQuantizeBase, ObserverBase


class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]),
        ],
    )
):
    pass


def non_strict_load_model(model: torch.nn.Module, checkpoint_state_dict: dict) -> _IncompatibleKeys:
    model_state_dict = model.state_dict()
    incorrect_shapes = []
    for key in list(checkpoint_state_dict.keys()):
        if key not in model_state_dict:
            continue
        if "_extra_state" in key:
            log.warning(f"Skipping TransformerEngine FP8 extra state key {key}.")
            continue
        model_param = model_state_dict[key]
        if TORCH_VERSION >= (1, 8) and isinstance(model_param, torch.nn.parameter.UninitializedParameter):
            continue
        if not isinstance(model_param, torch.Tensor):
            raise ValueError(f"Model state for {key} is not a tensor: {type(model_param)}")

        shape_model = tuple(model_param.shape)
        shape_checkpoint = tuple(checkpoint_state_dict[key].shape)
        if shape_model == shape_checkpoint:
            continue

        has_observer_base_classes = (
            TORCH_VERSION >= (1, 8)
            and "ObserverBase" in globals()
            and "FakeQuantizeBase" in globals()
        )
        if has_observer_base_classes:
            module = model
            for key_part in key.split(".")[:-1]:
                module = getattr(module, key_part)
            if isinstance(module, (ObserverBase, FakeQuantizeBase)):
                continue

        incorrect_shapes.append((key, shape_checkpoint, shape_model))
        checkpoint_state_dict.pop(key)

    incompatible = model.load_state_dict(checkpoint_state_dict, strict=False)
    missing_keys = [key for key in incompatible.missing_keys if "_extra_state" not in key]
    unexpected_keys = [key for key in incompatible.unexpected_keys if "_extra_state" not in key]
    return _IncompatibleKeys(
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        incorrect_shapes=incorrect_shapes,
    )
