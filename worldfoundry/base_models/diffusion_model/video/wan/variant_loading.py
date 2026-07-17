"""Checkpoint loading shared by in-tree Wan inference variants."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Mapping

import torch

from worldfoundry.core.model_loading.file import load_state_dict


def _model_config(
    root: Path,
    additional_kwargs: Mapping | None,
) -> tuple[dict, dict]:
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Wan transformer config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    overrides = dict(additional_kwargs or {})
    mapping = overrides.pop("dict_mapping", {})
    for config_key, model_key in mapping.items():
        overrides[model_key] = config[config_key]
    return config, overrides


def _compatible_state_dict(model, state_dict: dict) -> dict:
    model_state = model.state_dict()
    patch_key = "patch_embedding.weight"
    if patch_key in state_dict and patch_key in model_state:
        source = state_dict[patch_key]
        target = model_state[patch_key]
        if source.shape != target.shape:
            adjusted = source.new_zeros(target.shape)
            common = tuple(
                slice(0, min(source_size, target_size))
                for source_size, target_size in zip(source.shape, target.shape)
            )
            adjusted[common] = source[common]
            state_dict = dict(state_dict)
            state_dict[patch_key] = adjusted
    return {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }


def _load_into_meta(model, state_dict: dict, dtype: torch.dtype, root: Path):
    from diffusers.models.modeling_utils import load_model_dict_into_meta

    model_keys = set(model.state_dict())
    missing = model_keys.difference(state_dict)
    if missing:
        raise ValueError(
            f"Checkpoint {root} is missing {len(missing)} transformer tensors."
        )
    parameters = inspect.signature(load_model_dict_into_meta).parameters
    kwargs = {"dtype": dtype}
    if "device" in parameters:
        kwargs["device"] = "cpu"
    if "model_name_or_path" in parameters:
        kwargs["model_name_or_path"] = str(root)
    load_model_dict_into_meta(model, state_dict, **kwargs)
    return model


def load_wan_transformer(
    model_class,
    pretrained_model_path: str | Path,
    *,
    subfolder: str | None = None,
    additional_kwargs: Mapping | None = None,
    low_cpu_mem_usage: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load one Diffusers-configured Wan transformer for inference."""
    root = Path(pretrained_model_path)
    if subfolder is not None:
        root /= subfolder
    config, overrides = _model_config(root, additional_kwargs)
    state_dict = load_state_dict(str(root), device="cpu")

    if low_cpu_mem_usage:
        try:
            from accelerate import init_empty_weights

            with init_empty_weights():
                model = model_class.from_config(config, **overrides)
            state_dict = _compatible_state_dict(model, state_dict)
            return _load_into_meta(model, state_dict, torch_dtype, root)
        except (ImportError, TypeError):
            pass

    model = model_class.from_config(config, **overrides)
    state_dict = _compatible_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    return model.to(dtype=torch_dtype)


class WanVariantLoadingMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        subfolder=None,
        transformer_additional_kwargs=None,
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16,
    ):
        return load_wan_transformer(
            cls,
            pretrained_model_path,
            subfolder=subfolder,
            additional_kwargs=transformer_additional_kwargs,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
        )


__all__ = ["WanVariantLoadingMixin", "load_wan_transformer"]
