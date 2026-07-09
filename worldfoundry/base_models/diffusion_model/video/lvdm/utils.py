"""Shared utility helpers for LVDM foundation models."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping

import numpy as np
import torch
import torch.distributed as dist


def get_obj_from_str(string: str, reload: bool = False, invalidate_cache: bool = False):
    """Get obj from str.

    Args:
        string: The string.
        reload: The reload.
        invalidate_cache: When True, flush import caches before resolving the target.
    """
    module_path, object_name = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module = importlib.import_module(module_path)
        importlib.reload(module)
    return getattr(importlib.import_module(module_path), object_name)


def instantiate_from_config(config, **additional_kwargs):
    """Instantiate from config.

    Args:
        config: Mapping with ``target`` or ``class_path`` plus optional ``params`` or ``init_args``.
    """
    if isinstance(config, str) and config in {"__is_first_stage__", "__is_unconditional__"}:
        return None
    if not isinstance(config, Mapping):
        raise KeyError("Expected key `target` or `class_path` to instantiate.")

    target = config.get("target") or config.get("class_path")
    if not target:
        raise KeyError("Expected key `target` or `class_path` to instantiate.")

    params = config.get("params") or config.get("init_args") or {}
    if additional_kwargs:
        merged = dict(params)
        merged.update(additional_kwargs)
        params = merged
    return get_obj_from_str(str(target))(**params)


def count_params(model, verbose: bool = False):
    """Count params.

    Args:
        model: The model.
        verbose: The verbose.
    """
    total_params = sum(param.numel() for param in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def check_istarget(name: str, para_list: list[str]):
    """Check istarget.

    Args:
        name: The name.
        para_list: The para list.
    """
    for para in para_list:
        if para in name:
            return True
    return False


def load_npz_from_dir(data_dir: str) -> np.ndarray:
    data = [np.load(os.path.join(data_dir, data_name))["arr_0"] for data_name in os.listdir(data_dir)]
    return np.concatenate(data, axis=0)


def load_npz_from_paths(data_paths: list[str]) -> np.ndarray:
    data = [np.load(data_path)["arr_0"] for data_path in data_paths]
    return np.concatenate(data, axis=0)


def resize_numpy_image(image, max_resolution: int = 512 * 512, resize_short_edge=None):
    import cv2

    height, width = image.shape[:2]
    if resize_short_edge is not None:
        scale = resize_short_edge / min(height, width)
    else:
        scale = max_resolution / (height * width)
        scale = scale**0.5
    height = int(np.round(height * scale / 64)) * 64
    width = int(np.round(width * scale / 64)) * 64
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)


def setup_dist(args) -> None:
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group("nccl", init_method="env://")
