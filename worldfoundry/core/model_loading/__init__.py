"""Model loading helpers shared by WorldFoundry runtime integrations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "InferenceModel": "worldfoundry.core.model_loading.inference_model",
    "ModelConfig": "worldfoundry.core.model_loading.config",
    "convert_keys_dict_to_single_str": "worldfoundry.core.model_loading.file",
    "convert_state_dict_keys_to_single_str": "worldfoundry.core.model_loading.file",
    "convert_state_dict_to_keys_dict": "worldfoundry.core.model_loading.file",
    "get_init_context": "worldfoundry.core.model_loading.model",
    "hash_model_file": "worldfoundry.core.model_loading.file",
    "hash_state_dict_keys": "worldfoundry.core.model_loading.file",
    "build_rename_dict": "worldfoundry.core.model_loading.file",
    "load_keys_dict": "worldfoundry.core.model_loading.file",
    "load_model": "worldfoundry.core.model_loading.model",
    "load_model_loader_registry": "worldfoundry.core.model_loading.registry_config",
    "load_model_with_disk_offload": "worldfoundry.core.model_loading.model",
    "load_state_dict": "worldfoundry.core.model_loading.file",
    "load_state_dict_non_strict": "worldfoundry.core.model_loading.state_dict",
    "load_state_dict_from_folder": "worldfoundry.core.model_loading.file",
    "load_state_dict_from_safetensors_index": "worldfoundry.core.model_loading.file",
    "load_torch_checkpoint": "worldfoundry.core.model_loading.file",
    "load_torch_state_dict": "worldfoundry.core.model_loading.file",
    "non_strict_load_model": "worldfoundry.core.model_loading.state_dict",
    "search_for_embeddings": "worldfoundry.core.model_loading.file",
    "search_for_files": "worldfoundry.core.model_loading.file",
    "search_parameter": "worldfoundry.core.model_loading.file",
    "split_state_dict_with_prefix": "worldfoundry.core.model_loading.file",
    "ModelLoaderRegistry": "worldfoundry.core.model_loading.registry_config",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = sorted(_EXPORT_MODULES)
