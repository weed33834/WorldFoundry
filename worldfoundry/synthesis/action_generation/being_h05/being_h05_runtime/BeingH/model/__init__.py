# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module


# Note: BeingH and BeingHConfig are not imported here to avoid circular imports.
# Import them directly: from BeingH.model.beingvla import BeingH, BeingHConfig.
_EXPORTS = {
    "Qwen2Config": ("worldfoundry.base_models.llm_mllm_core.mllm.qwen.beingh.qwen2_navit", "Qwen2Config"),
    "Qwen2Model": ("worldfoundry.base_models.llm_mllm_core.mllm.qwen.beingh.qwen2_navit", "Qwen2Model"),
    "Qwen2ForCausalLM": ("worldfoundry.base_models.llm_mllm_core.mllm.qwen.beingh.qwen2_navit", "Qwen2ForCausalLM"),
    "Qwen2ForCausalLM_MLP": ("worldfoundry.base_models.llm_mllm_core.mllm.qwen.beingh.qwen2.modeling_qwen2", "Qwen2ForCausalLM"),
    "Qwen3Config": ("worldfoundry.base_models.llm_mllm_core.mllm.qwen.beingh.qwen3_navit", "Qwen3Config"),
    "Qwen3Model": ("worldfoundry.base_models.llm_mllm_core.mllm.qwen.beingh.qwen3_navit", "Qwen3Model"),
    "Qwen3ForCausalLM": ("worldfoundry.base_models.llm_mllm_core.mllm.qwen.beingh.qwen3_navit", "Qwen3ForCausalLM"),
    "SiglipVisionConfig": ("vit_model.siglip_navit", "SiglipVisionConfig"),
    "SiglipVisionModel": ("vit_model.siglip_navit", "SiglipVisionModel"),
    "InternVisionConfig": ("vit_model.internvit.modeling_intern_vit", "InternVisionConfig"),
    "InternVisionModel": ("vit_model.internvit.modeling_intern_vit", "InternVisionModel"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    import_path = module_name if module_name.startswith("worldfoundry.") else f"{__name__}.{module_name}"
    value = getattr(import_module(import_path), attr_name)
    globals()[name] = value
    return value
