"""Module for base_models -> diffusion_model -> video -> wan -> wan_2p1 -> modules -> __init__.py functionality."""

from importlib import import_module

__all__ = [
    "GanAttentionBlock",
    "HuggingfaceTokenizer",
    "RegisterTokens",
    "T5Decoder",
    "T5Encoder",
    "T5EncoderModel",
    "T5Model",
    "VaceWanModel",
    "WanModel",
    "WanVAE",
    "flash_attention",
]

_EXPORTS = {
    "flash_attention": ".attention",
    "GanAttentionBlock": ".action_model",
    "RegisterTokens": ".action_model",
    "WanModel": ".model",
    "T5Model": ".t5",
    "T5Encoder": ".t5",
    "T5Decoder": ".t5",
    "T5EncoderModel": ".t5",
    "HuggingfaceTokenizer": ".tokenizers",
    "VaceWanModel": ".vace_model",
    "WanVAE": ".vae",
}


def __getattr__(name):
    """Getattr.

    Args:
        name: The name.
    """
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
