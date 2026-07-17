"""Wan base runtime backed by in-tree foundation model implementations."""

from importlib import import_module

__all__ = [
    "WAN_VARIANTS",
    "Wan",
    "WanTextEncoder",
    "WanVAEWrapper",
    "WanVariant",
    "available_wan_variants",
    "get_wan_variant",
    "import_wan_variant_symbol",
    "wan_variant_module",
    "wan_variant_package",
    "wan_variant_root",
]

_EXPORTS = {
    "Wan": "wan_runtime_wrapper",
    "WanTextEncoder": "runtime_components",
    "WanVAEWrapper": "runtime_components",
    "WAN_VARIANTS": "registry",
    "WanVariant": "registry",
    "available_wan_variants": "registry",
    "get_wan_variant": "registry",
    "import_wan_variant_symbol": "registry",
    "wan_variant_module": "registry",
    "wan_variant_package": "registry",
    "wan_variant_root": "registry",
}


def __getattr__(name: str):
    """Getattr.

    Args:
        name: The name.
    """
    if name in _EXPORTS:
        module = import_module(f"{__name__}.{_EXPORTS[name]}")
        return getattr(module, name)
    raise AttributeError(name)
