from importlib import import_module

__all__ = ["CasualWorldActionTransformer"]


def __getattr__(name: str):
    if name == "CasualWorldActionTransformer":
        return import_module(f"{__name__}.transformer_wa_casual").CasualWorldActionTransformer
    raise AttributeError(name)
