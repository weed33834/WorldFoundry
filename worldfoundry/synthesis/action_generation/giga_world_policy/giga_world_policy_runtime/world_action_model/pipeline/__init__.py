from importlib import import_module

__all__ = ["WAPipeline"]


def __getattr__(name: str):
    if name == "WAPipeline":
        return import_module(f"{__name__}.wa_pipeline").WAPipeline
    raise AttributeError(name)
