from importlib import import_module

__all__ = ["GigaBrain0Pipeline", "GigaBrain0Policy"]


def __getattr__(name: str):
    if name == "GigaBrain0Policy":
        return import_module(f"{__name__}.models.vla.giga_brain_0").GigaBrain0Policy
    if name == "GigaBrain0Pipeline":
        return import_module(f"{__name__}.pipelines.vla.giga_brain_0").GigaBrain0Pipeline
    raise AttributeError(name)
