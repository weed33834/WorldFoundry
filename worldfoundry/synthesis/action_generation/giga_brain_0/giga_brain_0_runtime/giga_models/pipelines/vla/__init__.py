from importlib import import_module

__all__ = ["GigaBrain0Pipeline"]


def __getattr__(name: str):
    if name == "GigaBrain0Pipeline":
        return import_module(f"{__name__}.giga_brain_0").GigaBrain0Pipeline
    raise AttributeError(name)
