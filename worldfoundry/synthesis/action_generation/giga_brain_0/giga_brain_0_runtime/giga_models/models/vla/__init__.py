from importlib import import_module

__all__ = ["GigaBrain0Policy"]


def __getattr__(name: str):
    if name == "GigaBrain0Policy":
        return import_module(f"{__name__}.giga_brain_0").GigaBrain0Policy
    raise AttributeError(name)
