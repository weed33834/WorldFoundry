from importlib import import_module

_EXPORTS = {
    "LoGeRRepresentation": ".loger_representation",
    "Pi3Representation": ".pi3_representation",
    "Pi3XRepresentation": ".pi3x_representation",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
