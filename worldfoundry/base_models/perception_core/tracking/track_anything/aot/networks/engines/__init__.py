"""Module for base_models -> three_dimensions -> optical_flow -> track_anything -> aot -> networks -> engines -> __init__.py functionality."""

from .aot_engine import AOTEngine, AOTInferEngine
from .deaot_engine import DeAOTEngine, DeAOTInferEngine


def build_engine(name, phase='train', **kwargs):
    """Build engine.

    Args:
        name: The name.
        phase: The phase.
    """
    if name == 'aotengine':
        if phase == 'train':
            return AOTEngine(**kwargs)
        elif phase == 'eval':
            return AOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    elif name == 'deaotengine':
        if phase == 'train':
            return DeAOTEngine(**kwargs)
        elif phase == 'eval':
            return DeAOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
