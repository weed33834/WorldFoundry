"""Kairos API package.

Runtime entrypoints import concrete API modules explicitly so registry
registration stays deterministic and model-only imports do not pull in the
full inference pipeline.
"""

__all__ = ["KairosEmbodiedAPI"]


def __getattr__(name):
    if name == "KairosEmbodiedAPI":
        from .kairos_embodied_api import KairosEmbodiedAPI

        return KairosEmbodiedAPI
    raise AttributeError(name)
