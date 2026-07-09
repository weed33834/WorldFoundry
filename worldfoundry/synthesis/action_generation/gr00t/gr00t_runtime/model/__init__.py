from __future__ import annotations

__all__ = ["Gr00tN1d7", "Gr00tN1d7Processor"]


def __getattr__(name: str):
    if name == "Gr00tN1d7":
        from .gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

        return Gr00tN1d7
    if name == "Gr00tN1d7Processor":
        from .gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

        return Gr00tN1d7Processor
    raise AttributeError(name)
