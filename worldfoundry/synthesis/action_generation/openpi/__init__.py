"""Public OpenPI API with lazy inference dependency loading."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "OpenPISynthesis":
        from .openpi_synthesis import OpenPISynthesis

        return OpenPISynthesis
    if name in {"Pi0Synthesis", "Pi05Synthesis", "Pi0FastSynthesis"}:
        from .variants_synthesis import Pi0FastSynthesis, Pi0Synthesis, Pi05Synthesis

        return {
            "Pi0Synthesis": Pi0Synthesis,
            "Pi05Synthesis": Pi05Synthesis,
            "Pi0FastSynthesis": Pi0FastSynthesis,
        }[name]
    raise AttributeError(name)


__all__ = ["OpenPISynthesis", "Pi0FastSynthesis", "Pi0Synthesis", "Pi05Synthesis"]
