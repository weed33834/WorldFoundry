"""Infer-only vendored Prismatic package for WorldFoundry runtimes.

Keep package import lightweight. Import model loading helpers from their concrete
modules so action inference does not eagerly import RLDS/dlimp training stacks.
"""

__all__: list[str] = []
