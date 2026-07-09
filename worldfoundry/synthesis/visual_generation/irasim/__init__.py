"""This module serves as a top-level entry point or re-exporter for the IRASimSynthesis class.

It simplifies access to the core IRASimulation synthesis functionality by making
`IRASimSynthesis` directly available when importing from the parent package.
"""
from .irasim_synthesis import IRASimSynthesis

__all__ = ["IRASimSynthesis"]