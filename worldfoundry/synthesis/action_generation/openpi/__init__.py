"""
This module serves as part of the public API for the current package,
exposing key components for direct import.

It primarily re-exports the `OpenPISynthesis` class, making it accessible
at the top level of the package for easier consumption by external modules.
"""

from .openpi_synthesis import OpenPISynthesis

# Defines the public API of this module, specifying which names will be
# imported when `from package import *` is used.
__all__ = ["OpenPISynthesis"]