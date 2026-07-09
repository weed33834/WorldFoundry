"""
Package for Gen3C synthesis utilities.

This __init__.py file serves to expose the Gen3CSynthesis class from the
.gen3c_synthesis submodule, making it directly importable from the top-level package.
"""
from .gen3c_synthesis import Gen3CSynthesis

# Defines the public API for the package when a client uses 'from package import *'.
__all__ = ["Gen3CSynthesis"]