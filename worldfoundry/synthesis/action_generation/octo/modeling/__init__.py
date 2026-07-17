"""Inference architecture for the in-tree Octo policy.

Modules stay lazy so resolving a checkpoint component does not eagerly import
Orbax and the complete model loader.
"""
