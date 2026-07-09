"""Module for the VGGTOmega operator implementation."""

from __future__ import annotations

from .vggt_operator import VGGTOperator


class VGGTOmegaOperator(VGGTOperator):
    """Operator marker for the VGGT-Omega checkpoint variant."""

    MODEL_ID = "vggt-omega"
