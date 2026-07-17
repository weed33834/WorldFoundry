"""Small model interfaces shared by the in-tree MolmoBot inference modules."""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Sequence

import torch


class OLMoOutput(NamedTuple):
    """Backbone output needed by action inference.

    MolmoBot does not decode language, so ``logits`` is normally ``None``.  The
    action head consumes ``internal['layer_hidden_states']`` instead.
    """

    logits: Optional[torch.Tensor] = None
    attn_key_values: Optional[Sequence[tuple[torch.Tensor, torch.Tensor]]] = None
    hidden_states: Optional[Sequence[torch.Tensor]] = None
    internal: Optional[Dict[str, Sequence[torch.Tensor]]] = None


class ModelBase(torch.nn.Module):
    """Base type retained for configuration/type compatibility."""


__all__ = ["ModelBase", "OLMoOutput"]
