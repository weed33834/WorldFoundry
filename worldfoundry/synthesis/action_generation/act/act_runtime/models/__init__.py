from __future__ import annotations

from .detr_vae import build as build_vae


def build_ACT_model(args):
    """Build the ACT DETR-VAE policy architecture.

    Args:
        args: Namespace-like object with ACT model hyperparameters.
    """
    return build_vae(args)
