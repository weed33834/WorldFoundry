"""ViT-based human anatomy abnormality detector."""

from .paths import config_path

__all__ = ["compute_abnormality", "config_path"]


def compute_abnormality(*args, **kwargs):
    from .inference import compute_abnormality as _compute_abnormality

    return _compute_abnormality(*args, **kwargs)
