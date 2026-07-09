"""Instance anomaly detection inference shared by video benchmark runners."""

__all__ = ["compute_anomaly"]


def compute_anomaly(*args, **kwargs):
    from .inference import compute_anomaly as _compute_anomaly

    return _compute_anomaly(*args, **kwargs)
