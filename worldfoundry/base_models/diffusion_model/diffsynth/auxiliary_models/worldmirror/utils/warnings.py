"""
Wrapper utilities for warnings.
"""

import warnings
from functools import wraps


def suppress_traceback(fn):
    """Suppress traceback.

    Args:
        fn: The fn.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        """Wrapper."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            e.__traceback__ = e.__traceback__.tb_next.tb_next
            raise

    return wrapper


class no_warnings:
    """No warnings implementation."""
    def __init__(self, action: str = "ignore", **kwargs):
        """Init.

        Args:
            action: The action.
        """
        self.action = action
        self.filter_kwargs = kwargs

    def __call__(self, fn):
        """Call.

        Args:
            fn: The fn.
        """
        @wraps(fn)
        def wrapper(*args, **kwargs):
            """Wrapper."""
            with warnings.catch_warnings():
                warnings.simplefilter(self.action, **self.filter_kwargs)
                return fn(*args, **kwargs)

        return wrapper

    def __enter__(self):
        """Enter."""
        self.warnings_manager = warnings.catch_warnings()
        self.warnings_manager.__enter__()
        warnings.simplefilter(self.action, **self.filter_kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit.

        Args:
            exc_type: The exc type.
            exc_val: The exc val.
            exc_tb: The exc tb.
        """
        self.warnings_manager.__exit__(exc_type, exc_val, exc_tb)
